import jax.numpy as jnp
import flax.linen as nn

class RMSNorm(nn.Module):
    config: dict = None
    eps: float = 1e-6
    
    @nn.compact
    def __call__(self, hidden_states):
        variance = jnp.mean(hidden_states ** 2, axis=-1, keepdims=True)
        hidden_states = hidden_states * jnp.reciprocal(jnp.sqrt(variance + self.eps))
        weight = self.param('weight', nn.initializers.ones, (self.config.hidden_size,))
        return weight * hidden_states
    
class PositionEmbeds():
    """This is effectively just a helper class right now and doesn't need a state. We handle the heavy lifting in preprocessing.
    Since it's clunky to create a cache for the RoPE arrays inside a compiled jax.jit function,
    I advise to handle all tensor shape logic in a Preprocessor class instead."""
    def inverse_half(self, hidden_dims):
        """Rotates half the hidden dims of the input."""
        x1 = hidden_dims[..., : hidden_dims.shape[-1] // 2]
        x2 = hidden_dims[..., hidden_dims.shape[-1] // 2 :]
        return jnp.concatenate((-x2, x1), axis=-1)

    def __call__(self, q, k, position_ids):
        """Basic RoPE implementation. Assumes that position_ids carries the cached cos/sin arrays due to jax.jit simplicity"""
        cos, sin = position_ids
        q_embed = (q * cos) + (self.inverse_half(q) * sin)
        k_embed = (k * cos) + (self.inverse_half(k) * sin)
        return q_embed, k_embed

class Attention(nn.Module):
    config: dict=None

    def setup(self):
        """We would normally use @nn.compact here, but since the Attention class is the most important and most complicated,
        I've found it easier to use the basic pytorch notation for debugging. Then we can do things like use if statements for initialization."""
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_heads
        self.head_dim = self.hidden_size // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads}).")

        self.pos = PositionEmbeds()
        #proj outputs (b, s, n * d) - dim(3)
        self.q_proj = nn.Dense(self.num_heads * self.head_dim, False, self.config.dtype, name="q_proj")
        self.k_proj = nn.Dense(self.num_heads * self.head_dim, False, self.config.dtype, name="k_proj")
        self.v_proj = nn.Dense(self.num_heads * self.head_dim, False, self.config.dtype, name="v_proj")
        self.o_proj = nn.Dense(self.hidden_size, False, dtype=self.config.dtype, name="o_proj")

    def attn(self, layers, d, attn_mask): 
        """Compute attention weights on the seq dimension"""
        q, k, v = layers #Our sorting function packs these into a tuple, so we have to unpack them
        attn_weights = jnp.einsum('...hqd,...hkd->...hqk', q, k) / jnp.sqrt(d)

        #Create causal mask
        length = attn_mask.shape[2]
        mask = jnp.tri(length, length)
        mask = jnp.expand_dims(jnp.where(mask == 1, jnp.NINF, mask), (0, 1))
        mask = attn_mask + mask #Merge causal mask and attention mask

        attn_weights = attn_weights + mask
        attn_weights = jnp.exp(attn_weights - attn_weights.max(axis=-1, keepdims=True))
        attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)
        return jnp.einsum('...hqk,...hkd->...hqd', attn_weights, v)

    def shift_dims(self, q, k, v, b, s):
        """(b, s, d) -> (batch, num_heads, seq, dim)"""
        return(
            jnp.transpose(jnp.reshape(q, (b, s, self.num_heads, self.head_dim)), (0, 2, 1, 3)),
            jnp.transpose(jnp.reshape(k, (b, s, self.num_heads, self.head_dim)), (0, 2, 1, 3)),
            jnp.transpose(jnp.reshape(v, (b, s, self.num_heads, self.head_dim)), (0, 2, 1, 3)))

    def __call__(self, hidden_state, attn_mask, position_ids):
        #attn_mask should come in as (b, 1, seq, 1 for broadcasting)
        b, s, d = hidden_state.shape #(batch, seq, hidden_dim)

        #reshapes to (batch, seq, num_heads, head_dim),  dim(4)
        q, k = self.pos(self.q_proj(hidden_state), self.k_proj(hidden_state), position_ids) #Adds RoPE
        v = self.v_proj(hidden_state)

        hidden_state = jnp.transpose(self.attn(self.shift_dims(q, k, v, b, s), d, attn_mask), (0, 2, 1, 3))
        #And back to (b, s, n, d)
        
        #For debug only. Should be stripped away for jit compilation
        """if hidden_state.shape != (b, s, self.num_heads, self.head_dim):
            raise ValueError(f"`hidden_state` should be of size {(b, s, self.num_heads, self.head_dim)}, but is"
                             f" {hidden_state.shape}")"""

        return self.o_proj(jnp.reshape(hidden_state, (b, s, d))) #o_proj merges q, k, and v back to hidden dims

class MLP(nn.Module):
    config: dict = None

    @nn.compact
    def __call__(self, hidden_state):
        gate = nn.Dense(self.config.intermediate_size, False, self.config.dtype, name="mlp_gate")(hidden_state)
        gate = nn.activation.silu(gate)
        hidden_state = nn.Dense(self.config.intermediate_size, False, self.config.dtype, name="mlp_in")(hidden_state)
        return nn.Dense(self.config.hidden_size, False, self.config.dtype, name="mlp_out")(gate * hidden_state)

class Decoder(nn.Module):
    """Assembles all components into a single layer"""
    config: dict

    @nn.compact
    def __call__(self, hidden_state, attention_mask, position_ids):
        residual = hidden_state
        hidden_state = RMSNorm(self.config, name="input_norm")(hidden_state)
        hidden_state = Attention(self.config, name="attention")(hidden_state, attention_mask, position_ids)
        residual = hidden_state = residual + hidden_state

        hidden_state = RMSNorm(self.config, name="output_norm")(hidden_state)
        hidden_state = MLP(self.config, name="mlp")(hidden_state)
        hidden_state = residual + hidden_state

        return hidden_state, attention_mask, position_ids #Decoder MUST return all variables that it ingests to work w/ nn.Sequential

class Model(nn.Module):
    """Headless model"""
    config: dict=None

    @nn.compact
    def __call__(self, embeds, attn_mask, position_ids):
        #Layers need to output all variables to work w/ nn.sequential & jit
        hidden_state, _, _ = nn.Sequential([Decoder(self.config, name=f"layer_{i}")
                                        for i, _ in enumerate(range(self.config.layers))])(embeds, attn_mask, position_ids)
        return hidden_state
    
class LMHead(nn.Module):
    """Language modeling head. Uses causal mask for input prediction."""
    config: dict=None

    @nn.compact
    def __call__(self, input_ids, attn_mask, position_ids):
        """Due to Jax's pure function setup, we cache and calculate RoPE outside the model, and pass it as position_ids"""
        input_embeds = nn.Embed(self.config.vocab_size, self.config.hidden_size, self.config.dtype, name="text_embeds")(input_ids)
        hidden_state = Model(self.config)(input_embeds, attn_mask, position_ids)
        logits = nn.Dense(self.config.vocab_size, False, self.config.dtype, name="logits")(hidden_state) #Generate probability distribution over vocabulary
        return logits
