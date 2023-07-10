import jax.numpy as jnp
from jax.lax import cond
import flax.linen as nn

# --------------------------------------------
# | A hand crafted axial transformer in flax |
# --------------------------------------------

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
    """RoPE is weird, because it biases the later tokens. This should be considered if using this seriously."""
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
    """Axial attention
    -----------
    Works by treating the tertiary dim as a batch layer, which is differentiable."""
    config: dict=None

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_heads
        self.head_dim = self.config.head_dim

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads}).")

        self.pos = PositionEmbeds()
        #proj outputs (h, w, n * d) - dim(3)
        self.q_proj = nn.Dense(self.num_heads * self.head_dim, False, self.config.dtype, name="q_proj")
        self.k_proj = nn.Dense(self.num_heads * self.head_dim, False, self.config.dtype, name="k_proj")
        self.v_proj = nn.Dense(self.num_heads * self.head_dim, False, self.config.dtype, name="v_proj")
        self.o_proj = nn.Dense(self.hidden_size, False, dtype=self.config.dtype, name="o_proj")

    def attn(self, layers, d, attn_mask, causal_mask=True): 
        """Compute attention weights on the width rows"""
        q, k, v = layers #Our sorting function packs these into a tuple
        attn_weights = jnp.einsum('...hqd,...hkd->...hqk', q, k) / jnp.sqrt(d)
        attn_weights = attn_weights + attn_mask
        attn_weights = nn.softmax(attn_weights, axis=-1)
        return jnp.einsum('...hqk,...hkd->...hqd', attn_weights, v)

    def shift_dims_y(self, q, k, v, h, w):
        """For calculating height attention: (h, w, dim) -> (w, head, h, dim)"""
        return (
            jnp.transpose(q, (1, 2, 0, 3)),
            jnp.transpose(k, (1, 2, 0, 3)),
            jnp.transpose(jnp.reshape(v, (h, w, self.num_heads, self.head_dim)), (1, 2, 0, 3)))

    def shift_dims_x(self, q, k, v, h, w):
        """For calculating width attention: (h, w, dim) -> (h, head, w, dim)"""
        return(
            jnp.transpose(jnp.reshape(q, (h, w, self.num_heads, self.head_dim)), (0, 2, 1, 3)),
            jnp.transpose(jnp.reshape(k, (h, w, self.num_heads, self.head_dim)), (0, 2, 1, 3)),
            jnp.transpose(jnp.reshape(v, (h, w, self.num_heads, self.head_dim)), (0, 2, 1, 3)))

    def __call__(self, hidden_state, attn_mask, position_ids):
        #attn_mask should come in as (h, 1, w, 1)
        h, w, d = hidden_state.shape

        #reshapes to (h, w, n, d) dim(4)
        q, k = self.pos(jnp.reshape(self.q_proj(hidden_state), (h, w, self.num_heads, self.head_dim)),
                        jnp.reshape(self.k_proj(hidden_state), (h, w, self.num_heads, self.head_dim)),
                            position_ids) #Adds RoPE
        v = self.v_proj(hidden_state)

        #Reshapes heads and runs axial attention
        w_weights = jnp.transpose(self.attn(self.shift_dims_x(q, k, v, h, w), d, attn_mask), (0, 2, 1, 3))
        h_weights = jnp.transpose(self.attn(self.shift_dims_y(q, k, v, h, w), d, jnp.transpose(attn_mask, (2, 1, 0, 3))), (2, 0, 1, 3))
        #And back to (h, w, n, d)
        
        #For debug only. Needs to be stripped away for jit compilation
        if h_weights.shape != (h, w, self.num_heads, self.head_dim):
            raise ValueError(f"`h_weights` should be of size {(h, w, self.num_heads, self.head_dim)}, but is"
                             f" {h_weights.shape}")
        if w_weights.shape != (h, w, self.num_heads, self.head_dim):
            raise ValueError(f"`w_weights` should be of size {(h, w, self.num_heads, self.head_dim)}, but is"
                             f" {w_weights.shape}")

        attn_output = self.o_proj( #Merges axial weights, and transforms them back into (h, w, d)
                        jnp.concatenate([
                            jnp.reshape(w_weights, (h, w, d)), #"W" should be first, as it's identical to treating h as a batch dimension.
                            jnp.reshape(h_weights, (h, w, d))],
                                axis=-1)) #o_proj reduces the final dim back to config.hidden_size for next layer
        return attn_output

class MLP(nn.Module):
    config: dict = None

    @nn.compact
    def __call__(self, hidden_state):
        gate = nn.Dense(self.config.intermediate_size, False, self.config.dtype, name="mlp_gate")(hidden_state)
        gate = nn.activation.silu(gate)
        hidden_state = nn.Dense(self.config.intermediate_size, False, self.config.dtype, name="mlp_in")(hidden_state)
        return nn.Dense(self.config.hidden_size, False, self.config.dtype, name="mlp_out")(gate * hidden_state)

class Decoder(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, hidden_state, attention_mask, position_ids):
        residual = hidden_state
        hidden_state = RMSNorm(self.config, name="input_norm")(hidden_state)
        hidden_state = Attention(self.config, name="attention")(hidden_state, attention_mask, position_ids)
        residual = hidden_state = residual + hidden_state

        hidden_state = RMSNorm(self.config, name="output_norm")(hidden_state)
        #Bottleneck merges here, once we're ready to implement it
        hidden_state = MLP(self.config, name="decoder_mlp")(hidden_state)
        hidden_state = residual + hidden_state

        return hidden_state, attention_mask, position_ids #Decoder MUST return all variables that it ingests to work w/ nn.Sequential
    
class SequenceClassificationModel(nn.Module):
    """cv head"""
    config: dict=None
    patch_shape = (config.patch_size, config.patch_size)
    @nn.compact
    def __call__(self, image, attn_mask, position_ids):
        """attn_mask needs to be passed with padding. with target and source."""
        image_embeds = nn.Conv(features=self.config.hidden_size, kernel_size=self.patch_shape, strides=self.patch_shape, padding='VALID', name="patch_emb")(image)
        hidden_state, _, _ = nn.Sequential([Decoder(self.config, name=f"head_{i}")
                                        for i, _ in enumerate(range(self.config.head_layers))])(image_embeds, attn_mask, position_ids)
        logits = nn.Dense(self.config.vocab_size, False, self.config.dtype, name="lm_logits")(hidden_state) #Generate probability distribution over vocabulary
        return logits
    