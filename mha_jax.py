import jax
import jax.numpy as jnp
from flax import nnx
import optax

class MultiHeadAttention(nnx.Module):
  def __init__(self, d_model, num_heads, *, rngs):
    assert d_model % num_heads == 0
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_head = d_model // num_heads

    self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
    self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
    self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
    self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)

  def __call__(self, x):
    B, T, D = x.shape
    
    q = self.q_proj(x).reshape(B, T, self.num_heads, self.d_head).swapaxes(1, 2)
    k = self.k_proj(x).reshape(B, T, self.num_heads, self.d_head).swapaxes(1, 2)
    v = self.v_proj(x).reshape(B, T, self.num_heads, self.d_head).swapaxes(1, 2)
    
    attn_scores = (q @ k.swapaxes(-1, -2)) / (self.d_head**0.5)
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    
    combined = (attn_weights @ v).swapaxes(1,2).reshape(B, T, D)
    
    return self.out_proj(combined)