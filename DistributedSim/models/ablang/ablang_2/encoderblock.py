import torch
import math
from torch import nn
import torch.nn.functional as F
import einops
from rotary_embedding_torch import RotaryEmbedding
from mixture_of_experts import MoE

class TransformerEncoder(torch.nn.Module):
    """
    Single Transformer Encoder.
    
    """
    def __init__(
        self, 
        hidden_embed_size,
        n_attn_heads,
        attn_dropout: float = 0.0,
        layer_norm_eps: float = 1e-05,
        a_fn: str = "gelu",
        use_moe: bool = False,
    ):
        super().__init__()
        
        assert hidden_embed_size % n_attn_heads == 0, \
        "Embedding dimension must be devidable with the number of heads." 
        
        self.multihead_attention = MultiHeadAttention(
            embed_dim = hidden_embed_size, 
            num_heads = n_attn_heads,
            attention_dropout_prob = attn_dropout
        )
        
        activation_fn, scale = get_activation_fn(a_fn)
        
        self.use_moe = use_moe
        if use_moe:        
            self.intermediate_layer = MoE(
                dim = hidden_embed_size,
                num_experts = 16,                             # increase the experts (# parameters) of your model without increasing computation
                hidden_dim = hidden_embed_size * 4 * scale,           # size of hidden dimension in each expert, defaults to 4 * dimension
                activation = activation_fn,                    # use your preferred activation, will default to GELU
                second_policy_train = 'random',               # in top_2 gating, policy for whether to use a second-place expert
                second_policy_eval = 'random',                # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
                second_threshold_train = 0.2,
                second_threshold_eval = 0.2,
                capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
                capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
                loss_coef = 1e-2                # multiplier on the auxiliary expert balancing auxiliary loss
            )
        else:
            self.intermediate_layer = torch.nn.Sequential(
                torch.nn.Linear(hidden_embed_size, hidden_embed_size * 4 * scale),
                activation_fn(),
                torch.nn.Linear(hidden_embed_size * 4, hidden_embed_size),
            )
        
        self.pre_attn_layer_norm = torch.nn.LayerNorm(hidden_embed_size, eps=layer_norm_eps)
        self.final_layer_norm = torch.nn.LayerNorm(hidden_embed_size, eps=layer_norm_eps)
        
    def forward(self, hidden_embed, attn_mask=None, return_attn_weights: bool = False):
        
        residual = hidden_embed
        hidden_embed = self.pre_attn_layer_norm(hidden_embed)
        hidden_embed, attn_weights = self.multihead_attention(hidden_embed, 
                                                              attn_mask=attn_mask, 
                                                              return_attn_weights=return_attn_weights
                                                             )
        hidden_embed = residual + hidden_embed
        
        residual = hidden_embed
        hidden_embed = self.final_layer_norm(hidden_embed)
        if self.use_moe:
            hidden_embed, aux_loss = self.intermediate_layer(hidden_embed)
            hidden_embed = residual + hidden_embed
            return hidden_embed, attn_weights, aux_loss
        else:
            hidden_embed = self.intermediate_layer(hidden_embed)
            hidden_embed = residual + hidden_embed
            return hidden_embed, attn_weights, 0    
    
class MultiHeadAttention(torch.nn.Module):

    def __init__(
        self, 
        embed_dim,
        num_heads, 
        attention_dropout_prob: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            
        self.reset_parameters()
        
        self.rotary_emb = RotaryEmbedding(dim = self.head_dim)
        
    def reset_parameters(self):
        
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        
    def attention(self, q, k, v, attn_mask=None):
        
        # scaled_dot_product_attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) 
        attn_weights = attn_weights / math.sqrt(self.head_dim)
                
        if attn_mask is not None:
            attn_mask = einops.rearrange(attn_mask, 'b_size (h1 h2 seq_len) -> b_size h1 h2 seq_len', h1=1, h2=1)
            
            attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn = self.attention_dropout(attn_weights)
        attn = torch.matmul(attn, v)
        return attn, attn_weights       

    def forward(self, x, attn_mask=None, return_attn_weights: bool = False):
        
        batch_size, seq_len, embed_dim = x.size()
        
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q *= self.scaling ## WHY DO WE DO THIS?????
        
        q = q.contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [n_batch, n_heads, seq_len, head_dim]
        k = k.contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [n_batch, n_heads, seq_len, head_dim]
        v = v.contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [n_batch, n_heads, seq_len, head_dim]
        
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        
        # Determine value outputs
        attn, attn_weights = self.attention(q, k, v, attn_mask=attn_mask) # attn_weights [n_batch, n_heads, seq_len (target), seq_len (source)]
    
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)       
        
        attn = self.out_proj(attn)

        if return_attn_weights:
            return attn, attn_weights
        else:
            return attn, None
        
        
        
class SwiGLU(torch.nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1) 
        return F.silu(gate) * x
        
def get_activation_fn(a_fn):
    
    if a_fn == "gelu":
        return torch.nn.GELU, 1
    
    elif a_fn == "swiglu":
        return SwiGLU, 2
    