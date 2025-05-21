from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .encoderblock import TransformerEncoder, get_activation_fn


class AbLang(torch.nn.Module):
    """
    AbLang inspired by ESM-2's architecture.
    """
    
    def __init__(
        self,
        vocab_size,
        hidden_embed_size,
        n_attn_heads,
        n_encoder_blocks,
        padding_tkn,
        mask_tkn,
        layer_norm_eps: float = 1e-12,
        a_fn: str = "gelu",
        dropout: float = 0.0, 
        use_tkn_dropout: bool = False,
        use_moe: bool = False,
    ):
        super().__init__()
                
        self.AbRep = AbRep(
            vocab_size,
            hidden_embed_size,
            n_attn_heads,
            n_encoder_blocks,
            padding_tkn,
            mask_tkn,
            layer_norm_eps,
            a_fn,
            dropout, 
            use_tkn_dropout,
            use_moe,
        )       
        self.AbHead = AbHead(
            vocab_size,
            hidden_embed_size,
            self.AbRep.aa_embed_layer.weight,
            layer_norm_eps,
            a_fn,
        )
        
    def forward(self, tokens, return_attn_weights=False, return_rep_layers=[], return_aux_loss=False):
        
        representations = self.AbRep(tokens, return_attn_weights, return_rep_layers)
        
        if return_attn_weights:
            return representations.attention_weights
        
        elif return_rep_layers != []:
            return representations.many_hidden_states
        elif return_aux_loss:
            likelihoods = self.AbHead(representations.last_hidden_states)
            return likelihoods, representations.all_aux_loss
        else:
            likelihoods = self.AbHead(representations.last_hidden_states)
            return likelihoods
    
    def get_aa_embeddings(self):
        "Extracts the trained aa_embeddings."
        return self.AbRep.aa_embed_layer

    
class AbRep(torch.nn.Module):
    """
    AbRep (antibody representations), takes the tokenized sequence and create hidden_embed (representations).
    """
    
    def __init__(
        self, 
        vocab_size,
        hidden_embed_size,
        n_attn_heads,
        n_encoder_blocks,
        padding_tkn,
        mask_tkn,
        layer_norm_eps: float = 1e-12,
        a_fn: str = "gelu",
        dropout: float = 0.1, 
        use_tkn_dropout: bool = False,
        use_moe: bool = False,
    ):
        super().__init__()
        self.padding_tkn = padding_tkn
        self.mask_tkn = mask_tkn
        self.use_tkn_dropout = use_tkn_dropout
        
        self.aa_embed_layer = nn.Embedding(
            vocab_size, 
            hidden_embed_size, 
            padding_idx=padding_tkn,
        )   
        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoder(
                hidden_embed_size,
                n_attn_heads,
                attn_dropout = dropout,
                layer_norm_eps = layer_norm_eps,
                a_fn = a_fn,
                use_moe = use_moe,
            ) for _ in range(n_encoder_blocks)]
        )
        self.layer_norm_after_encoder_blocks = nn.LayerNorm(hidden_embed_size, eps=layer_norm_eps)
        
    def token_dropout(hidden_embed, tokens, padding_mask):
        
        hidden_embed.masked_fill_((tokens == self.mask_tkn).unsqueeze(-1), 0.0)
        # x: B x T x C
        mask_ratio_train = 0.15 * 0.8
        src_lengths = (~padding_mask).sum(-1)
        mask_ratio_observed = (tokens == self.mask_tkn).sum(-1).to(x.dtype) / src_lengths
        hidden_embed = hidden_embed * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
        
        return hidden_embed
        
    def forward(self, 
                tokens, 
                return_attn_weights=False, 
                return_rep_layers=[],
               ):
        
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_tkn)

        hidden_embed = self.aa_embed_layer(tokens)
        
        if self.use_tkn_dropout:
            hidden_embed = token_dropout(hidden_embed, tokens, padding_mask)        
        
        return_rep_layers = set(return_rep_layers)
        rep_layers = {}
        if 0 in return_rep_layers: rep_layers[0] = hidden_embed
            
        all_attn_weights = []
        all_aux_loss = 0
        
        for n_layer, encoder_block in enumerate(self.encoder_blocks):
            hidden_embed, attn_weights, aux_loss = encoder_block(hidden_embed, padding_mask, return_attn_weights)
            all_aux_loss += aux_loss
            
            if (n_layer + 1) in return_rep_layers: 
                rep_layers[n_layer + 1] = hidden_embed
            
            if return_attn_weights: 
                all_attn_weights.append(attn_weights)
           
        hidden_embed = self.layer_norm_after_encoder_blocks(hidden_embed)

        return DataAbRep(last_hidden_states=hidden_embed, many_hidden_states=rep_layers, attention_weights=all_attn_weights, all_aux_loss=all_aux_loss)
    

class AbHead(torch.nn.Module):
    """
    AbHead (antibody head model), creates amino acid probabilities for each position based on the hidden_embed (representations).
    """
    
    def __init__(
        self, 
        vocab_size,
        hidden_embed_size,
        weights,
        layer_norm_eps: float = 1e-12,
        a_fn: str = "gelu",
    ):
        super().__init__()
        
        activation_fn, scale = get_activation_fn(a_fn)
        
        self.ff = torch.nn.Sequential(
            nn.Linear(hidden_embed_size, hidden_embed_size * scale),
            activation_fn(),
            nn.LayerNorm(hidden_embed_size, eps=layer_norm_eps),
        )

        self.weights = weights
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_embed):
        
        hidden_embed = self.ff(hidden_embed)
        logits = F.linear(hidden_embed, self.weights) + self.bias
        
        return logits


    
@dataclass
class DataAbRep():
    """
    Dataclass used to store AbRep output.
    """

    last_hidden_states: torch.FloatTensor
    many_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attention_weights: Optional[Tuple[torch.FloatTensor]] = None
    all_aux_loss: torch.FloatTensor = 0


class AblangWrapper(torch.nn.Module):
  def __init__(self, ablang_model: AbLang, hparams):
    super().__init__()
    self.ablang_model = ablang_model
    
    self.vocab_size = hparams.vocab_size
    self.mask_tkn_id = hparams.mask_tkn 
    self.pad_tkn_id = hparams.pad_tkn
    # Assuming start_tkn is in hparams, similar to LossAndPerplexity context
    # If not, this might need adjustment (e.g., use tokenizer.start_token_id or hardcode 22)
    self.start_tkn_id = hparams.start_tkn 

    self.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

  def forward(self, tokens, return_attn_weights=False, return_rep_layers=[], return_aux_loss=False):
    # tokens shape: (batch_size, seq_len)
    # This method calculates loss based on the provided 'calculate_perplexity_slow' template.

    batch_size = tokens.size(0)
    seq_len = tokens.size(-1)
    device = tokens.device

    if batch_size == 0 or seq_len == 0:
      # Handle empty input: return a zero loss.
      # Set requires_grad based on model's training state.
      return torch.tensor(0.0, device=device, requires_grad=self.ablang_model.training if hasattr(self.ablang_model, 'training') else False)

    # As per template: `repeat_tokenized_seqs = tokenized_seqs.repeat(tokenized_seqs.size(-1), 1)`
    # This creates `seq_len` blocks, each block being a copy of the original `tokens` batch.
    # Shape of repeat_tokens: (batch_size * seq_len, seq_len)
    repeat_tokens = tokens.repeat(seq_len, 1)

    # As per template: `diagonal_mask = torch.ones(tokenized_seqs.size(-1)-1).diag(1).repeat(tokenized_seqs.size(0), 1)`
    # 1. Create base for diagonal: torch.ones(seq_len - 1)
    diag_base_values = torch.ones(max(0, seq_len - 1), device=device)
    # 2. Create S x S matrix with 1s on the first super-diagonal:
    diagonal_mask_matrix = torch.diag(diag_base_values, diagonal=1) # Shape: (seq_len, seq_len)
    
    # 3. Repeat this S x S mask matrix B times (batch_size times) along the 0-dim.
    # Shape of expanded_diagonal_mask: (batch_size * seq_len, seq_len)
    expanded_diagonal_mask = diagonal_mask_matrix.repeat(batch_size, 1)

    # Mask tokens based on the expanded_diagonal_mask.
    # `self.mask_tkn_id` corresponds to `self.mask_tkn` in the template.
    masked_tokens = repeat_tokens.masked_fill(expanded_diagonal_mask == 1, self.mask_tkn_id)

    # Prepare labels for loss calculation.
    # Labels are the original tokens at positions where the mask was applied. Other positions are -100.
    labels = repeat_tokens.clone()
    labels[expanded_diagonal_mask != 1] = -100 # Set to -100 where no mask is applied

    # Further filter labels: ignore specific token IDs (PAD, START, MASK tokens themselves)
    # Template uses: (repeat_tokenized_seqs!=22) * (repeat_tokenized_seqs!=21) * (repeat_tokenized_seqs!=25)
    # These correspond to START_TKN, PAD_TKN, MASK_TKN.
    condition_pad = (repeat_tokens == self.pad_tkn_id)
    condition_start = (repeat_tokens == self.start_tkn_id)
    condition_mask_itself = (repeat_tokens == self.mask_tkn_id) # If original token was MASK
    
    ignore_condition = condition_pad | condition_start | condition_mask_itself
    labels[ignore_condition] = -100 # Set to -100 for these specific tokens

    # Get logits from the AbLang model.
    raw_model_output = self.ablang_model(
      masked_tokens,
      return_attn_weights=return_attn_weights,
      return_rep_layers=return_rep_layers,
      return_aux_loss=return_aux_loss  # Important for MoE models
    )

    logits = raw_model_output
    moe_aux_loss = None

    if isinstance(raw_model_output, tuple):
      logits = raw_model_output[0] # Logits are typically the first element.
      # Check for MoE auxiliary loss if requested and applicable.
      # Assumes AbLang model has AbRep.use_moe attribute.
      if return_aux_loss and \
         hasattr(self.ablang_model, 'AbRep') and \
         hasattr(self.ablang_model.AbRep, 'use_moe') and \
         self.ablang_model.AbRep.use_moe:
        # Assuming aux_loss, if present, is the second element and a scalar tensor.
        # This matches common patterns where AbLang.forward might return (logits, aux_loss, ...).
        if len(raw_model_output) > 1 and \
           torch.is_tensor(raw_model_output[1]) and \
           raw_model_output[1].ndim == 0: # MoE aux_loss is typically scalar.
          moe_aux_loss = raw_model_output[1]

    # Calculate CrossEntropyLoss.
    # logits shape: ( (B*S), S, vocab_size )
    # labels shape: ( (B*S), S )
    # Reshape for CrossEntropyLoss:
    # logits -> ( (B*S)*S, vocab_size )
    # labels -> ( (B*S)*S )
    main_loss = self.CrossEntropyLoss(logits.reshape(-1, self.vocab_size), labels.reshape(-1))
    
    total_loss = main_loss
    if moe_aux_loss is not None:
      total_loss = total_loss + moe_aux_loss # Add MoE auxiliary loss if present.
    
    return total_loss


