from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from .encoderblock import TransformerEncoder, get_activation_fn


@dataclass
class AbLangConfig:
    vocab_size: int
    hidden_embed_size: int
    n_attn_heads: int
    n_encoder_blocks: int
    padding_tkn: int
    mask_tkn: int
    start_tkn: int  # For AblangWrapper
    layer_norm_eps: float = 1e-12
    a_fn: str = "gelu"
    dropout: float = 0.0  # Corresponds to attn_dropout in TransformerEncoder via AbRep
    use_tkn_dropout: bool = False
    use_moe: bool = False


class AbLang(torch.nn.Module):
    """
    AbLang inspired by ESM-2's architecture.
    """

    def __init__(self, config: AbLangConfig):
        super().__init__()

        self.AbRep = AbRep(config)
        self.AbHead = AbHead(
            config.vocab_size,
            config.hidden_embed_size,
            self.AbRep.aa_embed_layer.weight,
            config.layer_norm_eps,
            config.a_fn,
        )
        self.vocab_size = config.vocab_size
        self.criterion = CrossEntropyLoss()

    def forward(
        self,
        batch,
        return_attn_weights=False,
        return_rep_layers=[],
        return_aux_loss=False,
    ):
        tokens, labels = batch


        representations = self.AbRep(tokens, return_attn_weights, return_rep_layers)

        if return_attn_weights:
            return representations.attention_weights

        elif return_rep_layers != []:
            return representations.many_hidden_states
        elif (
            return_aux_loss
        ):  # User of AbLang needs to be aware if MoE is used to interpret this
            likelihoods = self.AbHead(representations.last_hidden_states)
            return likelihoods, representations.all_aux_loss
        else:
            likelihoods = self.AbHead(representations.last_hidden_states)
            loss = self.criterion(
                likelihoods.view(-1, self.vocab_size), labels.view(-1)
            )
            return loss

    def get_aa_embeddings(self):
        "Extracts the trained aa_embeddings."
        return self.AbRep.aa_embed_layer

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the parameters of the embedding layer are not counted.
        This is similar to how ESM-2 reports model sizes.
        """
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            # Check if AbRep and its aa_embed_layer exist, which they should based on __init__
            if hasattr(self, "AbRep") and hasattr(self.AbRep, "aa_embed_layer"):
                n_params -= self.AbRep.aa_embed_layer.weight.numel()
        return n_params


class AbRep(torch.nn.Module):
    """
    AbRep (antibody representations), takes the tokenized sequence and create hidden_embed (representations).
    """

    def __init__(self, config: AbLangConfig):
        super().__init__()
        self.padding_tkn = config.padding_tkn
        self.mask_tkn = config.mask_tkn
        self.use_tkn_dropout = config.use_tkn_dropout
        self.use_moe = config.use_moe  # Stored for AblangWrapper to check

        self.aa_embed_layer = nn.Embedding(
            config.vocab_size,
            config.hidden_embed_size,
            padding_idx=config.padding_tkn,
        )
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoder(
                    config.hidden_embed_size,
                    config.n_attn_heads,
                    attn_dropout=config.dropout,  # This is AbLangConfig.dropout
                    layer_norm_eps=config.layer_norm_eps,
                    a_fn=config.a_fn,
                    use_moe=config.use_moe,
                )
                for _ in range(config.n_encoder_blocks)
            ]
        )
        self.layer_norm_after_encoder_blocks = nn.LayerNorm(
            config.hidden_embed_size, eps=config.layer_norm_eps
        )

    def token_dropout(self, hidden_embed, tokens, padding_mask):  # Added self
        hidden_embed.masked_fill_((tokens == self.mask_tkn).unsqueeze(-1), 0.0)
        # x: B x T x C
        mask_ratio_train = 0.15 * 0.8
        src_lengths = (~padding_mask).sum(-1)
        # Avoid division by zero if src_lengths contains zeros (e.g. empty sequences)
        # Add a small epsilon for stability, or handle zero lengths explicitly.
        # For now, keeping original logic which might lead to NaN/inf if src_lengths is 0.
        mask_ratio_observed = (tokens == self.mask_tkn).sum(-1).to(
            hidden_embed.dtype
        ) / src_lengths
        hidden_embed = (
            hidden_embed
            * (1 - mask_ratio_train)
            / (1 - mask_ratio_observed)[:, None, None]
        )

        return hidden_embed

    def forward(
        self,
        tokens,
        return_attn_weights=False,
        return_rep_layers=[],
    ):

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_tkn)

        hidden_embed = self.aa_embed_layer(tokens)
        # print(f"[AbRep.forward] hidden_embed (after embedding) shape: {hidden_embed.shape}, dtype: {hidden_embed.dtype}")
        # if hidden_embed.is_mps: # Calculate size only if it's on MPS to avoid errors if not yet moved
        #     element_size = hidden_embed.element_size() if hidden_embed.element_size() is not None else 4 # Default to 4 bytes if None
        #     print(f"  Estimated size of hidden_embed: {hidden_embed.numel() * element_size / (1024**3):.2f} GB")
        # else:
        #     print(f"  hidden_embed is on device: {hidden_embed.device}")

        if self.use_tkn_dropout:
            hidden_embed = self.token_dropout(
                hidden_embed, tokens, padding_mask
            )  # Call with self.

        current_return_rep_layers = set(
            return_rep_layers
        )  # Use a different name to avoid modifying input list
        rep_layers = {}
        if 0 in current_return_rep_layers:
            rep_layers[0] = hidden_embed

        all_attn_weights = []
        all_aux_loss = 0  # Original initialization

        for n_layer, encoder_block in enumerate(self.encoder_blocks):
            hidden_embed, attn_weights, aux_loss = encoder_block(
                hidden_embed, padding_mask, return_attn_weights
            )
            all_aux_loss += aux_loss  # aux_loss is tensor(0.0) or MoE loss

            if (n_layer + 1) in current_return_rep_layers:
                rep_layers[n_layer + 1] = hidden_embed

            if return_attn_weights:
                all_attn_weights.append(attn_weights)

        hidden_embed = self.layer_norm_after_encoder_blocks(hidden_embed)

        return DataAbRep(
            last_hidden_states=hidden_embed,
            many_hidden_states=rep_layers,
            attention_weights=all_attn_weights,
            all_aux_loss=all_aux_loss,
        )


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
            nn.Linear(
                hidden_embed_size, int(hidden_embed_size * scale)
            ),  # Ensure scale result is int
            activation_fn(),
            nn.LayerNorm(hidden_embed_size, eps=layer_norm_eps),  # Original LayerNorm
        )

        self.weights = weights
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_embed):
        hidden_embed = self.ff(hidden_embed)
        logits = F.linear(hidden_embed, self.weights) + self.bias

        return logits


@dataclass
class DataAbRep:
    """
    Dataclass used to store AbRep output.
    """

    last_hidden_states: torch.FloatTensor
    many_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # Original type hint
    attention_weights: Optional[Tuple[torch.FloatTensor]] = None  # Original type hint
    all_aux_loss: torch.FloatTensor = 0  # Original default


class AblangWrapper(torch.nn.Module):
    def __init__(
        self, config: AbLangConfig
    ):  # Changed: ablang_model, hparams -> config
        super().__init__()
        self.ablang_model = AbLang(config)  # Changed: Instantiate AbLang model here

        self.vocab_size = config.vocab_size  # Changed: from hparams to config
        self.mask_tkn_id = config.mask_tkn  # Changed: from hparams to config
        self.pad_tkn_id = config.padding_tkn  # Changed: from hparams to config
        self.start_tkn_id = config.start_tkn  # Changed: from hparams to config

        # Standard CrossEntropyLoss, ignores index -100 by default
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the parameters of the embedding layer are not counted.
        This is similar to how ESM-2 reports model sizes.
        """
        n_params = sum(
            p.numel() for p in self.ablang_model.parameters() if p.requires_grad
        )
        if non_embedding:
            # Assuming aa_embed_layer is the name of the embedding layer in AbRep
            # and AbRep is accessible as self.ablang_model.AbRep
            if hasattr(self.ablang_model, "AbRep") and hasattr(
                self.ablang_model.AbRep, "aa_embed_layer"
            ):
                n_params -= self.ablang_model.AbRep.aa_embed_layer.weight.numel()
        return n_params

    def fast_collater(self, tokens: torch.Tensor, mask_fraction: float = 0.15):
        """
        Prepares batch for Masked Language Modeling (MLM).
        Randomly masks a fraction of tokens and creates corresponding labels.
        - Input tokens are modified by replacing some tokens with `self.mask_tkn_id`.
        - Labels are the original token IDs for masked positions, and -100 elsewhere.
        """
        device = tokens.device
        input_ids = tokens.clone()
        labels = torch.full_like(
            tokens, -100, device=device
        )  # Initialize with ignore_index

        # Identify positions eligible for masking (e.g., not PAD or START tokens)
        can_be_masked = (tokens != self.pad_tkn_id) & (tokens != self.start_tkn_id)

        # Determine actual masked positions based on mask_fraction
        # Generate random probabilities for each token
        probabilities = torch.rand(tokens.shape, device=device)
        # Select tokens to mask where probability < mask_fraction and token can_be_masked
        masked_positions = (probabilities < mask_fraction) & can_be_masked

        # Apply masking
        input_ids[masked_positions] = self.mask_tkn_id
        labels[masked_positions] = tokens[
            masked_positions
        ]  # Set labels to original tokens for masked positions

        return {"input": input_ids, "labels": labels}

    def forward(
        self,
        tokens,
        return_attn_weights=False,
        return_rep_layers=[],
        return_aux_loss=False,
    ):
        # tokens shape: (batch_size, seq_len)
        # This method implements a fast perplexity calculation inspired by ESM-2, returning loss.
        # print(f"[AblangWrapper.forward.fast] Input tokens shape: {tokens.shape}, dtype: {tokens.dtype}, device: {tokens.device}")

        batch_size = tokens.size(0)
        seq_len = tokens.size(
            1
        )  # Corrected: tokens.size(-1) is also fine, but 1 is conventional for seq_len
        device = tokens.device

        if batch_size == 0 or seq_len == 0:
            print("[AblangWrapper.forward.fast] Empty input, returning zero loss.")
            return torch.tensor(
                0.0,
                device=device,
                requires_grad=(
                    self.ablang_model.training
                    if hasattr(self.ablang_model, "training")
                    else False
                ),
            )

        # Prepare MLM inputs and labels
        collated_batch = self.fast_collater(tokens)
        input_ids = collated_batch["input"]
        target_labels = collated_batch["labels"]

        # print(f"[AblangWrapper.forward.fast] MLM input_ids shape: {input_ids.shape}, target_labels shape: {target_labels.shape}")

        # Get raw model output (logits, or (logits, aux_loss) if MoE)
        raw_model_output = self.ablang_model(
            input_ids,
            return_attn_weights=return_attn_weights,  # Passed through, but not used by this loss calculation
            return_rep_layers=return_rep_layers,  # Passed through, but not used by this loss calculation
            return_aux_loss=return_aux_loss,  # Crucial for MoE models
        )

        logits = None
        moe_aux_loss = None

        if isinstance(raw_model_output, tuple):
            # Expected if return_aux_loss is True and model returns aux_loss (e.g., MoE)
            logits = raw_model_output[0]
            if (
                return_aux_loss
                and len(raw_model_output) > 1
                and torch.is_tensor(raw_model_output[1])
                and raw_model_output[1].ndim == 0
            ):
                moe_aux_loss = raw_model_output[1]
        #     print(f"[AblangWrapper.forward.fast] MoE auxiliary loss found: {moe_aux_loss.item()}")
        #   print(f"[AblangWrapper.forward.fast] Logits extracted from tuple, shape: {logits.shape}")
        elif torch.is_tensor(raw_model_output):
            # Expected if model returns only logits
            logits = raw_model_output
        #   print(f"[AblangWrapper.forward.fast] Logits (direct tensor output), shape: {logits.shape}")
        else:
            # Handle unexpected output type (e.g. DataAbRep if only_ab_rep=True in AbLang)
            # This wrapper assumes AbLang is configured to return logits for training.
            raise TypeError(
                f"Unexpected output type from ablang_model: {type(raw_model_output)}"
            )

        if logits is None:
            # print("[AblangWrapper.forward.fast] Logits tensor is None. This should not happen.")
            # Return a zero loss with requires_grad to avoid breaking training loop, though this indicates an issue.
            return torch.tensor(
                0.0,
                device=device,
                requires_grad=(
                    self.ablang_model.training
                    if hasattr(self.ablang_model, "training")
                    else False
                ),
            )

        # Calculate CrossEntropyLoss for MLM
        # logits shape: (batch_size, seq_len, vocab_size)
        # target_labels shape: (batch_size, seq_len)
        # Reshape for CrossEntropyLoss:
        # logits -> (batch_size * seq_len, vocab_size)
        # target_labels -> (batch_size * seq_len)
        main_loss = self.CrossEntropyLoss(
            logits.reshape(-1, self.vocab_size), target_labels.reshape(-1)
        )

        total_loss = main_loss
        if moe_aux_loss is not None:
            total_loss = total_loss + moe_aux_loss  # Add MoE auxiliary loss if present
        #   print(f"[AblangWrapper.forward.fast] Main loss: {main_loss.item()}, Total loss (with MoE): {total_loss.item()}")
        # else:
        #   print(f"[AblangWrapper.forward.fast] Total loss: {total_loss.item()}")

        return total_loss
