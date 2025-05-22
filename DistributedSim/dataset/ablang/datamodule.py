import os
import torch
import numpy as np
from torch.utils.data import Dataset


def mask_sequences(
    tkn_sequences,
    base_unmask,
    idx_corrupt,
    idx_leave,
    idx_mask,
    mask_tkn=23,
    pad_tkn=21,
):
    """
    Same as create_BERT_data, but also keeps start and stop.
    """

    masked_sequences = tkn_sequences.clone()
    # randomly changes idx_change in the data
    masked_sequences.scatter_(
        1,
        idx_corrupt,
        torch.randint(1, 21, masked_sequences.shape, device=masked_sequences.device),
    )
    masked_sequences.scatter_(1, idx_mask, mask_tkn)  # change idx_mask inputs to <mask>

    base_unmask.scatter_(1, idx_mask, 1)
    base_unmask.scatter_(1, idx_corrupt, 1)
    base_unmask.scatter_(1, idx_leave, 1)
    tkn_sequences[~base_unmask.long().bool()] = -100
    # This fixes potential errors in short/long masking
    tkn_sequences[(tkn_sequences == pad_tkn)] = -100

    return masked_sequences, tkn_sequences


def adjust_selection_mask(selection_mask, mask_num):

    idx = torch.arange(selection_mask.shape[1], 0, -1)
    indices = torch.argmax(selection_mask * idx, 1, keepdim=True)

    for test_idx in indices.reshape(-1):
        selection_mask[:, test_idx - mask_num - 1 : selection_mask.shape[1]] = 0

    return selection_mask


def get_indices(
    selection_mask,
    mask_num,
    p_corrupt=0.1,
    p_leave=0.1,
    cdr3_focus=1,
    mask_technique="shotgun",
):
    # allowed_mask[:, 106:118] *= cdr3_focus # Changes the chance of residues in the CDR3 getting masked. It\\'s 106 and 118 because the start token is present.

    if (mask_technique == "shotgun") or (mask_technique == "random"):
        indices = torch.multinomial(
            selection_mask, num_samples=mask_num, replacement=False
        )

    elif mask_technique == "span_long":
        selection_mask = adjust_selection_mask(selection_mask, mask_num)

        start_idx = torch.multinomial(
            selection_mask, num_samples=1, replacement=False
        ).repeat(1, mask_num)
        step_idx = torch.linspace(0, mask_num - 1, steps=mask_num, dtype=int).repeat(
            selection_mask.shape[0], 1
        )
        indices = start_idx + step_idx
        indices = indices.clamp(max=selection_mask.shape[1] - 1)

    elif mask_technique == "span_short":
        span_lens = np.random.choice([3, 4, 5], size=(5))
        span_sep_lens = (
            torch.normal(mean=15, std=6, size=(5,)).int().clamp(min=1, max=15)
        )

        start_idx = 0
        many_span_idx = []
        for span_len, span_sep_len in zip(span_lens, span_sep_lens):

            many_span_idx.append(
                torch.linspace(
                    start_idx, start_idx + span_len - 1, steps=span_len, dtype=int
                ).repeat(selection_mask.shape[0], 1)
            )
            start_idx += span_len + span_sep_len

        indices = torch.concatenate(many_span_idx, axis=1)
        indices = indices.clamp(max=selection_mask.shape[1] - 1)

    n_corrupt = max(int(indices.shape[1] * p_corrupt), 1)
    n_leave = max(int(indices.shape[1] * p_leave), 1)

    return torch.split(
        indices,
        split_size_or_sections=[
            n_corrupt,
            n_leave,
            max(indices.shape[-1] - (n_corrupt + n_leave), 0),
        ],
        dim=1,
    )


class AbDataset(Dataset):
    def __init__(
        self,
        file_path,
        tokenizer_instance,
        over_sample_data=0,
        is_train_data=False,
        block_size=None,
        pad_tkn=21,
        start_tkn=0,
        end_tkn=22,
        sep_tkn=25,
        mask_tkn=23,
        mask_percent=0.15,
        mask_variable=False,
        cdr3_focus=1,
        mask_technique="shotgun",
        change_percent=0.1,
        leave_percent=0.1,
    ):
        super().__init__()
        self.tokenizer = tokenizer_instance()
        self.over_sample_data = over_sample_data
        self.block_size = block_size
        self.data = self._load_data(file_path, is_train_data)

        self.pad_tkn = pad_tkn
        self.start_tkn = start_tkn
        self.end_tkn = end_tkn
        self.sep_tkn = sep_tkn
        self.mask_tkn = mask_tkn
        self.mask_percent = mask_percent
        self.mask_variable = mask_variable
        self.cdr3_focus = cdr3_focus
        self.mask_technique = mask_technique
        self.change_percent = change_percent
        self.leave_percent = leave_percent

    def _load_data(self, file_path, is_train_data):
        heavychain = []
        lightchain = []
        pairedchain = []

        heavy_path = os.path.join(file_path, "heavy_chains.txt")
        if os.path.isfile(heavy_path):
            with open(heavy_path, encoding="utf-8") as f:
                heavychain = [
                    f"{line}|"
                    for line in f.read().splitlines()
                    if (len(line) > 0 and not line.isspace())
                ]

        light_path = os.path.join(file_path, "light_chains.txt")
        if os.path.isfile(light_path):
            with open(light_path, encoding="utf-8") as f:
                lightchain = [
                    f"|{line}"
                    for line in f.read().splitlines()
                    if (len(line) > 0 and not line.isspace())
                ]

        paired_path = os.path.join(file_path, "paired_chains.txt")
        if os.path.isfile(paired_path):
            with open(paired_path, encoding="utf-8") as f:
                pairedchain = [
                    line
                    for line in f.read().splitlines()
                    if (len(line) > 0 and not line.isspace())
                ]

        if is_train_data and (self.over_sample_data == 1):
            all_data_lists_with_content = []
            if heavychain:
                all_data_lists_with_content.append(heavychain)
            if lightchain:
                all_data_lists_with_content.append(lightchain)
            if pairedchain:
                all_data_lists_with_content.append(pairedchain)

            if not all_data_lists_with_content:
                return []

            max_len = 0
            for lst in all_data_lists_with_content:
                if len(lst) > max_len:
                    max_len = len(lst)

            oversampled_data = []
            if heavychain:
                scale_h = max_len // len(heavychain) if len(heavychain) > 0 else 0
                oversampled_data.extend(heavychain * scale_h)
            if lightchain:
                scale_l = max_len // len(lightchain) if len(lightchain) > 0 else 0
                oversampled_data.extend(lightchain * scale_l)
            if pairedchain:
                scale_p = max_len // len(pairedchain) if len(pairedchain) > 0 else 0
                oversampled_data.extend(pairedchain * scale_p)
            return oversampled_data
        else:
            return heavychain + lightchain + pairedchain

    def __len__(self):
        return len(self.data)

    def _get_mask_arguments(self, tkn_sequences):
        mask_num = int(tkn_sequences.shape[1] * self.mask_percent)
        if self.mask_variable:
            mask_num = np.random.randint(10, mask_num + 10, size=None)

        change_percent = self.change_percent
        if self.change_percent == -1:
            change_percent = np.random.choice([0.1, 0.2, 0.4, 0.6, 0.8], size=None)

        if self.mask_technique == "mix":
            mask_technique = np.random.choice(
                ["shotgun", "span_long", "span_short"],
                p=(1 / 3, 1 / 3, 1 / 3),
                size=None,
            )
            return mask_num, mask_technique, change_percent
        else:
            return mask_num, self.mask_technique, change_percent

    def _get_unmasks(self, tkn_sequences):
        """
        base_unmask, tokens which are always unmasked
        selection_mask, tokens which can be selected to be masked
        """

        base_unmask = (
            (tkn_sequences == self.start_tkn)
            | (tkn_sequences == self.sep_tkn)
            | (tkn_sequences == self.end_tkn)
        )
        attention_mask = tkn_sequences.eq(self.pad_tkn)

        return base_unmask, (~(attention_mask + base_unmask)).float()

    def __getitem__(self, idx):
        sequence = self.data[idx]
        # Tokenize the single sequence.
        # The tokenizer is expected to return a 2D tensor [1, sequence_length].
        # w_extra_tkns=False means tokenizer should not add start/end tokens itself.
        # We set pad=False because we will handle padding/truncation to self.block_size manually.
        tokenized_output = self.tokenizer([sequence], w_extra_tkns=False, pad=False)
        tkn_sequences = tokenized_output[0]  # tokenizer returns a list of tensors

        # Manually truncate or pad to self.block_size
        if self.block_size is not None:
            if tkn_sequences.size(0) > self.block_size:
                tkn_sequences = tkn_sequences[: self.block_size]
            elif tkn_sequences.size(0) < self.block_size:
                padding_size = self.block_size - tkn_sequences.size(0)
                # Ensure pad_tkn is a tensor for concatenation if tkn_sequences is a tensor
                pad_tensor = torch.full(
                    (padding_size,),
                    self.pad_tkn,
                    dtype=tkn_sequences.dtype,
                    device=tkn_sequences.device,
                )
                tkn_sequences = torch.cat([tkn_sequences, pad_tensor], dim=0)

        # Add batch dimension back for subsequent processing if needed by _get_mask_arguments and _get_unmasks
        tkn_sequences = tkn_sequences.unsqueeze(0)

        mask_num, mask_technique, change_percent = self._get_mask_arguments(
            tkn_sequences
        )

        base_unmask, selection_mask = self._get_unmasks(tkn_sequences)

        idx_corrupt, idx_leave, idx_mask = get_indices(
            selection_mask,
            mask_num=mask_num,
            p_corrupt=change_percent,
            p_leave=self.leave_percent,
            cdr3_focus=self.cdr3_focus,
            mask_technique=mask_technique,
        )

        if mask_num == 0:  # For edge cases, where mask_num is 0
            masked_sequences = tkn_sequences.clone()
            # Create labels: -100 for non-masked tokens
            labels = tkn_sequences.clone()
            labels[~base_unmask.long().bool()] = -100
            return {
                "input": masked_sequences.squeeze(0),
                "labels": labels.squeeze(0),
            }

        masked_sequences, labels = mask_sequences(
            tkn_sequences,
            base_unmask,
            idx_corrupt,
            idx_leave,
            idx_mask,
            mask_tkn=self.mask_tkn,
            pad_tkn=self.pad_tkn,
        )

        return masked_sequences.squeeze(0), labels.squeeze(0)
