import json
import torch

from .vocab import ablang_vocab


class ABtokenizer:
    """
    Tokenizer for proteins (focus on handling antibodies). Both aa to token and token to aa.
    """

    def __init__(self, vocab_dir=None, max_sequence_length=None):
        self.set_vocab(vocab_dir)
        self.max_sequence_length = max_sequence_length

    def __call__(
        self, sequence_list, mode="encode", pad=True, w_extra_tkns=True, device="cpu"
    ):

        sequence_list = (
            [sequence_list] if isinstance(sequence_list, str) else sequence_list
        )

        if mode == "encode":
            data = [
                self.encode(seq, w_extra_tkns=w_extra_tkns, device=device)
                for seq in sequence_list
            ]
            if pad:
                return torch.nn.utils.rnn.pad_sequence(
                    data, batch_first=True, padding_value=self.pad_token
                )
            else:
                return data
        elif mode == "decode":
            return [
                self.decode(tokenized_seq, w_extra_tkns=w_extra_tkns)
                for tokenized_seq in sequence_list
            ]
        else:
            raise SyntaxError("Given mode doesn't exist. Use either encode or decode.")

    def set_vocab(self, vocab_dir):

        if vocab_dir:
            with open(vocab_dir, encoding="utf-8") as vocab_handle:
                self.vocab_to_token = json.load(vocab_handle)
        else:
            self.aa_to_token = ablang_vocab

        self.token_to_aa = {v: k for k, v in self.aa_to_token.items()}
        self.pad_token = self.aa_to_token["-"]
        self.start_token = self.aa_to_token["<"]
        self.end_token = self.aa_to_token[">"]
        self.sep_token = self.aa_to_token["|"]

    def encode(self, sequence, w_extra_tkns=True, device="cpu"):

        if w_extra_tkns:
            tokenized_seq_list = (
                [self.start_token]
                + [self.aa_to_token[resn] for resn in sequence]
                + [self.end_token]
            )
        else:
            tokenized_seq_list = [self.aa_to_token[resn] for resn in sequence]

        if self.max_sequence_length is not None:
            current_len = len(tokenized_seq_list)
            target_len = self.max_sequence_length

            if current_len > target_len:
                if target_len == 0:
                    tokenized_seq_list = []
                elif w_extra_tkns and target_len > 0:
                    if target_len == 1:
                        tokenized_seq_list = [tokenized_seq_list[0]]
                    else:
                        tokenized_seq_list = tokenized_seq_list[: target_len - 1] + [
                            self.end_token
                        ]
                else:
                    tokenized_seq_list = tokenized_seq_list[:target_len]

            elif current_len < target_len:
                padding_needed = target_len - current_len
                tokenized_seq_list.extend([self.pad_token] * padding_needed)

        return torch.tensor(tokenized_seq_list, dtype=torch.long, device=device)

    def decode(self, tokenized_seq, w_extra_tkns=True):

        if torch.is_tensor(tokenized_seq):
            tokenized_seq = tokenized_seq.cpu().numpy()

        if w_extra_tkns:
            return "".join([self.token_to_aa[token] for token in tokenized_seq[1:-1]])
        else:
            return "".join([self.token_to_aa[token] for token in tokenized_seq])
