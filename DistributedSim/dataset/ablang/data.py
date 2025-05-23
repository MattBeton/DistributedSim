import os

import numpy as np

from torch.utils.data import Dataset
from DistributedSim.models.ablang.ablang_2.tokenizers import ABtokenizer
from DistributedSim.dataset.ablang.datacollators import ABcollator


class AbDataset(Dataset):
    def __init__(self, file_path, tokenizer, over_sample_data=0):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.collator = ABcollator(tokenizer)
        self.over_sample_data = over_sample_data
        self.data = self._get_data(self.file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.collator(self.data[idx])
        squeezed_input = item["input"].squeeze(0)
        return squeezed_input, item["labels"]

    def _get_data(self, file_path, is_train_data=True):
        "Reads txt file of sequences."
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")

        if os.path.isfile(os.path.join(file_path, "heavy_chains.txt")):
            with open(
                os.path.join(file_path, "heavy_chains.txt"), encoding="utf-8"
            ) as f:
                heavychain = [
                    f"{line}|"
                    for line in f.read().splitlines()
                    if (len(line) > 0 and not line.isspace())
                ]
        else:
            heavychain = []

        if os.path.isfile(os.path.join(file_path, "light_chains.txt")):
            with open(
                os.path.join(file_path, "light_chains.txt"), encoding="utf-8"
            ) as f:
                lightchain = [
                    f"|{line}"
                    for line in f.read().splitlines()
                    if (len(line) > 0 and not line.isspace())
                ]
        else:
            lightchain = []

        if os.path.isfile(os.path.join(file_path, "paired_chains.txt")):
            with open(
                os.path.join(file_path, "paired_chains.txt"), encoding="utf-8"
            ) as f:
                pairedchain = [
                    line
                    for line in f.read().splitlines()
                    if (len(line) > 0 and not line.isspace())
                ]
        else:
            pairedchain = []

        if is_train_data and (self.over_sample_data == 1):
            sizes = [len(heavychain), len(lightchain), len(pairedchain)]
            scale = (np.max(sizes) / sizes).astype(np.int16)
            return (
                heavychain * scale[0] + lightchain * scale[1] + pairedchain * scale[2]
            )

        else:
            return heavychain + lightchain + pairedchain


if __name__ == "__main__":
    tokenizer = ABtokenizer(max_sequence_length=256)
    dataset = AbDataset(
        file_path="/Users/sethhowes/Desktop/exo/DistributedSim/data/train_data",
        tokenizer=tokenizer,
    )
    print(len(dataset))
    dataset[0]
