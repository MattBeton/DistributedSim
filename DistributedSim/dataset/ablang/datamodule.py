import os
# import numpy as np # No longer needed in the rewritten selection
import torch # For Subset
from torch.utils.data import DataLoader, Dataset, Subset # Added Dataset, Subset

from .datacollators import ABcollator
# Assuming pytorch_lightning as pl is imported elsewhere and available


class AbDataset(Dataset):
  def __init__(self, file_path, tokenizer_instance, over_sample_data=0, is_train_data=False):
    super().__init__()
    self.tokenizer = tokenizer_instance # Expecting an already instantiated tokenizer
    self.over_sample_data = over_sample_data
    self.data = self._load_data(file_path, is_train_data)

  def _load_data(self, file_path, is_train_data):
    heavychain = []
    lightchain = []
    pairedchain = []

    heavy_path = os.path.join(file_path, 'heavy_chains.txt')
    if os.path.isfile(heavy_path):
      with open(heavy_path, encoding="utf-8") as f:
        heavychain = [f"{line}|" for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    light_path = os.path.join(file_path, 'light_chains.txt')
    if os.path.isfile(light_path):
      with open(light_path, encoding="utf-8") as f:
        lightchain = [f"|{line}" for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    
    paired_path = os.path.join(file_path, 'paired_chains.txt')
    if os.path.isfile(paired_path):
      with open(paired_path, encoding="utf-8") as f:
        pairedchain = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    
    if is_train_data and (self.over_sample_data == 1):
      all_data_lists_with_content = []
      if heavychain: all_data_lists_with_content.append(heavychain)
      if lightchain: all_data_lists_with_content.append(lightchain)
      if pairedchain: all_data_lists_with_content.append(pairedchain)

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

  def __getitem__(self, idx):
    return self.data[idx]


class AbDataModule(pl.LightningDataModule):

  def __init__(self, data_hparams, tokenizer_class): # tokenizer_class is expected
    super().__init__()
    self.data_hparams = data_hparams
    self.tokenizer = tokenizer_class() # Instantiate tokenizer
    self.over_sample_data = data_hparams.over_sample_data
      
  def setup(self, stage=None): # called on every GPU
    
    self.traincollater = ABcollator(
      self.tokenizer, 
      pad_tkn = self.data_hparams.pad_tkn,
      start_tkn = self.data_hparams.start_tkn,
      end_tkn = self.data_hparams.end_tkn,
      sep_tkn = self.data_hparams.sep_tkn,
      mask_tkn = self.data_hparams.mask_tkn,
      mask_percent=self.data_hparams.mask_percent,
      mask_variable=self.data_hparams.variable_masking,
      cdr3_focus=self.data_hparams.cdr3_focus,
      mask_technique=self.data_hparams.mask_technique,
      change_percent = self.data_hparams.change_percent,
      leave_percent = self.data_hparams.leave_percent,
    )
    
    self.evalcollater = ABcollator(
      self.tokenizer, 
      pad_tkn = self.data_hparams.pad_tkn,
      start_tkn = self.data_hparams.start_tkn,
      end_tkn = self.data_hparams.end_tkn,
      sep_tkn = self.data_hparams.sep_tkn,
      mask_tkn = self.data_hparams.mask_tkn,
      mask_percent = .15,
      mask_variable = self.data_hparams.variable_masking,
      cdr3_focus = 1.,
      mask_technique = "shotgun",
      change_percent = .1,
      leave_percent = .1,
    )
    
    train_file_path = os.path.join(self.data_hparams.data_path, 'train_data')
    self.train_dataset = AbDataset(
      file_path=train_file_path, 
      tokenizer_instance=self.tokenizer, 
      over_sample_data=self.over_sample_data, 
      is_train_data=True
    )
    
    eval_file_path = os.path.join(self.data_hparams.eval_path, 'eval_data')
    full_val_dataset = AbDataset(
      file_path=eval_file_path,
      tokenizer_instance=self.tokenizer,
      over_sample_data=0, # No oversampling for validation
      is_train_data=False 
    )
    
    # Original code sliced validation data: self.val = ...[:1000]
    num_val_samples_to_take = 1000
    if len(full_val_dataset) > num_val_samples_to_take:
      self.val_dataset = Subset(full_val_dataset, range(num_val_samples_to_take))
    else:
      self.val_dataset = full_val_dataset # Take all if less than or equal to 1000
      
  def train_dataloader(self):
    return DataLoader(self.train_dataset,
                      batch_size=self.data_hparams.train_batch_size,
                      collate_fn=self.traincollater,
                      num_workers=self.data_hparams.cpus,
                      shuffle=True,
                      pin_memory=True,
                      )

  def val_dataloader(self): # rule of thumb is: num_worker = 4 * num_GPU
    return DataLoader(self.val_dataset, 
                      batch_size=self.data_hparams.eval_batch_size, 
                      collate_fn=self.evalcollater, 
                      num_workers=self.data_hparams.cpus,
                      pin_memory=True,
                      # shuffle=False is default and usually desired for validation
                      )

  # The get_data method has been moved into AbDataset._load_data