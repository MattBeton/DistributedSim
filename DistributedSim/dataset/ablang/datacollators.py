import torch
import math

import numpy as np


class ABcollator():
    """
    This collator creates; 
    1. masked input data
    2. labels with only masks visible
    
    Padded tokens are also masked.
    """
    def __init__(
        self, 
        tokenizer, 
        pad_tkn=21, 
        start_tkn=0, 
        end_tkn=22,
        sep_tkn=25, 
        mask_tkn=23,
        mask_percent=.15, 
        mask_variable=False, 
        cdr3_focus=1, 
        mask_technique='shotgun',
        change_percent=.1,
        leave_percent=.1,
    ):
        
        self.tokenizer = tokenizer
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
        
    def get_mask_arguments(self, tkn_sequences):
        
        mask_num = int(tkn_sequences.shape[1] * self.mask_percent)
        if self.mask_variable: mask_num = np.random.randint(10, mask_num + 10, size=None)
        
        change_percent = self.change_percent
        if self.change_percent == -1: change_percent = np.random.choice([.1, .2, .4, .6, .8], size=None)
        
        if self.mask_technique == 'mix':
            mask_technique = np.random.choice(['shotgun', 'span_long', 'span_short'], p = (1/3, 1/3,1/3), size=None) 
            return mask_num, mask_technique, change_percent
        else:
            return mask_num, self.mask_technique, change_percent
        
    def get_unmasks(self, tkn_sequences):
        """
        base_unmask, tokens which are always unmasked
        selection_mask, tokens which can be selected to be masked
        """
    
        base_unmask = ((tkn_sequences == self.start_tkn) | (tkn_sequences == self.sep_tkn) | (tkn_sequences == self.end_tkn))
        attention_mask = tkn_sequences.eq(self.pad_tkn)

        return base_unmask, (~(attention_mask + base_unmask)).float()
        
        
    def __call__(self, batch):
        
        tkn_sequences = self.tokenizer(batch, w_extra_tkns=False, pad=True)
        
        mask_num, mask_technique, change_percent = self.get_mask_arguments(tkn_sequences)
        
        base_unmask, selection_mask = self.get_unmasks(tkn_sequences)      
        
        idx_corrupt, idx_leave, idx_mask = get_indices(
            selection_mask, 
            mask_num=mask_num, 
            p_corrupt=change_percent, 
            p_leave=self.leave_percent, 
            cdr3_focus=self.cdr3_focus, 
            mask_technique=mask_technique
        )
        
        if mask_num == 0: # For edge cases, where mask_num is 0
            masked_sequences = tkn_sequences.clone()
            tkn_sequences[~base_unmask.long().bool()] = -100
            return {'input':masked_sequences, 'labels':tkn_sequences.view(-1), 'sequences':batch}
        
        masked_sequences, tkn_sequences = mask_sequences(
            tkn_sequences, 
            base_unmask,
            idx_corrupt, idx_leave, idx_mask,
            mask_tkn=self.mask_tkn, 
        )
        
        return {'input':masked_sequences, 'labels':tkn_sequences.view(-1), 'sequences':batch}

    
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
    masked_sequences.scatter_(1, idx_corrupt, torch.randint(1, 21, masked_sequences.shape, device=masked_sequences.device)) # randomly changes idx_change in the data 
    masked_sequences.scatter_(1, idx_mask, mask_tkn) # change idx_mask inputs to <mask>
    
    base_unmask.scatter_(1, idx_mask, 1)
    base_unmask.scatter_(1, idx_corrupt, 1)
    base_unmask.scatter_(1, idx_leave, 1)
    tkn_sequences[~base_unmask.long().bool()] = -100
    tkn_sequences[(tkn_sequences == pad_tkn)] = -100 # This fixes potential errors in short/long masking
    
    return masked_sequences, tkn_sequences
        
        
def adjust_selection_mask(selection_mask, mask_num):

    idx = torch.arange(selection_mask.shape[1], 0, -1)
    indices = torch.argmax(selection_mask * idx, 1, keepdim=True)
    
    for test_idx in indices.reshape(-1):
        selection_mask[:,test_idx - mask_num - 1:selection_mask.shape[1]] = 0
    
    return selection_mask

    
def get_indices(
    selection_mask, 
    mask_num, 
    p_corrupt=.1, 
    p_leave=.1, 
    cdr3_focus = 1,
    mask_technique = 'shotgun'
):
    #allowed_mask[:, 106:118] *= cdr3_focus # Changes the chance of residues in the CDR3 getting masked. It's 106 and 118 because the start token is present.
    
    if (mask_technique == 'shotgun') or (mask_technique == 'random'):
        indices = torch.multinomial(selection_mask, num_samples=mask_num, replacement=False)
    
    elif mask_technique == 'span_long':
        selection_mask = adjust_selection_mask(selection_mask, mask_num)
        
        start_idx = torch.multinomial(selection_mask, num_samples = 1, replacement = False).repeat(1, mask_num)
        step_idx = torch.linspace(0, mask_num-1, steps = mask_num, dtype = int).repeat(selection_mask.shape[0], 1)
        indices = start_idx + step_idx
        indices = indices.clamp(max = selection_mask.shape[1] - 1)
        
    elif mask_technique == 'span_short':        
        span_lens = np.random.choice([3, 4, 5], size=(5))
        span_sep_lens = torch.normal(mean=15, std=6, size=(5,)).int().clamp(min = 1, max = 15)

        start_idx = 0
        many_span_idx = []
        for span_len, span_sep_len in zip(span_lens, span_sep_lens):
            
            many_span_idx.append(
                torch.linspace(start_idx, start_idx+span_len-1, steps = span_len, dtype = int).repeat(selection_mask.shape[0], 1)
            )
            start_idx += (span_len + span_sep_len)

        indices = torch.concatenate(many_span_idx, axis=1)
        indices = indices.clamp(max = selection_mask.shape[1] - 1)   
        
    n_corrupt = max(int(indices.shape[1]*p_corrupt), 1)
    n_leave  = max(int(indices.shape[1]*p_leave ), 1)

    return torch.split(indices, split_size_or_sections = [n_corrupt, n_leave, max(indices.shape[-1] - (n_corrupt + n_leave), 0)], dim = 1)
