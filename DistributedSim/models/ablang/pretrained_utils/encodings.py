from dataclasses import dataclass
import numpy as np
import torch

from .extra_utils import paired_msa_numbering, ab_msa_numbering, create_alignment, res_to_list, res_to_seq


class AbEncoding:
    
    
    def __init__(self, model, tokenizer, spread, max_position, device = 'cpu', ncpu = 1):
        
        self.AbRep = model.AbRep
        self.tokenizer = tokenizer
        self.spread = spread
        self.max_position = max_position
        self.device = device
        self.ncpu = ncpu
        
    def get_seq_coding(self, seqs, align=False, chain = 'H'):
        """
        Sequence specific representations
        """
        
        tokens = self.tokenizer(seqs, pad=True, device=self.device)

        residue_states = self.AbRep(tokens).last_hidden_states
        
        if torch.is_tensor(residue_states): residue_states = residue_states.cpu().detach().numpy()
        
        lens = np.vectorize(len)(seqs)
        
        lens = np.tile(lens.reshape(-1,1,1), (residue_states.shape[2], 1))
        
        seq_codings = np.apply_along_axis(res_to_seq, 2, np.c_[np.swapaxes(residue_states,1,2), lens])
        
        del lens
        del residue_states
        
        return seq_codings
        
    def get_res_coding(self, seqs, align=False, chain = 'H'):
        """
        Residue specific representations.
        """
           
        if align:
            
            if chain == 'P':
                anarci_out, seqs, number_alignment = paired_msa_numbering(seqs, self.ncpu)
                
            else:
                anarci_out, seqs, number_alignment = ab_msa_numbering(seqs, chain = chain, ncpus = self.ncpu)
                
            
            tokens = self.tokenizer(seqs, pad=True, w_extra_tkns=False, device=self.device)
            with torch.no_grad():
                residue_states = self.AbRep(tokens).last_hidden_states
            
            if torch.is_tensor(residue_states): residue_states = residue_states.cpu().detach().numpy()
            
            residue_output = np.array([create_alignment(
                res_embed, oanarci, seq, number_alignment
            ) for res_embed, oanarci, seq in zip(residue_states, anarci_out, seqs)])
            
            del residue_states
            del tokens
            
            return output(aligned_embeds=residue_output, number_alignment=number_alignment.apply(lambda x: '{}{}'.format(*x[0]), axis=1).values)
            
        else:
            
            tokens = self.tokenizer(seqs, pad=True, device=self.device)
            residue_states = self.AbRep(tokens).last_hidden_states
        
            if torch.is_tensor(residue_states): residue_states = residue_states.cpu().detach().numpy()
            
            residue_output = [res_to_list(state, seq) for state, seq in zip(residue_states, seqs)]

            return residue_output
        
        

@dataclass
class output():
    """
    Dataclass used to store output.
    """

    aligned_embeds: None
    number_alignment: None