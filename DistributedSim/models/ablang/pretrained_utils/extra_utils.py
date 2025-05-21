import os, string, re
from dataclasses import dataclass

from numba import jit
from numba.typed import Dict, List
from numba.types import unicode_type

import numpy as np


def res_to_list(state, seq):
    return state[1:1+len(seq)]

def res_to_seq(a, mode='mean'):
    """
    Function for how we go from n_values for each amino acid to n_values for each sequence.
    
    We leave out the start, end and padding tokens.
    """
    if mode=='sum':
        return a[1:(1+int(a[-1]))].sum()
    
    elif mode=='mean':
        return a[1:(1+int(a[-1]))].mean()
    
    elif mode=='restore':
        
        return a[0][0:(int(a[-1]))]

def get_number_alignment(oanarci):
    """
    Creates a number alignment from the anarci results.
    """
    
    import pandas as pd
    
    alist = []
    
    for aligned_seq in oanarci[1]:
        alist.append(pd.DataFrame(aligned_seq[0][0])[0])

    unsorted_alignment = pd.concat(alist).drop_duplicates()
    max_alignment = get_max_alignment()
    
    return max_alignment.merge(unsorted_alignment.to_frame(), left_on=0, right_on=0)

def get_max_alignment():
    """
    Create maximum possible alignment for sorting
    """
    
    import pandas as pd

    sortlist = []

    for num in range(1, 128+1):

        if num==112:
            for char in string.ascii_uppercase[::-1]:
                sortlist.append([(num, char)])

            sortlist.append([(num,' ')])

        else:
            sortlist.append([(num,' ')])
            for char in string.ascii_uppercase:
                sortlist.append([(num, char)])
                
    return pd.DataFrame(sortlist)


def paired_msa_numbering(ab_seqs, ncpus = 10):
    
    import pandas as pd
    
    tmp_seqs = [pairs.split("|") for pairs in ab_seqs]
    
    anarci_out_heavy, seqs_heavy, number_alignment_heavy = ab_msa_numbering([i[0] for i in tmp_seqs], 'H', ncpus)
    anarci_out_light, seqs_light, number_alignment_light = ab_msa_numbering([i[1] for i in tmp_seqs], 'L', ncpus)
    
    number_alignment = pd.concat([number_alignment_heavy, pd.DataFrame([[("|",""), "|"]]), number_alignment_light]).reset_index(drop=True)
    seqs = [f"{heavy}|{light}" for heavy, light in zip(seqs_heavy, seqs_light)]
    anarci_out = [
        heavy + [(("|",""), "|", "|")] + light for heavy, light in zip(anarci_out_heavy, anarci_out_light)
    ]
    
    return anarci_out, seqs, number_alignment


def ab_msa_numbering(seqs, chain = 'H', ncpus = 10):
    
    import pandas as pd
    import anarci
    
    anarci_out = anarci.run_anarci(
        pd.DataFrame(seqs).reset_index().values.tolist(), 
        ncpu=ncpus, 
        scheme='imgt',
        allowed_species=['human', 'mouse'],
    )
    number_alignment = get_number_alignment(anarci_out)
    number_alignment[1] = chain
    
    anarci_out = [[(i[0], chain, i[1]) for i in onarci[0][0]] for onarci in anarci_out[1]]
    seqs = [''.join([i[2] for i in onarci]).replace('-','') for onarci in anarci_out]
    
    return anarci_out, seqs, number_alignment


def create_alignment(res_embeds, oanarci, seq, number_alignment):
    
    import pandas as pd

    datadf = pd.DataFrame(oanarci)
    sequence_alignment = number_alignment.merge(datadf, how='left', on=[0, 1]).fillna('-')[2]

    idxs = np.where(sequence_alignment.values == '-')[0]
    
    idxs = [idx-num for num, idx in enumerate(idxs)]
    
    aligned_embeds = pd.DataFrame(np.insert(res_embeds[:1+len(seq)], idxs , 0, axis=0))

    return pd.concat([aligned_embeds, sequence_alignment], axis=1).values

def turn_into_numba(anarcis):
    """
    Turns the nested anarci dictionary into a numba item, allowing us to use numba on it.
    """
    
    anarci_list = List.empty_list(unicode_type)
    [anarci_list.append(str(anarci)) for anarci in anarcis]

    return anarci_list

@jit(nopython=True)
def get_spread_sequences(seq, spread, start_position, numbaList):
    """
    Test sequences which are 8 positions shorter (position 10 + max CDR1 gap of 7) up to 2 positions longer (possible insertions).
    """

    for diff in range(start_position-8, start_position+2+1):
        numbaList.append('*'*diff+seq)
    
    return numbaList

def get_sequences_from_anarci(out_anarci, max_position, spread):
    """
    Ensures correct masking on each side of sequence
    """
    
    if out_anarci == 'ANARCI_error':
        return np.array(['ANARCI-ERR']*spread)
    
    end_position = int(re.search(r'\d+', out_anarci[::-1]).group()[::-1])
    # Fixes ANARCI error of poor numbering of the CDR1 region
    start_position = int(re.search(r'\d+,\s\'.\'\),\s\'[^-]+\'\),\s\(\(\d+,\s\'.\'\),\s\'[^-]+\'\),\s\(\(\d+,\s\'.\'\),\s\'[^-]+\'\),\s\(\(\d+,\s\'.\'\),\s\'[^-]+',
                                   out_anarci).group().split(',')[0]) - 1
    
    sequence = "".join(re.findall(r"(?i)[A-Z*]", "".join(re.findall(r'\),\s\'[A-Z*]', out_anarci))))

    sequence_j = ''.join(sequence).replace('-','').replace('X','*') + '*'*(max_position-int(end_position))
    
    numba_list = List.empty_list(unicode_type)

    spread_seqs = np.array(get_spread_sequences(sequence_j, spread, start_position, numba_list))

    return spread_seqs
