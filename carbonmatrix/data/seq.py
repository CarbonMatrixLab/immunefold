import numpy as np

from esm.data import Alphabet 
from carbonmatrix.common import residue_constants

esm_alphabet = Alphabet.from_architecture(name='ESM-1b')

def str_seq_to_index(str_seq, mapping=residue_constants.restype_order_with_x, map_unknown_to_x=True):
    seq = []
    for aa in str_seq:
      if aa not in mapping and not map_unknown_to_x:
          raise ValueError(f'Invalid character in the sequence: {aa}')
      seq.append(mapping.get(aa, mapping['X']))

    return np.array(seq)
    
def create_esm_seq(str_seq):
    L = len(str_seq)
    seq = np.zeros((L + 2,), dtype=np.int64)
    seq[0] = esm_alphabet.cls_idx
    seq[-1] = esm_alphabet.eos_idx

    for i, a in enumerate(str_seq):
        seq[i+1] = esm_alphabet.get_idx(a)

    return seq
