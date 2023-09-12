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
