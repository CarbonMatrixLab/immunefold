import os
import numpy as np

from carbonmatrix.data.antibody.seq import extract_framework_struc

def get_reference_framework_struc():
    struc_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '1hzh_H_L.npy')
    x = np.load(struc_file, allow_pickle=True).item()

    return extract_framework_struc(x['heavy_chain'], x['light_chain'])