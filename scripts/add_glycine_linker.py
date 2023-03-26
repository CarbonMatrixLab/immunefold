import os
import argparse
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd

def add_linker(names, args):
    for i, n in enumerate(names):
        npz_file = os.path.join(args.npz_dir, n + '.npz')
        out_file = os.path.join(args.out_dir, n + '.fasta')
        data = np.load(npz_file)
        if 'heavy_str_seq' in data and 'light_str_seq' in data:
            heavy_str_seq = str(data['heavy_str_seq'])
            light_str_seq = str(data['light_str_seq'])
            seq = light_str_seq + 'G' * 50 + heavy_str_seq
            with open(out_file, 'w') as fw:
                fw.write(f'>{n}\n{seq}\n')

def main(args):
    names = list(pd.read_csv(args.name_idx, names=['name'], header=None)['name'])

    add_linker(names, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_idx', type=str, required=True)
    parser.add_argument('--npz_dir', type=str, required=True, help='npz dir')
    parser.add_argument('--out_dir', type=str, required=True, help='output dir, default=\'.\'')
    args = parser.parse_args()
    
    main(args)
