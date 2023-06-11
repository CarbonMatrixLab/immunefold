import os
import argparse
import pandas as pd
from pathlib import Path

def read_fasta(fasta_file):
    with open(fasta_file) as f:
        lines = f.readlines()
    return lines[1].strip(), lines[3].strip()

def main(args):
    names = list(pd.read_csv(args.name_idx, names=['name'], header=None)['name'])

    with open(args.output_file, 'w') as fw:
        for i, n in enumerate(names):
            fasta_file = os.path.join(args.fasta_dir, n + '.fasta')
            if not os.path.exists(fasta_file):
                continue
            
            H_seq, L_seq = read_fasta(fasta_file)
            fw.write(f'>{n}\n{H_seq}:{L_seq}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--name_idx', type=str, required=True)
    parser.add_argument('-p', '--fasta_dir', type=str, required=True)
    parser.add_argument('-o', '--output_file', type=str, required=True)

    args = parser.parse_args()

    main(args)
