from pathlib import Path
import os
import sys
import argparse

def main(args):
    with open(args.clu_file) as f, open(args.output, 'w') as fw:
        prev_id = None
        
        for line in f:
            clu_id, seq = line.strip().split()
            seq = seq.strip('\"')

            if clu_id != prev_id:
                if prev_id is not None:
                    fw.write('\t'.join(clu) + '\n')
                clu = []
                prev_id = clu_id
            clu.append(seq)
        
        if prev_id is not None:
            fw.write('\t'.join(clu) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--clu_file', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    main(args)
