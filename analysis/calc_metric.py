import sys
import argparse

import numpy as np
import pandas as pd

def read_metric(in_file, max_hcdr3_len=50, filter_list=None):
    df = pd.read_csv(in_file, sep='\t')

    df = df[df['heavy_cdr3_len'] <= max_hcdr3_len]
    df = df[df['heavy_cdr3_len'] > 1]

    if filter_list:
        df = df[df['name'].isin(filter_list)]
    
    print(df['heavy_cdr3_len'].describe())
    
    print('size', df.shape[0])

    #head = [n for n in df.columns if n.endswith('_rmsd') or n.endswith('_len')]
    head = [n for n in df.columns if n.endswith('_rmsd')]
    
    metric = [df[n].mean() for n in head]

    print('\t'.join(head))
    print('\t'.join([f'{m:.2f}' for m in metric]))

def main(args):
    df = pd.read_csv(args.summary_file, sep='\t')
    df = df[df['resolution'] <= 2.5]
    names = list(df['name'])
    
    if args.black_list:
        with open(args.black_list) as f:
            black_list = [x.strip() for x in f]
        names = [n for n in names if n not in black_list]

    read_metric(args.metric_file, filter_list=names)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--metric_file', type=str, required=True)
  parser.add_argument('--summary_file', type=str, required=True)
  parser.add_argument('--black_list', type=str)

  args = parser.parse_args()


  main(args)
