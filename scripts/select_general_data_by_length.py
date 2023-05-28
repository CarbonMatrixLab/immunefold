import numpy as np
import argparse
import pandas as pd

import sys
import os

def main(args):
    name_file = args.input_name_idx
    data_dir = args.data_dir
    
    df = pd.read_csv(name_file, header=None, names=['name'])
    names = list(df['name'])
    
    def process_one(name, data):
        seq = str(data['seq'])
        struc_len  = np.sum(data['coord_mask'][:,1])
        return name, len(seq), struc_len
    
    data = []
    for n in names:
        d = np.load(f'{data_dir}/{n}.npz')
        ret = process_one(n, d)
        data.append(ret)
    
    columns = ['name', 'len', 'struc_len']
    df = pd.DataFrame(dict(zip(columns, zip(*data))))
    print(df.head())
    df = df.sort_values(by='len')
    df.to_csv(f'{args.output}_all.csv', sep='\t', index=False)
    
    df = df[df['len'] >= 80]
    df = df[df['len'] <= args.max_len]
    df.to_csv(f'{args.output}_len_{args.max_len}.csv', sep='\t', index=False)
    
    df = df.sample(frac=1, random_state=2022).reset_index(drop=True)
    
    validate = df.iloc[:args.validate_size]
    train = df.iloc[args.validate_size:]
    train.to_csv(f'{args.output}_train_len_{args.max_len}.csv', sep='\t', index=False)
    validate.to_csv(f'{args.output}_validate_len_{args.max_len}.csv', sep='\t', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name_idx', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--validate_size', type=int, default=200)

    args = parser.parse_args()
    
    main(args)
