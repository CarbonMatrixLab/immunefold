import os
import argparse
import pandas as pd
import numpy as np

from igai.preprocess.numbering import renumber_pdb_by_remote

def parse_sabdab_summary(path):

    df = pd.read_csv(path, sep='\t') #dtype={'Hchain':str,'Lchain':str})
    df = df.dropna(subset=['Hchain', 'Lchain'])

    def _resolution(x):
        if x == 'NOT':
            return np.inf
        return float(x.split(',')[0].strip())

    df['resolution'] = df['resolution'].apply(_resolution)
    
    def _filter(x):
        return x['resolution'] < 4.0 \
                and x['Hchain'].isupper() \
                and x['Lchain'].isupper()

    df = df[df.apply(_filter, axis=1)]
    df = df.drop_duplicates(['pdb'])

    return list(df['pdb'])

def main(args):
    os.makedirs(args.output, exist_ok=True) 
    
    proteins = parse_sabdab_summary(args.name_idx)[:1]
    print(f'{len(proteins)} paired strcutures')
    
    for name in proteins[:1]:
        print(name)
        old_pdb = os.path.join(args.pdb_dir, name + '.pdb')
        new_pdb = os.path.join(args.output, name + '.pdb')
        renumber_pdb_by_remote(old_pdb, new_pdb)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('-c', '--name_idx', type=str, required=True)
  parser.add_argument('-m', '--pdb_dir', type=str, default='pdb',
      help='pdb dir, default=\'pdb\'')
  parser.add_argument('-o', '--output', type=str, default='.',
      help='output dir, default=\'.\'')
  args = parser.parse_args()


  main(args)

