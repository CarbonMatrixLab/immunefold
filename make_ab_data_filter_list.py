import os
import pandas as pd
import numpy as np
import argparse
import logging
import json
import subprocess
import random

random.seed(2023)

def _save_heavy_fasta(save_path, df, concat=False):
    with open(save_path, 'w') as fw:
        for idx, r in df.iterrows():
            name = r['name']
            heavy_str_seq = np.load(os.path.join(args.data_dir, name + '.npz'))['heavy_str_seq']
            fw.write(f'>{name}\n{heavy_str_seq}\n')


def _save_glycine_fasta(save_path, df, concat=True):
    with open(save_path, 'w') as fw:
        for idx, r in df.iterrows():
            name = r['name']
            x = np.load(os.path.join(args.data_dir, name + '.npz'))
            heavy_str_seq, light_str_seq = x.get('heavy_str_seq',''), x.get('light_str_seq', '')
            if heavy_str_seq and light_str_seq:
                if concat:
                    fw.write(f'>{name}\n{heavy_str_seq}{"G" * 50}{light_str_seq}\n')
                else:
                    fw.write(f'>{name} heavy \n{heavy_str_seq}\n>{name} light\n{light_str_seq}\n')
            else:
                fw.write(f'>{name}\n{heavy_str_seq}\n')

def _save_seperate_fasta(save_dir, df, concat=False):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for idx, r in df.iterrows():
        name = r['name']
        save_path = os.path.join(save_dir, name + '.fasta') 
        with open(save_path, 'w') as fw:
            x = np.load(os.path.join(args.data_dir, name + '.npz'))
            heavy_str_seq, light_str_seq = x.get('heavy_str_seq',''), x.get('light_str_seq', '')
            if heavy_str_seq and light_str_seq:
                if concat:
                    fw.write(f'>{name}\n{heavy_str_seq}{"G" * 50}{light_str_seq}\n')
                else:
                    fw.write(f'>{name}heavy \n{heavy_str_seq}\n>{name} light\n{light_str_seq}\n')
            else:
                fw.write(f'>{name}\n{heavy_str_seq}\n')

def cluster(df, out_dir, suffix, seq_id, save_fasta=False):
    # cluster the train list
    seq_file = os.path.join(out_dir, f'{suffix}_H.fasta')
    _save_heavy_fasta(seq_file, df)
    
    out_file = os.path.join(out_dir, 'cluster_' + suffix)
    tmp_dir = os.path.join(out_dir, 'tmp_cluster_' + suffix)
    subprocess.run(
            ['mmseqs', 'easy-cluster', 
            '--min-seq-id', str(seq_id),
            '--cov-mode', '1',
            '--seq-id-mode', '2',
            seq_file, out_file, tmp_dir, 
            '--cluster-reassign'], shell=False)
    
    out_df = pd.read_csv(out_file + '_cluster.tsv', header=None, names=['n1', 'n2'], sep='\t')

    cluster_file = os.path.join(out_dir, f'{suffix}_cluster.idx')
    with open(cluster_file, 'w') as fw:
        for c, g in out_df.groupby('n1'):
            n2 = list(g['n2'])
            random.shuffle(n2)
            fw.write(' '.join(n2) + '\n')

    out_df = pd.read_csv(out_file + '_cluster.tsv', header=None, names=['n1', 'n2'], sep='\t')
    out_df['n1'].drop_duplicates().to_csv(os.path.join(out_dir, suffix + '.idx'), index=False, header=None)
    
    if save_fasta:
        df = df[df['name'].isin(list(out_df['n1'].drop_duplicates()))]
        _save_glycine_fasta(os.path.join(out_dir, f'{suffix}_glycine.fasta'), df)
        _save_seperate_fasta(os.path.join(out_dir, f'{suffix}_fasta'), df)

def main(args):
    train_resol_thres = 5.0
    test_resol_thres = 3.0
    overlap_seq_id = 0.95
    train_seq_id = 0.95
    test_seq_id = 0.95

    out_dir = f'{args.out_dir}/train_resol{train_resol_thres}_test_resol{test_resol_thres}_overlap{overlap_seq_id}_train{train_seq_id}_test{test_seq_id}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(args.summary_file, sep='\t', parse_dates=['date'])

    df = df[df['method'].isin(['X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY'])]
    
    df = df.fillna({'Hchain':'', 'Lchain':''})
    df = df[df['Hchain'] != '']
    
    df['name'] = df.apply((lambda x : '_'.join([x['pdb'], x['Hchain'], x['Lchain']])), axis=1)
    df = df[df['name'].apply(lambda x: os.path.exists(os.path.join(args.data_dir, x + '.npz')))]

    def _resolution(pdb):
        path = os.path.join(args.data_dir, pdb + '.json')
        if not os.path.exists(path):
            return np.nan
        
        with open(path) as f:
            info = json.load(f)
        return info['resolution']

    df['resolution'] = df['pdb'].apply(lambda x : _resolution(x))

    df = df[~df['resolution'].isna()]
    df = df[df['resolution'] < train_resol_thres]
    
    print(f'resolution <= {test_resol_thres}', df.shape[0])

    def _cdrh3_seq_len(name):
        data = np.load(os.path.join(args.data_dir, name + '.npz'))
        cdr_def = data['heavy_cdr_def']

        cdrh3_seq_len = np.sum(cdr_def == 5)

        return cdrh3_seq_len

    cdrh3_len_thres = 25
    df['cdrh3_seq_len'] = df['name'].apply(_cdrh3_seq_len)

    df = df[df['cdrh3_seq_len'] <= cdrh3_len_thres]
    print(f'cdrh3 len <= {cdrh3_len_thres}', df.shape[0])
    
    df.to_csv(os.path.join(out_dir, 'S1_summary_filter.csv'), sep='\t', index=False)

    test = df[df['date'] >= '2022-01-01']
    test = test[test['resolution'] < test_resol_thres]
    
    train = df[df['date'] < '2022-01-01']
    
    train[['name']].to_csv(os.path.join(out_dir, 'S1_train.idx'), index=False, header=False)
    test[['name']].to_csv(os.path.join(out_dir, 'S1_test.idx'), index=False, header=False)
    print('train and test', train.shape[0], test.shape[0])

    train_file = os.path.join(out_dir, f'S1_train.fasta')
    test_file = os.path.join(out_dir, f'S1_test.fasta')
    _save_heavy_fasta(train_file, train)
    _save_heavy_fasta(test_file, test)

    out_file = os.path.join(out_dir, 'test_search_train')
    tmp_dir = os.path.join(out_dir, 'tmp_search')
    subprocess.run(
            ['mmseqs', 'easy-search', 
            '--min-seq-id', str(overlap_seq_id),
            '--seq-id-mode', '2',
            test_file, train_file, out_file, tmp_dir], shell=False)

    overlap_test_names = list(pd.read_csv(out_file, header=None, sep='\t')[0].drop_duplicates())
    print('overlap', len(overlap_test_names))

    test = test[~test['name'].isin(overlap_test_names)]
    print('test, fiter overlap', test.shape)

    cluster(train, out_dir, 'train', train_seq_id, save_fasta=False)
    
    cluster(test, out_dir, 'test', test_seq_id, save_fasta=True)
    
    test_pair = test[test['Lchain'] != ''] 
    test_heavy = test[test['Lchain'] == ''] 

    cluster(test_pair, out_dir, 'test_pair', test_seq_id, save_fasta=True)
    cluster(test_heavy, out_dir, 'test_heavy', test_seq_id, save_fasta=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_file', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    
    main(args)
