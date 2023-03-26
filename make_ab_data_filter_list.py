import os
import pandas as pd
import numpy as np
import argparse
import logging

def main(args):
    df = pd.read_csv(args.summary_file, sep='\t', parse_dates=['date'])
    print('orignal', df.shape[0])
    
    df = df.fillna({'Hchain':'', 'Lchain':''})
    df['name'] = df.apply((lambda x : '_'.join([x['pdb'], x['Hchain'], x['Lchain']])), axis=1)
    
    df = df[df['name'].apply(lambda x: os.path.exists(os.path.join(args.data_dir, x + '.npz')))]
    print('data', df.shape[0])
    
    df = df[df['resolution'] != 'NOT']
    print('with resolution', df.shape[0])

    df['resolution'] = df['resolution'].apply(lambda x: min(map(float, str(x).split(','))))

    resol_thres = 5.
    df = df[df['resolution'] <= resol_thres]
    print(f'resolution <= {resol_thres}', df.shape[0])

    def _cdrh3(name):
        data = np.load(os.path.join(args.data_dir, name + '.npz'))
        
        heavy_str_seq, light_str_seq, cdrh3_seq = '', '', ''
        
        if 'heavy_str_seq'  in data:
            cdr_def = data['heavy_cdr_def']
            heavy_str_seq = str(data['heavy_str_seq'])
            cdrh3_seq = ''.join([s for c, s in zip(cdr_def, heavy_str_seq) if c == 5])
        
        if 'light_str_seq'  in data:
            light_str_seq = str(data['light_str_seq'])

        return heavy_str_seq, light_str_seq, cdrh3_seq

    cdrh3_len_thres = 80
    df['heavy_str_seq'], df['light_str_seq'], df['cdrh3_seq'] = zip(*df['name'].apply(_cdrh3))

    df = df[df['cdrh3_seq'].apply(lambda x : len(x) <= cdrh3_len_thres)]
    print(f'cdrh3 len <= {cdrh3_len_thres}', df.shape[0])

    
    df.to_csv(os.path.join(args.out_dir, 'S1_summary_filter.csv'), sep='\t', index=False)

    test = df[df['date'] >= '2022-01-01']
    #test = test[test['resolution'] < 5.]
    
    train = df[df['date'] < '2022-01-01']
    #train = train[train['resolution'] < 5.]
    
    train[['name']].to_csv(os.path.join(args.out_dir, 'S1_train.idx'), index=False, header=False)
    test[['name']].to_csv(os.path.join(args.out_dir, 'S1_test.idx'), index=False, header=False)
    print('train and test', train.shape[0], test.shape[0])

    def _save_fasta(save_path, df, concat=False):
        def _save(fw, x):
            name = x['name']
            heavy_str_seq, light_str_seq = x['heavy_str_seq'], x['light_str_seq']
            if heavy_str_seq and light_str_seq:
                if concat:
                    fw.write(f'>{name}\n{heavy_str_seq}{"G" * 20}{light_str_seq}\n')
                else:
                    fw.write(f'>{name}heavy \n{heavy_str_seq}\n>{name} light\n{light_str_seq}\n')
            else:
                fw.write(f'>{name}\n{heavy_str_seq}{light_str_seq}\n')
        with open(save_path, 'w') as fw:
            df.apply((lambda x: _save(fw, x)), axis=1)

    _save_fasta(os.path.join(args.out_dir, f'S1_train.fasta'), train)
    _save_fasta(os.path.join(args.out_dir, f'S1_test.fasta'), test)
    
    _save_fasta(os.path.join(args.out_dir, f'S1_train_concat.fasta'), train, concat=True)
    _save_fasta(os.path.join(args.out_dir, f'S1_test_concat.fasta'), test, concat=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_file', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    
    main(args)
