import os
import pandas as pd
import numpy as np

def main():
    base_dir = '../raw_ab_data'
    
    df = pd.read_csv(os.path.join(base_dir, 'raw_all.idx'), header=None, names=['name'])
    
    idx = list(df['name'])
    
    path = os.path.join(base_dir, 'sabdab_summary_all_20221024.tsv')

    df = pd.read_csv(path, sep='\t', parse_dates=['date']) #dtype={'Hchain':str,'Lchain':str})
    
    df = df.dropna(subset=['Hchain', 'Lchain'])

    def _get_name(x):
        heavy_chain_id, light_chain_id = x['Hchain'], x['Lchain']
        if heavy_chain_id.islower() and heavy_chain_id.upper() == light_chain_id:
            heavy_chain_id = heavy_chain_id.upper()
        elif light_chain_id.islower() and light_chain_id.upper() == heavy_chain_id:
            light_chain_id = light_chain_id.upper()
        return '_'.join([x['pdb'], heavy_chain_id, light_chain_id])
    df['name'] = df.apply(_get_name, axis=1)
    df = df[df['name'].isin(idx)]
    df = df.drop_duplicates(['name'])

    def _resolution(x):
        if x == 'NOT':
            return -1.0
        return min([float(xx.strip()) for xx in x.split(',')])

    df['resolution'] = df['resolution'].apply(_resolution)
    df = df[df['resolution'] < 5.]

    def _cdrh3(name):
        data = np.load(os.path.join(base_dir, 'npz', name + '.npz'))
        cdr_def = data['heavy_cdr_def']
        seq = str(data['heavy_str_seq'])
        cdrh3_seq = []
        for c, s in zip(cdr_def, seq):
            if c == 5:
                cdrh3_seq.append(s)
        cdrh3_seq = ''.join(cdrh3_seq)
        cdrh3_seq_len = np.sum(cdr_def==5) 
        assert cdrh3_seq_len == len(cdrh3_seq)
        return cdrh3_seq_len, cdrh3_seq

    df['cdrh3_seq_len'], df['cdrh3_seq'] = zip(*df['name'].apply(_cdrh3))

    df = df[df['cdrh3_seq_len'] <= 25]
    
    df.to_csv(os.path.join(base_dir, 'summary_filter_step1.csv'), sep='\t', index=False)

    #test = df[df['date'] > '2021-07-01']
    test = df[df['date'] >= '2022-01-01']
    #test = test[test['resolution'] < 5.]
    
    train = df[df['date'] < '2022-01-01']
    #train = train[train['resolution'] < 5.]
    
    train[['name']].to_csv(os.path.join(base_dir, 'raw_train.idx'), index=False, header=False)
    test[['name']].to_csv(os.path.join(base_dir, 'raw_test.idx'), index=False, header=False)
    print('train and test', train.shape[0], test.shape[0])

    def _read_fasta(path):
        with open(path) as f:
            lines = f.readlines()
            if len(lines) > 2:
                return lines[1].strip(), lines[3].strip()
            else:
                return lines[1].strip(), lines[0].strip().split()[1]
    
    def _save_fasta(save_path, idxs, single_chain=False):
        with open(save_path, 'w') as fw:
            for idx in idxs:
                heavy_seq, light_seq  = _read_fasta(os.path.join(base_dir, 'npz', idx + '.fasta'))
                if single_chain:
                    fw.write(f'>{idx} Heavy\n{heavy_seq}\n>{idx} Light\n{light_seq}\n')
                else:
                    fw.write(f'>{idx}\n{heavy_seq}{"G" * 10}{light_seq}\n')

    _save_fasta(os.path.join(base_dir, f'train.fasta'), list(train['name']))
    _save_fasta(os.path.join(base_dir, f'test.fasta'), list(test['name']))
    
    _save_fasta(os.path.join(base_dir, f'train_cat.fasta'), list(train['name']), single_chain=True)
    _save_fasta(os.path.join(base_dir, f'test_cat.fasta'), list(test['name']), single_chain=True)

if __name__ == '__main__':
    main()
