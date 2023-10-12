import pandas as pd
import sys

summary_file = sys.argv[1]
cluster_file = sys.argv[2]
out_file = sys.argv[3]

df = pd.read_csv(summary_file, sep='\t')
print(df.shape)
df = df.dropna(subset=['light_str_seq', 'heavy_str_seq'])
print(df.shape)
#df = df.fillna({'light_str_seq' : '', 'heavy_str_seq' : ''})

clu = pd.read_csv(cluster_file, sep='\t', names=['n1', 'n2'])
print(clu['n1'].nunique())

clu_dict = dict(zip(clu['n2'], clu['n1']))

df = df[df['name'].isin(clu_dict)]

df['clu'] = df['name'].apply(lambda x : clu_dict[x])

print(df.head())
df = df.loc[df.groupby('clu')['resolution'].idxmin()]#.reset_index()#.merge(df, how='left', on='clu')

with open(out_file, 'w') as fw:
    for n in df['name']:
        fw.write(n + '\n')
df.to_csv(out_file + '.csv', sep='\t', index=False)
