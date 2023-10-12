import sys
import pandas as pd

input_file = sys.argv[1]
out_file = sys.argv[2]

df = pd.read_csv(input_file, header=None, names=['n1', 'n2'], sep='\t')
print(df.head())

with open(out_file, 'w') as fw:
    for c, g in df.groupby('n1'):
        fw.write(' '.join(list(g['n2'])) + '\n')
