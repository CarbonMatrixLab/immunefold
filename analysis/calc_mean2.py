import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], sep='\t')
head = [n for n in df.columns if n.endswith('_rmsd')]
metric = [df[n].mean() for n in head]
print(df['heavy_fr4_rmsd'].mean())
print('\t'.join(head))
print('\t'.join([f'{m:.2f}' for m in metric]))
