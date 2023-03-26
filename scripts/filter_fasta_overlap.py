import os
import sys
import pandas as pd

input_fasta = sys.argv[1]
out_fasta = sys.argv[2]
input_list = sys.argv[3]

with open(input_list) as f:
    names = [line.strip() for line in f]

with open(input_fasta) as f, open(out_fasta, 'w') as fw:
    lines = f.readlines()
    N = len(lines) // 2
    for i in range(N):
        n = lines[i * 2].split()[0][1:]
        if n in names:
            continue
        fw.write(lines[i * 2])
        fw.write(lines[i * 2 + 1])
