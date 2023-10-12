import sys
import os
import logging
import argparse
import dataclasses
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../carbonmatrix'))
from carbonmatrix.common import residue_constants
from carbonmatrix.data.seq import str_seq_to_index

@dataclasses.dataclass(frozen=True)
class SeqItem:
    head: str
    str_seq : str
    aligned_str_seq : str
    length : int
    seq_to_column : Sequence[int]
    column_to_seq : Sequence[int]

@dataclasses.dataclass(frozen=True)
class MSA:
    query_seq: str
    num_columns: int
    num_rows:int
    data: Sequence[SeqItem]

def parse_a3m(a3m_file):
    heads, aligned_str_seqs = [], []
    
    with open(a3m_file) as f:
        head, seq = None, ''
        for line in f:
            if line.startswith('>'):
                if head is not None:
                    heads.append(head)
                    aligned_str_seqs.append(seq)
                head, seq = line.strip(), ''
            else:
                seq = seq + line.strip()
    
        if head is not None:
            heads.append(head)
            aligned_str_seqs.append(seq)

    assert len(heads) == len(aligned_str_seqs)
    
    str_seqs = [
            seq.replace('.', '').replace('-', '').upper() for seq in aligned_str_seqs
            ]
    
    aligned_seqs = np.array([[residue_constants.alpha_to_int.get(a, residue_constants.UNK) for a in str_seq if (a.isupper() or a == '-')] for str_seq in aligned_str_seqs])

    return str_seqs, aligned_str_seqs, aligned_seqs

def split_calc_seq_weights(seqs, thres):
    M, L = seqs.shape
    num_each = 10000
    split_indices = np.arange(num_each, M, num_each)
    sub = np.split(seqs, split_indices, axis=0)
    counts = np.zeros((M,))

    for i in range(len(sub)):
        for j in range(i, len(sub)):
            logger.debug(f'split pair {i}, {j} when M= {M}')
            pair_dis = np.sum(np.equal(sub[i][:,None], sub[j][None,:]), axis=-1)
            pair_dis = (pair_dis > thres * L)
            
            left = np.sum(pair_dis, axis=1)
            right = np.sum(pair_dis, axis=0)
            
            counts[i * num_each : i * num_each + left.shape[0]] += left
            
            if i != j:
                counts[j * num_each : j * num_each + right.shape[0]] += right
    weights = 1. / counts

    return weights

def calc_seq_weights(seqs, thres):
    M, L = seqs.shape
   
    if M < 10000:
        pair_dis = np.sum(np.equal(seqs[None, :], seqs[:, None]), axis=-1)
        pair_dis = np.sum(pair_dis > thres * L, axis=1)
        weights = 1. / pair_dis
        return weights
    else:
        return split_calc_seq_weights(seqs, thres)

def calc_aligned_indices(aligned_str_seq):
    aligned_n, seq_n = 1, 1
    indices_to_aligned_column, indices_from_aligned_column = [], []
    
    for a in aligned_str_seq:
        if a.isupper():
            indices_to_aligned_column.append(aligned_n)
            indices_from_aligned_column.append(seq_n)
            aligned_n += 1
            seq_n += 1
        elif a.islower():
            indices_to_aligned_column.append(0)
            seq_n += 1
        elif a == '-':
            indices_from_aligned_column.append(0)
            aligned_n += 1
    
    return indices_to_aligned_column, indices_from_aligned_column

def _parse_a3m_seq(head, seq):
    str_seq, aligned_str_seq = [], []
    seq_to_column, column_to_seq = [], []

    column_idx, seq_idx = 0, 0

    for aa in seq.strip():
        if aa == '-':
            column_to_seq.append(-1)
            column_idx += 1
            aligned_str_seq.append(aa)
        elif aa.islower():
            seq_to_column.append(-1)
            str_seq.append(aa.upper())
            seq_idx += 1
        else:
            column_to_seq.append(seq_idx)
            seq_to_column.append(column_idx)
            str_seq.append(aa.upper())
            seq_idx += 1
            column_idx += 1
            aligned_str_seq.append(aa)

    str_seq = ''.join(str_seq)
    aligned_str_seq = ''.join(aligned_str_seq)

    return SeqItem(
            head = head,
            str_seq=str_seq,
            aligned_str_seq=aligned_str_seq,
            length=len(str_seq),
            column_to_seq = column_to_seq,
            seq_to_column = seq_to_column)

def parse_a3m(a3m_file):
    data = []
    with open(a3m_file) as f:
        head, seq = None, ''
        for line in f:
            if line.strip() == '':
                continue
            if line.startswith('>'):
                if head is not None:
                    data.append((head, seq))
                head = line.strip()[1:]
                seq = ''
            else:
                seq += line.strip()
        if head is not None:
            data.append((head, seq))
    query_seq = data[0][1]
    
    parsed_data = [_parse_a3m_seq(*x) for x in data]

    return MSA(query_seq=query_seq,
            num_columns = len(query_seq),
            num_rows = len(parsed_data),
            data = parsed_data)

def save_fasta(msa, weights, out_fasta_file):
    assert(weights.shape[0] == msa.num_rows)

    with open(out_fasta_file, 'w') as fw:
        for i, m in enumerate(msa.data):
            fw.write(f'>{m.head}\t{weights[i]}\n{m.str_seq}\n')

def main(args):
    msa = parse_a3m(args.in_a3m_file)


    seqs = np.array([str_seq_to_index(x.aligned_str_seq) for x in msa.data])

    weights = calc_seq_weights(seqs, thres=0.9)
    
    save_fasta(msa, weights, args.out_fasta_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_a3m_file', type=str, required=True)
    parser.add_argument('--out_fasta_file', type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    main(args)
