import os
import argparse
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd

from carbonmatrix.common.ab.metrics import calc_ab_metrics
from carbonmatrix.common import residue_constants

from Bio.PDB.PDBParser import PDBParser

def read_fasta(fasta_file):
    with open(fasta_file) as f:
        lines = f.readlines()
    return lines[1].strip(), lines[3].strip()

def make_pred_coords(pdb_file, heavy_len, light_len, alg_type):
    sep_pad_num = 50

    parser = PDBParser(QUIET=1)
    model = parser.get_structure('pdb', pdb_file)[0]

    if alg_type in ['omegafold']:
        residues = list(model.get_residues())
        residues = residues[:heavy_len] + residues[heavy_len+sep_pad_num:heavy_len+sep_pad_num+light_len]
    elif alg_type in ['esmfold']:
        residues = list(model['A'].get_residues()) + list(model['B'].get_residues())
    else:
        residues = list(model['H'].get_residues()) + list(model['L'].get_residues())

    coords = np.zeros((len(residues), 3))

    for i, r in enumerate(residues):
        coords[i] = r['CA'].get_coord()

    return coords

def make_one(name, gt_npz_file, pred_file, alg_type):

    gt_fea = np.load(gt_npz_file, allow_pickle=True).item()
    heavy_chain, light_chain = gt_fea['heavy_chain'], gt_fea['light_chain']

    gt_coords = np.concatenate([heavy_chain['coords'], light_chain['coords']], axis=0)
    gt_coord_mask = np.concatenate([heavy_chain['coord_mask'], light_chain['coord_mask']], axis=0)
    cdr_def = np.concatenate([heavy_chain['region_index'], light_chain['region_index']], axis=0)

    str_heavy_seq = str(heavy_chain['seq'])
    str_light_seq = str(light_chain['seq'])

    ca_mask = gt_coord_mask[:, 1]
    gt_ca = gt_coords[:,1]
    
    N = len(str_heavy_seq) + len(str_light_seq)
    assert(N==gt_ca.shape[0] and N==ca_mask.shape[0] and N==cdr_def.shape[0])

    pred_ca = make_pred_coords(pred_file, len(str_heavy_seq), len(str_light_seq), alg_type)

    assert (N == pred_ca.shape[0])

    ab_metrics = calc_ab_metrics(gt_ca, pred_ca, ca_mask, cdr_def, remove_middle_residues=True)

    return ab_metrics

def main(args):
    names = list(pd.read_csv(args.name_idx, names=['name'], header=None)['name'])

    metrics = []
    for i, n in enumerate(names):
        gt_file = os.path.join(args.gt_dir, n + '.npy')
        pred_file = os.path.join(args.pred_dir, n + '.pdb')
        if os.path.exists(gt_file) and os.path.exists(pred_file):
            one_metric = OrderedDict({'name' : n})
            rmsd_metric = make_one(n, gt_file, pred_file, args.alg_type)
            one_metric.update(rmsd_metric)
            metrics.append(one_metric)

    columns = metrics[0].keys()
    metrics = zip(*map(lambda x:x.values(), metrics))

    df = pd.DataFrame(dict(zip(columns,metrics)))
    print('all', df.shape)
    print(df['heavy_cdr3_coverage'].describe())

    len_thres=3
    df = df[df['heavy_cdr3_len'] >=len_thres]
    print('length >= ', len_thres, df.shape)
    
    df = df[df['heavy_cdr3_coverage'] >= 1.0]
    print('coverage=1.0', df.shape)
    
    df.to_csv(args.output, sep='\t', index=False)
    df = df.dropna()

    print('final total', df.shape[0])
    for r in df.columns:
        if r.endswith('rmsd'):
            rmsd = df[r].values
            mean = np.mean(rmsd)
            std = np.std(rmsd)
            max_ = np.max(rmsd)
            print(f'{r:15s} {mean:6.2f} {std:6.2f} {max_:6.2f}')
    
    #print(df['heavy_cdr3_len'].describe())
    #print(df['heavy_cdr3_rmsd'].describe())
    #print(df['full_rmsd'].describe())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--name_idx', type=str, required=True)
    parser.add_argument('-t', '--alg_type', type=str, choices=['igfold', 'esmfold', 'omegafold', 'carbonmatrix'], required=True)
    parser.add_argument('-g', '--gt_dir', type=str, required=True)
    parser.add_argument('-p', '--pred_dir', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)

    args = parser.parse_args()

    main(args)

