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

    gt_fea = np.load(gt_npz_file)
    gt_coords = np.concatenate([gt_fea['heavy_coords'], gt_fea['light_coords']], axis=0)
    gt_coord_mask = np.concatenate([gt_fea['heavy_coord_mask'], gt_fea['light_coord_mask']], axis=0)
    cdr_def = np.concatenate([gt_fea['heavy_cdr_def'], gt_fea['light_cdr_def']], axis=0)

    str_heavy_seq, str_light_seq = str(gt_fea['heavy_str_seq']), str(gt_fea['light_str_seq'])

    ca_mask = gt_coord_mask[:, 1]
    gt_ca = gt_coords[:,1]

    pred_ca = make_pred_coords(pred_file, len(str_heavy_seq), len(str_light_seq), alg_type)

    assert (gt_ca.shape[0] == pred_ca.shape[0] and gt_ca.shape[0] == cdr_def.shape[0])

    ab_metrics = calc_ab_metrics(gt_ca, pred_ca, ca_mask, cdr_def)

    return ab_metrics

def main(args):
    names = list(pd.read_csv(args.name_idx, names=['name'], header=None)['name'])

    metrics = []
    for i, n in enumerate(names):
        gt_file = os.path.join(args.gt_dir, n + '.npz')
        pred_file = os.path.join(args.pred_dir, n + '.pdb')
        if os.path.exists(gt_file) and os.path.exists(pred_file):
            one_metric = OrderedDict({'name' : n})
            rmsd_metric = make_one(n, gt_file, pred_file, args.alg_type)
            one_metric.update(rmsd_metric)
            metrics.append(one_metric)

    columns = metrics[0].keys()
    metrics = zip(*map(lambda x:x.values(), metrics))

    df = pd.DataFrame(dict(zip(columns,metrics)))

    df = df[df['heavy_cdr3_len'] > 2]
    df = df[df['full_len'] > 200]

    df.to_csv(args.output, sep='\t', index=False)
    df = df.dropna()

    print('total', df.shape[0])
    for r in df.columns:
        if r.endswith('rmsd'):
            rmsd = df[r].values
            mean = np.mean(rmsd)
            print(f'{r:15s} {mean:6.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--name_idx', type=str, required=True)
    parser.add_argument('-t', '--alg_type', type=str, choices=['igfold', 'esmfold', 'omegafold', 'carbonmatrix'], required=True)
    parser.add_argument('-g', '--gt_dir', type=str, required=True)
    parser.add_argument('-p', '--pred_dir', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)

    args = parser.parse_args()

    main(args)

