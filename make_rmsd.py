import os
import argparse
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd

from abfold.preprocess.parser import parse_pdb, get_struc2seq_map
from abfold.preprocess.parser import make_struc_feature
from abfold.common.ab_utils import get_antibody_regions, calc_ab_metrics
from abfold.common import residue_constants

def read_fasta(fasta_file):
    with open(fasta_file) as f:
        lines = f.readlines()
    return lines[1].strip(), lines[3].strip()

def make_pred_feature(str_seq, structure):
    n = len(str_seq)
    print(n, len(structure), 'len')
    assert n > 0
    assert n == len(structure)

    coords = np.zeros((n, 14, 3), dtype=np.float32)
    coord_mask = np.zeros((n, 14), dtype=bool)
    
    for seq_idx, residue in enumerate(structure):
            if residue.resname not in residue_constants.restype_name_to_atom14_names:
                continue
            res_atom14_list = residue_constants.restype_name_to_atom14_names[residue.resname]
            for atom in residue.get_atoms():
                if atom.id not in res_atom14_list:
                    continue
                atom14idx = res_atom14_list.index(atom.id)
                coords[seq_idx, atom14idx] = atom.get_coord()
                coord_mask[seq_idx, atom14idx]= True
    
    feature = dict(str_seq=str_seq,
            coords=coords,
            coord_mask=coord_mask)

    return feature

def make_one(name, seq_file, gt_npz_file, pred_file):
    str_heavy_seq, str_light_seq = read_fasta(seq_file)

    N = len(str_heavy_seq) + len(str_light_seq)
   
    gt_fea = np.load(gt_npz_file)
    gt_coords = np.concatenate([gt_fea['light_coords'], gt_fea['heavy_coords']], axis=0)
    gt_coord_mask = np.concatenate([gt_fea['light_coord_mask'], gt_fea['heavy_coord_mask']], axis=0)
    cdr_def = np.concatenate([gt_fea['light_cdr_def'], gt_fea['heavy_cdr_def']], axis=0)

    struc = parse_pdb(pred_file)
    heavy_struc, light_struc = struc['H'], struc['L']
    heavy_fea = make_pred_feature(str_heavy_seq, heavy_struc)
    light_fea = make_pred_feature(str_light_seq, light_struc)
    pred_coords = np.concatenate([light_fea['coords'], heavy_fea['coords']], axis=0)

    gt_ca, pred_ca = gt_coords[:,1], pred_coords[:,1]
    ca_mask = gt_coord_mask[:, 1]
    
    assert (gt_ca.shape[0] == pred_ca.shape[0] and gt_ca.shape[0] == cdr_def.shape[0])

    ab_metrics = calc_ab_metrics(gt_ca, pred_ca, ca_mask, cdr_def)

    return ab_metrics

def main(args):
    names = list(pd.read_csv(args.name_idx, names=['name'], header=None)['name'])

    metrics = []
    for i, n in enumerate(names):
        seq_file = os.path.join(args.fasta_dir, n + '.fasta')
        gt_file = os.path.join(args.gt_dir, n + '.npz')
        pred_file = os.path.join(args.pred_dir, n + '.pdb')
        print(seq_file)
        print(gt_file)
        print(pred_file)
        if os.path.exists(seq_file) and os.path.exists(gt_file) and os.path.exists(pred_file):
            one_metric = OrderedDict({'name' : n})
            rmsd_metric = make_one(n, seq_file, gt_file, pred_file)
            one_metric.update(rmsd_metric)
            print(one_metric)
            metrics.append(one_metric)

    columns = metrics[0].keys()
    metrics = zip(*map(lambda x:x.values(), metrics))
    df = pd.DataFrame(dict(zip(columns,metrics)))
    df.to_csv(args.output, sep='\t', index=False)

    print(df.describe())

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('-c', '--name_idx', type=str, required=True)
  parser.add_argument('-s', '--fasta_dir', type=str, required=True,
          help='fasta dir')
  parser.add_argument('-g', '--gt_dir', type=str, required=True,
      help='groud truth pdb dir')
  parser.add_argument('-p', '--pred_dir', type=str, required=True,
      help='predict pdb dir')
  parser.add_argument('-o', '--output', type=str, required=True,
      help='output dir, default=\'.\'')
  args = parser.parse_args()


  main(args)
