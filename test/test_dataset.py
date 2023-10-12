import json
import argparse

import pandas as pd
import numpy as np
import torch

from abfold.common import residue_constants
from abfold.model import r3
from abfold.trainer.geometry import (
        atom37_to_frames,
        atom37_to_torsion_angles)
from abfold.model.atom import (
        torsion_angles_to_frames,
        frames_and_literature_positions_to_atom14_pos)
from abfold.trainer import dataset
from abfold.trainer.loss import compute_renamed_ground_truth
from abfold.model.utils import batched_select

def test_geometry(batch):

    bb_frames = r3.rigids_op(batch['rigidgroups_gt_frames'], lambda x: x[:, :, 0])

    all_frames_to_global = torsion_angles_to_frames(batch['seq'], bb_frames, batch['torsion_angles_sin_cos'])

    pred_pos = frames_and_literature_positions_to_atom14_pos(batch['seq'], all_frames_to_global)
    
    mask = torch.logical_and(
            batch['atom14_gt_exists'],
            torch.all(torch.logical_not(torch.logical_xor(batch['atom14_atom_exists'],batch['atom14_gt_exists'])), dim=-1, keepdim=True))

    gt_pos = batch['atom14_gt_positions']
    
    batch_alt = compute_renamed_ground_truth(batch, pred_pos)
    alt_gt_pos = batch_alt['renamed_atom14_gt_positions']

    r1 = torch.sum(torch.square(gt_pos - pred_pos), dim=-1) * mask
    r2 = torch.sum(torch.square(alt_gt_pos - pred_pos), dim=-1) * mask
    r = torch.minimum(r1, r2)

    rmsd = torch.mean(torch.sqrt(torch.sum(r, dim=[1, 2]) / torch.sum(mask, dim=[1, 2])))
    print(torch.mean(rmsd))
    
    atom_r = batched_select(r, batch['residx_atom37_to_atom14'], batch_dims=2)
    mask = batched_select(mask, batch['residx_atom37_to_atom14'], batch_dims=2)
    mask = torch.logical_and(mask, batch['atom37_atom_exists'])
    
    rmsd = torch.sqrt(torch.sum(atom_r, dim=[0, 1]) / torch.sum(mask, dim=[0, 1]))
    #print(rmsd)

def main(args):
    
    device = torch.device('cuda:0')
    print(residue_constants.restype_rigidgroup_mask, 'mask')

    with open(args.model_features, 'r', encoding='utf-8') as f:
        feats = json.load(f)
        for i in range(len(feats)):
            feat_name, feat_args = feats[i]
            if 'device' in feat_args and feat_args['device'] == '%(device)s':
                feat_args['device'] = device
                feats[i] = (feat_name, feat_args)
    
    name_idx = list(pd.read_csv(args.name_idx, sep='\t', dtype={'name':str})['name'])

    train_loader = dataset.load(
            data_dir = args.data_dir,
            name_idx = name_idx,
            feats = feats,
            batch_size = 32, data_type='general')

    for data in train_loader:
        test_geometry(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_idx', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_features', type=str, required=True)
    args = parser.parse_args()
    main(args)
