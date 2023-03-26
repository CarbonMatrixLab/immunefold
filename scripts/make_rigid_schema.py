import json
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
import json

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

device=torch.device('cuda')
restype_atom37_rigid_group_positions = torch.zeros([21, 37, 3], device=device)
restype_atom37_rigid_group_count = torch.zeros([21, 37], device=device)

def compute_one(batch):
    global restype_atom37_rigid_group_positions
    global restype_atom37_rigid_group_count

    aatype = batch['seq']
    atom37_gt_pos = batch['atom37_gt_positions'] # (b, l, 37, 3)
    rigid_group_idx = batched_select(torch.tensor(residue_constants.restype_rigidgroup_base_atom37_idx, device=device), aatype)
    rigid_group_positions = batched_select(atom37_gt_pos, rigid_group_idx, batch_dims=2) # (b, l, 8, 3, 3)
    atom_group_idx = batched_select(torch.tensor(residue_constants.restype_atom37_to_rigid_group, device=device), aatype) # (b, l, 37)
    atom_group_positions = batched_select(rigid_group_positions, atom_group_idx, batch_dims=2) # (b, l, 37, 3, 3)

    mask = torch.logical_and(
            batch['atom37_gt_exists'],
            torch.all(torch.logical_not(torch.logical_xor(batch['atom37_atom_exists'], batch['atom37_gt_exists'])), dim=-1, keepdim=True)) # (b, l, 37)

    local_frames = r3.rigids_from_3_points(
            atom_group_positions[:, :, :, 0, :],
            atom_group_positions[:, :, :, 1, :],
            atom_group_positions[:, :, :, 2, :])
  
    ref_positions = r3.rigids_mul_vecs(r3.invert_rigids(local_frames), atom37_gt_pos)
    ref_positions = ref_positions.masked_fill(~mask[...,None], 0.) # (b l 37 3)
    
    aatype_onehot = F.one_hot(aatype, num_classes=21).to(device=device, dtype=torch.float32) # (b, l, 21)
    restype_atom37_rigid_group_positions.add_(
            torch.sum(ref_positions[:,:,None] * aatype_onehot[:,:,:,None,None], dim=[0, 1]))
    restype_atom37_rigid_group_count.add_(
            torch.sum(mask[:,:,None] * aatype_onehot[:,:,:,None], dim=[0,1]))
    

def main(args):
    
    device = torch.device('cuda:0')

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
            batch_size = 256, data_type='general')

    for data in train_loader:
        compute_one(data)
    
    results = restype_atom37_rigid_group_positions / restype_atom37_rigid_group_count[...,None]

    
    results = results.to('cpu').numpy()
    schema = defaultdict(list)
    for restype, groups in residue_constants.rigid_group_atom_positions.items():
        restype_idx = residue_constants.resname_to_idx[restype]
        for atom, group_idx, _ in groups:
            atom_idx = residue_constants.atom_order[atom]
            pos = list(results[restype_idx, atom_idx])
            schema[restype].append([atom, group_idx, tuple(round(float(x), 3) for x in pos)])

    print(schema)

    with open('./abfold/common/new_rigid_schema.json', 'w') as fw:
        json.dump(schema, fw)
    with open('./abfold/common/old_rigid_schema.json', 'w') as fw:
        json.dump(residue_constants.rigid_group_atom_positions, fw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_idx', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_features', type=str, required=True)
    args = parser.parse_args()
    main(args)
