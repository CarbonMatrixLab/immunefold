import json
import argparse
import hydra

import pandas as pd
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from pytorch3d.transforms.rotation_conversions import quaternion_to_axis_angle, axis_angle_to_quaternion

from carbonmatrix.trainer.base_dataset import collate_fn_struc
from carbonmatrix.trainer.dataset import StructureDatasetNpzIO as StructureDataset 
from carbonmatrix.trainer import utils
from carbonmatrix.data.base_dataset import  TransformedDataLoader
from carbonmatrix.model import quat_affine 

@hydra.main(version_base=None, config_path="config", config_name="diffuser")
def main(cfg):
    utils.setup_ddp()
    world_rank = utils.get_world_rank()
    local_rank = utils.get_local_rank()
    device = utils.get_device()

    dataset = StructureDataset(cfg.train_data, cfg.train_name_idx, cfg.max_seq_len)
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)

    dataloader = TransformedDataLoader(
            dataset, feats=cfg.transforms, device=3,
            collate_fn = collate_fn_struc,
            sampler=sampler,
            batch_size=1,
            )

    for i, x in enumerate(dataloader):
        print('batch', local_rank, x['name'])
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
        
        rot, tran = x['rigidgroups_gt_frames']
        rot, tran = rot[:,:,0], tran[:,:,0]
        mask = x['rigidgroups_gt_exists'][:,:,0]
        
        '''
        axis_angle = rotation_to_so3vec(rot)
        angle = torch.sqrt(torch.sum(torch.square(axis_angle), axis=-1))
        print('time', data['t'])
        print('angle', angle)

        rot0 = so3vec_to_rotation(axis_angle)
        print(rot0)
        print(rot)
        '''

        print(rot.shape, tran.shape, mask.shape)
        quat = matrix_to_quaternion(rot)
        print('quat sign', quat[...,0])
        
        '''
        rot1 = quaternion_to_matrix(quat)
        rot2 = quat_affine.quat_to_rot(quat)

        delta = torch.sqrt(torch.sum(torch.square(rot2 - rot1), axis=[-1,-2]))
        delta = torch.sum(delta * mask)
        print('delta', delta, torch.sum(mask))

        print(quat)
        print('rot0')
        print(rot)
        print('rot1')
        print(rot1)
        print('rot2')
        print(rot2)
        '''
        
        '''
        axis_angle = quaternion_to_axis_angle(quat)
        angle = torch.sqrt(torch.sum(torch.square(axis_angle), axis=-1))
        print('time', data['t'])
        print('angle', angle)

        quat1 = axis_angle_to_quaternion(axis_angle)

        print('quat0')
        print(quat)

        print('quat1')
        print(quat1)
        '''
        trans_t = x['rigids_t'][1]
        print('trans_t', trans_t)
        print(torch.mean(trans_t))
        print(torch.std(trans_t))
        break

if __name__ == '__main__':
    main()
