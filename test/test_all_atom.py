import sys
import torch
from einops import rearrange
import numpy as np

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Chain import Chain as PDBChain

from abfold.utils import Kabsch
from abfold.model.utils import batched_select
from abfold.common import residue_constants
from abfold.common.utils import str_seq_to_index
from abfold.trainer.geometry import (
    atom37_to_frames,
    atom37_to_torsion_angles)
from abfold.model.atom import (
    torsion_angles_to_frames,
    frames_and_literature_positions_to_atom14_pos)
from abfold.model import r3

def create_batch_from_pdb(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('id', pdb_file)
    residues = [x for x in structure.get_residues()]
    num_residue = len(residues)

    atom37_coords = np.zeros((num_residue, 37, 3), dtype=np.float32)
    atom37_coord_mask = np.zeros((num_residue, 37), dtype=bool)
    str_seq = []

    for i, r in enumerate(residues):
        str_seq.append(residue_constants.restype_3to1.get(
            r.resname, residue_constants.restypes_with_x[-1]))

        for atom in r.get_atoms():
            atom_idx = residue_constants.atom_order[atom.id]
            atom37_coords[i, atom_idx] = atom.get_coord()
            atom37_coord_mask[i, atom_idx]= True
    
    aatype = str_seq_to_index(''.join(str_seq))
   
    # build the batch data
    aatype = torch.tensor(aatype)[None]
    atom37_coords = torch.tensor(atom37_coords)[None] 
    atom37_coord_mask = torch.tensor(atom37_coord_mask)[None]

    residx_atom37_to_atom14 = batched_select(torch.tensor(residue_constants.restype_atom37_to_atom14), aatype) 
    residx_atom14_to_atom37 = batched_select(torch.tensor(residue_constants.restype_atom14_to_atom37), aatype) 

    atom14_coords = batched_select(atom37_coords, residx_atom14_to_atom37, batch_dims=2)
    atom14_coord_mask = batched_select(atom37_coord_mask, residx_atom14_to_atom37, batch_dims=2)
    atom14_atom_exists = batched_select(torch.tensor(residue_constants.restype_atom14_mask, device=aatype.device), aatype)
    atom14_coord_mask = torch.logical_and(atom14_coord_mask, atom14_atom_exists)

    return dict(
            seq = [str_seq],
            aatype = aatype,
            atom37_coords = atom37_coords,
            atom37_coord_mask = atom37_coord_mask,
            atom14_coords = atom14_coords,
            atom14_coord_mask = atom14_coord_mask,
            )

def print_batch(batch, desc = None):
    if desc is not None:
        print(desc)

    for k, v in batch.items():
        if isinstance(v, str):
            print(k, len(v))
        elif isinstance(v, torch.Tensor):
            print(k, v.shape)
    print()

def main():
    batch = create_batch_from_pdb(sys.argv[1])
    print_batch(batch, desc='create batch')

    batch.update(atom37_to_frames(batch['aatype'], batch['atom37_coords'], batch['atom37_coord_mask']))
    batch.update(
        atom37_to_torsion_angles(batch['aatype'], batch['atom37_coords'], batch['atom37_coord_mask']))
    print_batch(batch, 'with frames and torsion angles')


    print('reconstruct the positons from the backbone frame and the torsion angles')

    pred_backb_frames = r3.rigids_op(batch['rigidgroups_gt_frames'], (lambda x: x[:,:,0]))
    pred_all_frames = torsion_angles_to_frames(batch['aatype'], pred_backb_frames, batch['torsion_angles_sin_cos'])
    print('all_frames_to_global', pred_all_frames[0].shape, pred_all_frames[1].shape)

    pred_atom14_pos = frames_and_literature_positions_to_atom14_pos(batch['aatype'], pred_all_frames)
    
    gt_atom14_pos = batch['atom14_coords']
    gt_atom14_mask = batch['atom14_coord_mask']
    
    #for i in range(gt_atom14_pos.shape[1]):
    #    print(gt_atom14
    print('shape', gt_atom14_pos.shape, pred_atom14_pos.shape)
   
    for i in range(pred_atom14_pos.shape[1]):
        rmsd = torch.sqrt(torch.sum(torch.sum(torch.square(pred_atom14_pos[0,i] - gt_atom14_pos[0,i]), dim=-1) * gt_atom14_mask[0,i]) / (1e-8 + torch.sum(gt_atom14_mask[0,i])))
        print('rmsd', i, rmsd, torch.sum(gt_atom14_mask[0, i]))
        
    rmsd = torch.sqrt(torch.sum(torch.sum(torch.square(pred_atom14_pos[0] - gt_atom14_pos[0]), dim=-1) * gt_atom14_mask[0]) / (1e-8 + torch.sum(gt_atom14_mask[0])))
    print('total rmsd', i, rmsd, torch.sum(gt_atom14_mask[0, i]))
    
    return

if __name__ == '__main__':
    main()
