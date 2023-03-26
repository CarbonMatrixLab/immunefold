import sys
from abfold.model.utils import batched_select2
from abfold.trainer import residue_constants

from abfold.trainer import residue_constants
from abfold.trainer.geometry import get_atom_from_pdb, atom37_to_frames,atom37_to_torsion_angles, torsion_angles_to_frames, frames_and_literature_positions_to_atom14_pos, atom37_to_atom14 
from abfold.utils import Kabsch
from einops import rearrange
import torch

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Chain import Chain as PDBChain


def get_atom_from_pdb(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('id', pdb_file)
    residues = [x for x in structure.get_residues()]
    num_residue = len(residues)

    coords = np.zeros((num_residue, 37, 3), dtype=np.float32)
    coord_mask = np.zeros((num_residue, 37), dtype=np.bool)
    seq = []

    for i, r in enumerate(residues):
        seq.append(residue_constants.restype_3to1.get(
            r.resname, residue_constants.restypes_with_x[-1]))

        for atom in r.get_atoms():
            atom_idx = residue_constants.atom_order[atom.id]
            coords[i, atom_idx] = atom.get_coord()
            coord_mask[i, atom_idx]= True
    
    aatype = residue_constants.sequence_to_index(''.join(seq), mapping=residue_constants.restype_order_with_x, map_unknown_to_x=True)
    
    aatype=torch.tensor(aatype)[None]
    coords=torch.tensor(coords)[None] 
    coord_mask=torch.tensor(coord_mask)[None]
    print(residue_constants.restype_atom37_to_atom14,'37-14')
    residx_atom14_to_atom37 = batched_select(torch.tensor(residue_constants.restype_atom14_to_atom37), aatype) 
    residx_atom37_to_atom14 = batched_select(torch.tensor(residue_constants.restype_atom37_to_atom14), aatype) 

    return dict(seq = [seq], aatype=aatype, coords=coords, coord_mask=coord_mask,
            residx_atom14_to_atom37=residx_atom14_to_atom37,
            residx_atom37_to_atom14=residx_atom37_to_atom14)


def main():
    batch = get_atom_from_pdb(sys.argv[1])
    batch['atom14_coords'] = atom37_to_atom14(batch['coords'], batch)
    batch['atom14_coord_mask'] = atom37_to_atom14(batch['coord_mask'], batch)
    
    for k, v in batch.items():
        try:
            print(k, v.shape)
        except:
            pass
    batch.update(
            atom37_to_frames(batch['aatype'], batch['coords'], batch['coord_mask']))
   
    batch.update(
        atom37_to_torsion_angles(batch['aatype'], batch['coords'], batch['coord_mask']))
    
    for k, v in batch.items():
        if k == 'seq':
            continue
        try:
            print(k, v.shape)
        except:
            print(k, v[0].shape, v[1].shape)

    print('torsion_angles_to_frames')
    bb_frames = tuple(map(lambda x: x[:,:,0], batch['rigidgroups_gt_frames']))
    all_frames_to_global = torsion_angles_to_frames(batch['aatype'], bb_frames, batch['torsion_angles_sin_cos'])
    print('all_frames_to_global', all_frames_to_global[0].shape, all_frames_to_global[1].shape)
    print('frames_and_literature_positions_to_atom14_pos')

    pred_pos = frames_and_literature_positions_to_atom14_pos(batch['aatype'], all_frames_to_global)
     
    mask = batched_select2(torch.tensor(residue_constants.restype_atom14_mask), batch['aatype'])
    coord_mask = torch.logical_and(mask, batch['atom14_coord_mask'])
    labels = batch['atom14_coords']
    pred = pred_pos

    rmsd = torch.sqrt(torch.sum(torch.sum(torch.square(labels-pred), dim=-1) * coord_mask) / torch.sum(coord_mask))
    print('rmsd', rmsd, torch.sum(coord_mask), coord_mask.shape, torch.sum(coord_mask)/ coord_mask.shape[1])
    
    flat_cloud_mask = rearrange(coord_mask, 'b l c -> b (l c)')
    coords_aligned, labels_aligned = Kabsch(
            rearrange(rearrange(pred, 'b l c d -> b (l c) d')[flat_cloud_mask], 'c d -> d c'),
            rearrange(rearrange(labels, 'b l c d -> b (l c) d')[flat_cloud_mask], 'c d -> d c'))
    
    rmsd = torch.sqrt(torch.mean(torch.sum(torch.square(coords_aligned-labels_aligned), dim=0)))
    print(rmsd)

if __name__ == '__main__':
    main()
