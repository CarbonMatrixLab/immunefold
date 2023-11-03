import logging
import os

import torch
from torch.nn import functional as F
import numpy as np

from Bio.PDB.Chain import Chain
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB.PDBIO import PDBIO

from carbonmatrix.common import residue_constants

def make_chain(aatypes, coords, chain_id, plddt=None):
    chain = Chain(chain_id)

    serial_number = 1

    def make_residue(i, aatype, coord, bfactor):
        nonlocal serial_number

        resname = residue_constants.restype_1to3.get(aatype, 'UNK')
        residue = Residue(id=(' ', i, ' '), resname=resname, segid='')
        for j, atom_name in enumerate(residue_constants.restype_name_to_atom14_names[resname]):
            if atom_name == '':
                continue

            atom = Atom(name=atom_name,
                    coord=coord[j],
                    bfactor=bfactor, occupancy=1, altloc=' ',
                    fullname=str(f'{atom_name:<4s}'),
                    serial_number=serial_number, element=atom_name[:1])
            residue.add(atom)

            serial_number += 1

        return residue

    for i, (aa, coord) in enumerate(zip(aatypes, coords)):
        bfactor = 0 if plddt is None else plddt[i]
        chain.add(make_residue(i + 1, aa, coord, bfactor))

    return chain

def make_gt_chain(chain_id, aatypes, coords, coord_mask, residue_ids=None):
    chain = Chain(chain_id)

    serial_number = 1

    def make_residue(resid, aatype, coord, mask, bfactor):
        nonlocal serial_number

        resnum, inscode = resid

        resname = residue_constants.restype_1to3.get(aatype, 'UNK')
        residue = Residue(id=(' ', resnum, inscode), resname=resname, segid='')
        for j, atom_name in enumerate(residue_constants.restype_name_to_atom14_names[resname]):
            if atom_name == '':
                continue
            if not mask[j]:
                continue
            atom = Atom(name=atom_name,
                    coord=coord[j],
                    bfactor=bfactor, occupancy=1, altloc=' ',
                    fullname=str(f'{atom_name:<4s}'),
                    serial_number=serial_number, element=atom_name[:1])
            residue.add(atom)

            serial_number += 1

        return residue

    N = len(aatypes)

    if residue_ids is None:
        residue_ids = [(i + 1, ' ') for i in range(N)]

    for i in range(N):
        bfactor = 0
        if np.sum(coord_mask[i]) > 0:
            chain.add(make_residue(residue_ids[i], aatypes[i], coords[i], coord_mask[i], bfactor))

    return chain

def save_ig_pdb(str_heavy_seq, str_light_seq, coord, pdb_path):
    assert len(str_heavy_seq) + len(str_light_seq) == coord.shape[0]

    heavy_chain = make_chain(str_heavy_seq, coord[:len(str_heavy_seq)], 'H')
    light_chain = make_chain(str_light_seq, coord[len(str_heavy_seq):], 'L')

    model = PDBModel(id=0)
    model.add(heavy_chain)
    model.add(light_chain)

    pdb = PDBIO()
    pdb.set_structure(model)
    pdb.save(pdb_path)

def save_pdb(multimer_str_seq, coord, pdb_path, chain_ids = None, plddt=None):
    model = PDBModel(id=0)

    if isinstance(multimer_str_seq, str):
        multimer_str_seq = [multimer_str_seq]

    assert(len(''.join(multimer_str_seq)) == coord.shape[0])

    if chain_ids is None:
        chain_ids = [chr(ord('A') + i) for i in range(len(multimer_str_seq))]

    assert(len(multimer_str_seq) == len(chain_ids))

    start_pos = 0
    for str_seq, chain_id in zip(multimer_str_seq, chain_ids):
        end_pos = start_pos + len(str_seq)
        chain_plddt = None if plddt is None else plddt[start_pos:end_pos]
        chain = make_chain(str_seq, coord[start_pos:end_pos], chain_id, chain_plddt)
        start_pos = end_pos

        model.add(chain)

    pdb = PDBIO()
    pdb.set_structure(model)
    pdb.save(pdb_path)

    return
