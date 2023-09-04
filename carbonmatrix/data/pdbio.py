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

def make_chain(aa_types, coords, chain_id):
    chain = Chain(chain_id)

    serial_number = 1

    def make_residue(i, aatype, coord):
        nonlocal serial_number
        
        resname = residue_constants.restype_1to3.get(aatype, 'UNK')
        residue = Residue(id=(' ', i, ' '), resname=resname, segid='')
        for j, atom_name in enumerate(residue_constants.restype_name_to_atom14_names[resname]):
            if atom_name == '':
                continue
            
            atom = Atom(name=atom_name, 
                    coord=coord[j],
                    bfactor=0, occupancy=1, altloc=' ',
                    fullname=str(f'{atom_name:<4s}'),
                    serial_number=serial_number, element=atom_name[:1])
            residue.add(atom)

            serial_number += 1

        return residue

    for i, (aa, coord) in enumerate(zip(aa_types, coords)):
        chain.add(make_residue(i + 1, aa, coord))

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

def save_general_pdb(str_seq, coord, pdb_path, coord_mask=None):
    assert len(str_seq) == coord.shape[0]

    chain = make_chain(str_seq, coord[:len(str_seq)], 'A')

    model = PDBModel(id=0)
    model.add(chain)
    
    pdb = PDBIO()
    pdb.set_structure(model)
    pdb.save(pdb_path)
