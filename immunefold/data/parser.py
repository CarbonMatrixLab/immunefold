from os.path import basename, splitext
from collections import OrderedDict
import numpy as np

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Chain import Chain as PDBChain
from Bio.PDB.Residue import Residue
from Bio.PDB.vectors import Vector as Vector, calc_dihedral
import random
from immunefold.common import residue_constants
import pdb
from Bio.SeqUtils import seq1


def extract_chain_subset(orig_chain, res_ids):
    chain = PDBChain(orig_chain.id)
    for residue in orig_chain:
        if residue.id in res_ids:
            residue.detach_parent()
            chain.add(residue)
    return chain

def parse_pdb(pdb_file, model=0):
    parser = PDBParser()
    structure = parser.get_structure('model', pdb_file)
    return structure[model]


def make_chain_feature(chain, idx):
    
    residues = list(chain.get_residues())
    residues = [r for r in residues if r.get_resname() in residue_constants.restype_3to1.keys()]
    str_seq = [seq1(r.get_resname()) for r in residues if r.get_resname() in residue_constants.restype_3to1.keys()]
    N = len(str_seq)
    
    coords = np.zeros((N, 14, 3), dtype=np.float32)
    coord_mask = np.zeros((N, 14), dtype=bool)
    # str_seq = []
    # for seq_idx, r in enumerate(residue):
    for jj, residue in enumerate(residues):
            # pdb.set_trace()
        if residue.get_resname() in residue_constants.restype_3to1.keys():
            res_atom14_list = residue_constants.restype_name_to_atom14_names[residue.resname]
            
            for atom in residue.get_atoms():
                if atom.id not in res_atom14_list:
                    continue
                # try:
                atom14idx = res_atom14_list.index(atom.id)
                coords[jj, atom14idx] = atom.get_coord()
                coord_mask[jj, atom14idx]= True

    chain_id = 4 * np.ones((N,), dtype=np.int32)
    gt_mask = np.ones((N,), dtype=np.bool_)
    if idx == None:
        min_idx = 0
        max_idx = N
    else:
        min_idx = max(0, idx-64)
        max_idx = min(idx+64, N+1)
    feature = dict(
            str_seq=(''.join(str_seq))[min_idx: max_idx],
            coords=coords[min_idx: max_idx],
            coord_mask=coord_mask[min_idx: max_idx],
            chain_id = chain_id[min_idx: max_idx],
            gt_mask = gt_mask[min_idx: max_idx])
    return feature

def make_feature_from_pdb(seq, ag_pdb, contact_idx):
    struc = parse_pdb(ag_pdb)

    chain_id = list((struc.get_chains()))
    # pdb.set_trace()
    # chain_num = len(list(struc.get_chains()))
    assert (len(chain_id) == 1)

    feat = make_chain_feature(struc[chain_id[0].id], contact_idx)
    ab_feat = make_ab_feature(seq)
    str_seq = seq + ':' + feat['str_seq']
    coords = np.concatenate([ab_feat['coords'], feat['coords']], axis=0)
    coord_mask = np.concatenate([ab_feat['coord_mask'], feat['coord_mask']], axis=0)
    chain_id = np.concatenate([ab_feat['chain_id'], feat['chain_id']], axis=0)
    gt_mask = np.concatenate([ab_feat['gt_mask'], feat['gt_mask']], axis=0)
    ab_ag_feat = dict(
            str_seq = str_seq,
            coords = coords,
            coord_mask = coord_mask,
            chain_id = chain_id,
            gt_mask = gt_mask)
    return ab_ag_feat

def make_ab_feature(seq):
    if ':' in seq:
        heavy_seq, light_seq = seq.split(':')  
        N = len(heavy_seq) + len(light_seq)
        heavy_chain_id = np.zeros((len(heavy_seq)), dtype=int)
        light_chain_id = np.ones((len(light_seq)), dtype=int)
        chain_id = np.concatenate([heavy_chain_id, light_chain_id], axis=0)
        str_seq = [heavy_seq, light_seq]
    else:
        heavy_seq = seq
        str_seq = [heavy_seq]
        N = len(heavy_seq)
        chain_id = np.zeros(N, dtype=int)

    
    coords = np.zeros((N, 14, 3), dtype=np.float32)
    coord_mask = np.zeros((N, 14), dtype=bool)
    # chain_id = np.zeros(N, dtype=int)
    gt_mask = np.zeros(N, dtype=bool)

    return dict(
            str_seq=''.join(str_seq),
            coords=coords,
            coord_mask=coord_mask,
            chain_id=chain_id,
            gt_mask = gt_mask)