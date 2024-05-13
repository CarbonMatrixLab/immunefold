import os
import argparse
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd

# from carbonmatrix.common.ab.metrics import calc_ab_metrics
# from carbonmatrix.common import residue_constants
import pdb
from Bio.PDB.Chain import Chain
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB.PDBIO import PDBIO

from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1
# from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}

restype_3to1 = {v: k for k, v in restype_1to3.items()}

restype_name_to_atom14_names = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    'GLY': ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'UNK': ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],

}

def make_chain(aatypes, coords, chain_id, plddt):
    chain = Chain(chain_id)

    serial_number = 1

    def make_residue(i, aatype, coord, bfactor):
        nonlocal serial_number

        resname = restype_1to3.get(aatype, 'UNK')
        residue = Residue(id=(' ', i, ' '), resname=resname, segid='')
        for j, atom_name in enumerate(restype_name_to_atom14_names[resname]):
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
        bfactor = plddt[i]
        chain.add(make_residue(i + 1, aa, coord, bfactor))

    return chain

# def make_gt_chain(chain_id, aatypes, coords, coord_mask, residue_ids=None):
#     chain = Chain(chain_id)

#     serial_number = 1

#     def make_residue(resid, aatype, coord, mask, bfactor):
#         nonlocal serial_number

#         resnum, inscode = resid

#         resname = restype_1to3.get(aatype, 'UNK')
#         residue = Residue(id=(' ', resnum, inscode), resname=resname, segid='')
#         for j, atom_name in enumerate(restype_name_to_atom14_names[resname]):
#             if atom_name == '':
#                 continue
#             if not mask[j]:
#                 continue
#             atom = Atom(name=atom_name,
#                     coord=coord[j],
#                     bfactor=bfactor, occupancy=1, altloc=' ',
#                     fullname=str(f'{atom_name:<4s}'),
#                     serial_number=serial_number, element=atom_name[:1])
#             residue.add(atom)

#             serial_number += 1

#         return residue

#     N = len(aatypes)

#     if residue_ids is None:
#         residue_ids = [(i + 1, ' ') for i in range(N)]

#     for i in range(N):
#         bfactor = 0
#         if np.sum(coord_mask[i]) > 0:
#             chain.add(make_residue(residue_ids[i], aatypes[i], coords[i], coord_mask[i], bfactor))

#     return chain

# def save_ig_pdb(str_heavy_seq, str_light_seq, coord, pdb_path):
#     assert len(str_heavy_seq) + len(str_light_seq) == coord.shape[0]

#     heavy_chain = make_chain(str_heavy_seq, coord[:len(str_heavy_seq)], 'H')
#     light_chain = make_chain(str_light_seq, coord[len(str_heavy_seq):], 'L')

#     model = PDBModel(id=0)
#     model.add(heavy_chain)
#     model.add(light_chain)

#     pdb = PDBIO()
#     pdb.set_structure(model)
#     pdb.save(pdb_path)

def save_pdb(str_seq_list, coord_list, pdb_path, plddt_list):
    model = PDBModel(id=0)

    if isinstance(str_seq_list, str):
        str_seq_list = [str_seq_list]

    chain_ids = list(coord_list.keys())
    pdb.set_trace()


    for i in range(len(chain_ids)):
        chain_id = chain_ids[i]
        str_seq = str_seq_list[chain_id]
        coord = coord_list[chain_id]
        plddt = plddt_list[chain_id]
        chain = make_chain(str_seq, coord, chain_id, plddt)

        model.add(chain)

    pdb_ = PDBIO()
    pdb_.set_structure(model)
    pdb_.save(pdb_path)

    return


# from https://github.com/charnley/rmsd/blob/master/rmsd/calculate_rmsd.py
def kabsch_rotation(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U



# have been validated with kabsch from RefineGNN
def kabsch(a, b):
    # find optimal rotation matrix to transform a into b
    # a, b are both [N, 3]
    # a_aligned = aR + t
    a, b = np.array(a), np.array(b)
    a_mean = np.mean(a, axis=0)
    b_mean = np.mean(b, axis=0)
    a_c = a - a_mean
    b_c = b - b_mean

    rotation = kabsch_rotation(a_c, b_c)
    # a_aligned = np.dot(a_c, rotation)
    # t = b_mean - np.mean(a_aligned, axis=0)
    # a_aligned += t
    t = b_mean - np.dot(a_mean, rotation)
    a_aligned = np.dot(a, rotation) + t

    return a_aligned, rotation, t
    
def make_coords(pdb_file):
    parser = PDBParser(QUIET=1)
    model = parser.get_structure('pdb', pdb_file)[0]
    name = pdb_file.split('/')[-1].split('.')[0]

    residues = dict()
    single_codes = dict()
    chain_ids = []
    for chain in model:
        chain_ids.append(chain.id)
    
    for chain_id in chain_ids:
        residue = list(model[chain_id].get_residues())
        residues.update(
            {chain_id: residue})
        single_code = ([seq1(r.get_resname()) for r in residue if r.get_resname() in restype_3to1.keys()])
        single_codes.update(
            {chain_id: single_code}
        )
    str_seq_list = [''.join(sc) for sc in single_codes]
    str_seq_list = dict()
    for i in range(len(single_codes)):
        str_seq_list.update(
            {chain_ids[i]: ''.join(single_codes[chain_ids[i]])}
        )
    coords_list = dict()
    coord_mask_list = dict()
    bfactor_list = dict()
    for i in range(len(residues)):
        chain_id = chain_ids[i]
        r = residues[chain_id]
        coords = np.zeros((len(r), 14, 3))
        coord_mask = np.zeros((len(r), 14), dtype=bool)
        bfactor = np.zeros((len(r)))
        # pdb.set_trace()            
        for j, residue in enumerate(r):
            # pdb.set_trace()
            res_atom14_list = restype_name_to_atom14_names[residue.resname]
            
            for atom in residue.get_atoms():
                if atom.id not in res_atom14_list:
                    continue
                atom14idx = res_atom14_list.index(atom.id)
                coords[j, atom14idx] = atom.get_coord()
                coord_mask[j, atom14idx]= True
                bfactor[j] = atom.get_bfactor()
            # pdb.set_trace()
        coords_list.update(
            {chain_ids[i]: coords}
            )
        coord_mask_list.update(
            {chain_ids[i]: coord_mask}
            )
        bfactor_list.update(
            {chain_ids[i]: bfactor}
            )

    return coords_list, coord_mask_list, str_seq_list, bfactor_list


def replace(source_coords, target_coords, source_bfactor, target_bfactor, source_str_seq, target_str_seq):
    if len(source_str_seq) == len(target_str_seq):
        domain_start, domain_end = 0, len(target_str_seq)
    elif len(source_str_seq) < len(target_str_seq):
        domain_start = target_str_seq.find(source_str_seq)
        assert domain_start != -1
        assert domain_start + len(source_str_seq) <= len(target_str_seq)
        domain_end = domain_start + len(source_str_seq)
    else:
        raise ValueError('source_str_seq should be shorter than target_str_seq')
    # pdb.set_trace()
    sc_coord = source_coords
    tc_coord = target_coords[domain_start:domain_end]
    sc_ca = source_coords[:, 1]
    tc_ca = tc_coord[:, 1]
    sc_ca_aligned, rotation, t = kabsch(sc_ca, tc_ca)
    sc_aligned = np.dot(sc_coord, rotation) + t
    target_coords[domain_start:domain_end] = sc_aligned
    target_bfactor[domain_start:domain_end] = source_bfactor
    return target_coords, target_bfactor
    
    

        


def merge(source_pdb, target_pdb, source_chains, target_chains):
    source_coords_list, source_coord_mask_list, source_str_seq_list, source_bfactor_list = make_coords(source_pdb)
    target_coords_list, target_coord_mask_list, target_str_seq_list, target_bfactor_list = make_coords(target_pdb)
    assert len(source_chains) == len(target_chains)
    for i in range(len(source_chains)):
        source_chain = source_chains[i]
        target_chain = target_chains[i]

        source_coords = source_coords_list[source_chain]
        target_coords = target_coords_list[target_chain]

        source_str_seq = source_str_seq_list[source_chain]
        target_str_seq = target_str_seq_list[target_chain]

        source_bfactor = source_bfactor_list[source_chain]
        target_bfactor = target_bfactor_list[target_chain]

        target_coords, target_bfactor = replace(source_coords, target_coords, source_bfactor, target_bfactor, source_str_seq, target_str_seq)

        target_coords_list[target_chain] = target_coords
        target_bfactor_list[target_chain] = target_bfactor
    return target_coords_list, target_coord_mask_list, target_str_seq_list, target_bfactor_list

    
def main(args):
    source_pdb = args.source_pdb
    target_pdb = args.target_pdb

    source_chains = args.source_chains
    source_chains = source_chains.split(',')
    target_chains = args.target_chains
    target_chains = target_chains.split(',')
    output_pdb = args.output_pdb

    target_coords_list, target_coord_mask_list, target_str_seq_list, target_bfactor_list = merge(source_pdb, target_pdb, source_chains, target_chains)
    save_pdb(target_str_seq_list, target_coords_list, output_pdb, target_bfactor_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge two pdb files')
    parser.add_argument('-s', '--source_pdb', type=str, help='source pdb file')
    parser.add_argument('-t', '--target_pdb', type=str, help='target pdb file')
    parser.add_argument('-sc', '--source_chains', type=str, help='source chains')
    parser.add_argument('-tc', '--target_chains', type=str, help='target chains')
    parser.add_argument('-o', '--output_pdb', type=str, help='output pdb file')
    args = parser.parse_args()
    main(args)


