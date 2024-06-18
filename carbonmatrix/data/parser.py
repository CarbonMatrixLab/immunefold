from os.path import basename, splitext
from collections import OrderedDict
import numpy as np

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Chain import Chain as PDBChain
from Bio.PDB.Residue import Residue
from Bio.PDB.vectors import Vector as Vector, calc_dihedral
import random
from carbonmatrix.common import residue_constants
import pdb

def extract_chain_subset(orig_chain, res_ids):
    chain = PDBChain(orig_chain.id)
    for residue in orig_chain:
        if residue.id in res_ids:
            residue.detach_parent()
            chain.add(residue)
    return chain

def parse_pdb(pdb_file, model=0):
    parser = PDBParser()
    structure = parser.get_structure(get_pdb_id(pdb_file), pdb_file)
    return structure[model]

def make_stage1_feature(structure):
    N = len(structure)

    coords = np.zeros((N, 14, 3), dtype=np.float32)
    coord_mask = np.zeros((N, 14), dtype=bool)
    str_seq = []

    for seq_idx, residue in enumerate(structure):
        aa = residue_constants.restype_3to1.get(residue.resname, 'X')
        str_seq.append(aa)
        res_atom14_list = residue_constants.restype_name_to_atom14_names.get(residue.resname,
                residue_constants.restype_name_to_atom14_names['UNK'])

        for atom in residue.get_atoms():
            #if atom.id not in ['CA', 'C', 'N', 'O']:
            #    continue
            atom14idx = res_atom14_list.index(atom.id)
            coords[seq_idx, atom14idx] = atom.get_coord()
            coord_mask[seq_idx, atom14idx]= True

    feature = dict(
            str_seq=''.join(str_seq),
            residx = np.arange(N),
            coords=coords,
            coord_mask=coord_mask)

    return feature

def make_stage1_feature_from_pdb(pdb_file):
    struc = parse_pdb(pdb_file)
    N = len(list(struc.get_chains()))
    assert (N in [1, 2])

    if N == 1:
        return make_stage1_feature(struc['A'])

    A = make_stage1_feature(struc['A'])
    B = make_stage1_feature(struc['B'])

    return dict(
            str_seq= A['str_seq'] + B['str_seq'],
            residx = np.concatenate([A['residx'], B['residx'] + A['residx'].shape[0] + residue_constants.residue_chain_index_offset], axis=0),
            coords=np.concatenate([A['coords'], B['coords']], axis=0),
            coord_mask=np.concatenate([A['coord_mask'], B['coord_mask']], axis=0))

def make_chain_feature(chain):
    N = len(structure)

    coords = np.zeros((N, 14, 3), dtype=np.float32)
    coord_mask = np.zeros((N, 14), dtype=bool)
    str_seq = []

    for seq_idx, residue in enumerate(structure):
        aa = residue_constants.restype_3to1.get(residue.resname, 'X')
        str_seq.append(aa)
        res_atom14_list = residue_constants.restype_name_to_atom14_names.get(residue.resname,
                residue_constants.restype_name_to_atom14_names['UNK'])

        for atom in residue.get_atoms():
            atom14idx = res_atom14_list.index(atom.id)
            coords[seq_idx, atom14idx] = atom.get_coord()
            coord_mask[seq_idx, atom14idx]= True

    feature = dict(
            str_seq=''.join(str_seq),
            coords=coords,
            coord_mask=coord_mask)

    return feature

def make_feature_from_pdb(pdb_file):
    struc = parse_pdb(pdb_file)

    chain_num = len(list(struc.get_chains()))
    assert (chain_num == 1)

    feat = make_chain_feature(struc.get_chains[0])

    return feat

def make_feature_from_npz(npz_file, is_ig_feature=False, shuffle_multimer_seq=False):
    x = np.load(npz_file)

    if not is_ig_feature:
        str_seq = str(x['seq']) if 'seq' in x else str(x['str_seq'])

        coords = x['coords']
        coord_mask = x['coord_mask']
    else:
        if 'heavy_str_seq' in x:
            str_seq = [str(x['heavy_str_seq'])]
            coords = [x['heavy_coords']]
            coord_mask = [x['heavy_coord_mask']]
            chain_id = [x['heavy_chain_id']]


            if 'light_str_seq' in x:
                light_str_seq = str(x['light_str_seq'])
                light_coords = x['light_coords']
                light_coord_mask = x['light_coord_mask']
                light_chain_id = x['light_chain_id']
                str_seq.append(light_str_seq)
                coords.append(light_coords)
                coord_mask.append(light_coord_mask)
                chain_id.append(light_chain_id)
            if 'antigen_str_seq' in x:
                antigen_str_seq = str(x['antigen_str_seq'])
                antigen_coords = x['antigen_coords']
                antigen_coord_mask = x['antigen_coord_mask']
                antigen_chain_id = x['antigen_chain_id']
                str_seq.append(antigen_str_seq)
                coords.append(antigen_coords)
                coord_mask.append(antigen_coord_mask)
                chain_id.append(antigen_chain_id)

            if shuffle_multimer_seq:
                chain = np.arange(len(str_seq))
                rand = random.randint(1, len(str_seq))
                random.shuffle(chain)
                chain = chain[:rand].tolist()
                chain.append(0)
                chain = set(chain)
                str_seq = [str_seq[i] for i in chain]
                coords = [coords[i] for i in chain]
                coord_mask = [coord_mask[i] for i in chain]
                chain_id = [chain_id[i] for i in chain]
                
            str_seq = ':'.join(str_seq)
            coords = np.concatenate(coords, axis=0)
            coord_mask = np.concatenate(coord_mask, axis=0)
            chain_id = np.concatenate(chain_id, axis=0)
            
        elif 'beta_str_seq' in x:
            str_seq = [str(x['beta_str_seq'])]
            coords = [x['beta_coords']]
            coord_mask = [x['beta_coord_mask']]
            chain_id = [x['beta_chain_id']]
            if 'alpha_str_seq' in x:
                alpha_str_seq = str(x['alpha_str_seq'])
                alpha_coords = x['alpha_coords']
                alpha_coord_mask = x['alpha_coord_mask']
                alpha_chain_id = x['alpha_chain_id']
                str_seq.append(alpha_str_seq)
                coords.append(alpha_coords)
                coord_mask.append(alpha_coord_mask)
                chain_id.append(alpha_chain_id)
            if 'antigen_str_seq' in x:
                antigen_str_seq = str(x['antigen_str_seq'])
                antigen_coords = x['antigen_coords']
                antigen_coord_mask = x['antigen_coord_mask']
                antigen_chain_id = x['antigen_chain_id']
                str_seq.append(antigen_str_seq)
                coords.append(antigen_coords)
                coord_mask.append(antigen_coord_mask)
                chain_id.append(antigen_chain_id)
            if 'mhc_str_seq' in x:
                mhc_str_seq = str(x['mhc_str_seq'])
                mhc_coords = x['mhc_coords']
                mhc_coord_mask = x['mhc_coord_mask']
                mhc_chain_id = x['mhc_chain_id']
                str_seq.append(mhc_str_seq)
                coords.append(mhc_coords)
                coord_mask.append(mhc_coord_mask)
                chain_id.append(mhc_chain_id)

            if shuffle_multimer_seq:
                chain = np.arange(len(str_seq))
                rand = random.randint(1, len(str_seq))
                random.shuffle(chain)
                chain = chain[:rand].tolist()
                chain.append(0)
                chain = set(chain)
                str_seq = [str_seq[i] for i in chain]
                coords = [coords[i] for i in chain]
                coord_mask = [coord_mask[i] for i in chain]
                chain_id = [chain_id[i] for i in chain]
            # pdb.set_trace()
            str_seq = ':'.join(str_seq)
            coords = np.concatenate(coords, axis=0)
            coord_mask = np.concatenate(coord_mask, axis=0)
            chain_id = np.concatenate(chain_id, axis=0)

    results = dict(
            str_seq = str_seq,
            coords = coords,
            coord_mask = coord_mask,
            chain_id = chain_id)
    if 'antigen_contact_idx' in x:
        results.update(
            {'antigen_contact_idx': x['antigen_contact_idx']}
        )

    return results
