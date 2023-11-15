import os
import sys
import argparse
import functools
import multiprocessing as mp
import logging
import itertools
import json
import traceback
from dataclasses import dataclass, field
import pandas as pd


import numpy as np

from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB.Structure import Structure as PDBStructure

from carbonmatrix.common import residue_constants
from carbonmatrix.data.mmcif_parsing import parse as mmcif_parse, MmcifObject
from carbonmatrix.data.pdbio import make_gt_chain
from carbonmatrix.data.antibody.seq import (
    make_ab_numbering, 
    calc_epitope,
    is_in_framework,
    extract_framework_struc)
from carbonmatrix.data.antibody import antibody_constants
from carbonmatrix.data.antibody import get_reference_framework_struc

reference_framework_struc = get_reference_framework_struc()

# parse the sabdab summary file
def parse_list(path):
    with open(path) as f:
        names = [n.strip() for n in f]

    for code in names:
        yield (code,)

def make_struc(str_seq, seq2struc, structure):
    n = len(str_seq)
    assert n > 0
    coords = np.zeros((n, 14, 3), dtype=np.float32)
    coord_mask = np.zeros((n, 14), dtype=bool)

    for seq_idx, residue_at_position in seq2struc.items():
        if not residue_at_position.is_missing and residue_at_position.hetflag == ' ':
            residue_id = (residue_at_position.hetflag,
                    residue_at_position.position.residue_number,
                    residue_at_position.position.insertion_code)

            residue = structure[residue_id]

            if residue.resname not in residue_constants.restype_name_to_atom14_names:
                continue

            res_atom14_list = residue_constants.restype_name_to_atom14_names[residue.resname]

            for atom in residue.get_atoms():
                if atom.id not in res_atom14_list:
                    continue
                atom14idx = res_atom14_list.index(atom.id)
                coords[seq_idx, atom14idx] = atom.get_coord()
                coord_mask[seq_idx, atom14idx]= True

    feature = dict(seq=str_seq,
            coords=coords,
            coord_mask=coord_mask)

    return feature

def save_meta(meta, file_path):
    with open(file_path, 'w') as fw:
        json.dump(meta, fw)
    return

def save_fasta(header, str_seq, file_path):
    with open(file_path, 'w') as fw:
        fw.write('>' + header + '\n')
        fw.write(str_seq + '\n')

    return

def save_pdb(feature, file_path):
    model = PDBModel(id=0)

    heavy = feature['heavy_chain']
    chain = make_gt_chain('H', heavy['seq'], heavy['coords'], heavy['coord_mask'], heavy['numbering'])
    model.add(chain)

    if 'light_chain' in feature:
        light = feature['light_chain']
        chain = make_gt_chain('L', light['seq'], light['coords'], light['coord_mask'], light['numbering'])
        model.add(chain)

    for i, chain_feat in enumerate(feature['epitopes']):
        new_chain_id = chr(ord('M') + i)
        residue_ids = [(j, ' ') for j in chain_feat['epitope_index']]
        chain = make_gt_chain(new_chain_id, chain_feat['seq'], chain_feat['coords'], chain_feat['coord_mask'], residue_ids)
        model.add(chain)

    structure = PDBStructure(id='ab')
    structure.add(model)

    io = PDBIO(use_model_flag=1)
    io.set_structure(structure)

    io.save(file_path)

    return

def save(code, antibody_antigen_complex, meta, output_base_dir):
    heavy_chain_id = antibody_antigen_complex['heavy_chain']['chain_id']

    if 'light_chain' in antibody_antigen_complex:
        light_chain_id = antibody_antigen_complex['light_chain']['chain_id']
    else:
        light_chain_id = ''

    name = code + '_' + heavy_chain_id + '_' + light_chain_id

    npy_file = os.path.join(output_base_dir, 'npy', name + '.npy')
    np.save(npy_file, antibody_antigen_complex, allow_pickle=True)

    heavy_seq = antibody_antigen_complex['heavy_chain']['seq']
    light_seq = ''
    if 'light_chain' in antibody_antigen_complex:
        light_seq = antibody_antigen_complex['light_chain']['seq']

    fasta_file = os.path.join(output_base_dir, 'heavy_fasta', name + '.fasta')
    save_fasta(name, heavy_seq, fasta_file)

    fasta_file = os.path.join(output_base_dir, 'heavy_first_fasta', name + '.fasta')
    if light_seq:
        pair_seq = heavy_seq + ':' + light_seq
    else:
        pair_seq = heavy_seq
    save_fasta(name, pair_seq, fasta_file)

    if light_seq:
        pair_seq = light_seq + ':' + heavy_seq
    else:
        pair_seq = heavy_seq
    fasta_file = os.path.join(output_base_dir, 'light_first_fasta', name + '.fasta')
    save_fasta(name, pair_seq, fasta_file)

    cdrh3_len = np.sum(antibody_antigen_complex['heavy_chain']['region_index'] == antibody_constants.region_type_order['CDR-H3'])
    meta.update(
        code = code,
        name = name,
        heavy_seq = heavy_seq,
        light_seq = light_seq,
        cdrh3_len = int(cdrh3_len),
    )
    meta_file = os.path.join(output_base_dir, 'meta', name + '.json')
    save_meta(meta, meta_file)

    pdb_file = os.path.join(output_base_dir, 'pdb', name + '.pdb')

    save_pdb(antibody_antigen_complex, pdb_file)

    return

def make_antibody_antigen_complex(heavy_light_chain_pairs: list, mmcif_object:MmcifObject):

    def _make_chain_struc(chain_id,):
        str_seq = mmcif_object.chain_to_seqres[chain_id]
        struc = make_struc(
            str_seq,
            mmcif_object.seqres_to_structure[chain_id],
            mmcif_object.structure[chain_id],
            )

        struc.update(chain_id = chain_id)

        return struc

    def _identify_antigen(heavy_light_pair, candidate_antigen_chains):

        antibody_coords, antibody_coord_mask = heavy_light_pair['heavy_chain']['coords'], heavy_light_pair['heavy_chain']['coord_mask']

        if 'light_chain' in heavy_light_pair:
            antibody_coords = np.concatenate([antibody_coords, heavy_light_pair['light_chain']['coords']])
            antibody_coord_mask = np.concatenate([antibody_coord_mask, heavy_light_pair['light_chain']['coord_mask']])

        epitopes = []

        for candidate_chain in candidate_antigen_chains:
            epitope_index, epitope_dist = calc_epitope(
                antibody_coords, antibody_coord_mask,
                candidate_chain['coords'], candidate_chain['coord_mask'],
                dist_thres=8.0)

            if epitope_index is None:
                continue

            str_seq = candidate_chain['seq']
            epitopes.append(
                dict(
                    chain_id = candidate_chain['chain_id'],
                    seq = ''.join([str_seq[i] for i in epitope_index]),
                    coords = candidate_chain['coords'][epitope_index],
                    coord_mask = candidate_chain['coord_mask'][epitope_index],
                    epitope_index = epitope_index,
                    epitope_dist=epitope_dist))


        heavy_light_pair.update(epitopes=epitopes)

        return heavy_light_pair

    ab_chain_ids = [pair['heavy_chain']['chain_id'] for pair in heavy_light_chain_pairs]
    if 'light_chain' in heavy_light_chain_pairs:
        ab_chain_ids.extend([pair['light_chain']['chain_id'] for pair in heavy_light_chain_pairs])

    candidate_antigen_chain_ids = set([k for k in mmcif_object.chain_to_seqres]) - set(ab_chain_ids)

    candidate_antigen_chains = [_make_chain_struc(chain_id) for chain_id in candidate_antigen_chain_ids]

    ret = []
    for pair in heavy_light_chain_pairs:
        ret.append(_identify_antigen(pair, candidate_antigen_chains))

    return ret

def kabsch_numpy(X, Y):
    """ Kabsch alignment of X into Y.
        Assumes X,Y are both (Dims x N_points). See below for wrapper.
    """
    # center X and Y to the origin
    X_ = X - X.mean(axis=-1, keepdims=True)
    Y_ = Y - Y.mean(axis=-1, keepdims=True)
    # calculate convariance matrix (for each prot in the batch)
    C = np.dot(X_, Y_.transpose())
    # Optimal rotation matrix via SVD
    V, S, W = np.linalg.svd(C)
    # determinant sign for direction correction
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = np.dot(V, W)
    # calculate rotations
    X_ = np.dot(X_.T, U).T
    # return centered and aligned
    return X_, Y_

def calc_antibody_rmsd(coords1, coord_mask1, coords2, coord_mask2):
    mask = np.logical_and(coord_mask1, coord_mask2)
    coords1 = coords1[mask, :].transpose()
    coords2 = coords2[mask, :].transpose()

    aligned_coords1, aligned_coords2 = kabsch_numpy(coords1, coords2)

    rmsd = np.sqrt(np.mean(np.sum(np.square(aligned_coords1 - aligned_coords2), axis=0)))

    aligned_length = coords1.shape[1]

    return rmsd, aligned_length

def make_chain_pairs(heavy_chains, light_chains, rmsd_threshold = 5.0, aligned_length_threshold = 128):
    reference_coords, reference_coord_mask = reference_framework_struc
    pairs = []

    for heavy_chain in heavy_chains:
        light_chain_candidates = []

        for light_chain in light_chains:
            framework_coords, framework_coord_mask = extract_framework_struc(heavy_chain, light_chain)

            rmsd, aligned_length = calc_antibody_rmsd(framework_coords, framework_coord_mask, reference_coords, reference_coord_mask)

            if rmsd < rmsd_threshold and aligned_length > aligned_length_threshold:
                light_chain_candidates.append((rmsd, light_chain))

        if len(light_chain_candidates) > 0:
            if len(light_chain_candidates) == 1:
                paired_rmsd, paired_light_chain = light_chain_candidates[0]
            else:
                best_rmsd, best_idx = 100., -1
                for i, (rmsd, chain) in enumerate(light_chain_candidates):
                    if best_rmsd < rmsd:
                        best_rmsd = rmsd
                        best_idx = i
                paired_rmsd, paired_light_chain = light_chain_candidates[best_idx]

            pairs.append(dict(heavy_chain=heavy_chain, light_chain=paired_light_chain, paired_rmsd=paired_rmsd))

    return pairs

def validate_chain_ids_in_pair(pairs):
    validated = True
    heavy_chain_id_used = set()
    for pair in pairs:
        heavy_chain_id, light_chain_id = pair['heavy_chain']['chain_id'], pair['light_chain']['chain_id']
        if heavy_chain_id in heavy_chain_id_used:
            validated = False
            break
        heavy_chain_id_used.add(heavy_chain_id)

    return validated

def make_ab_chain(chain_id, str_seq, structure, seqres_to_structure, chain_type):
    assert (chain_type in ['heavy_chain', 'light_chain'])
    allow = ['H'] if chain_type == 'heavy_chain' else ['K', 'L']

    ab_def = make_ab_numbering(str_seq, allow=allow)

    if ab_def is None:
        return None

    struc = make_struc(
        str_seq,
        seqres_to_structure,
        structure,
        )

    new_struc = {k : v[ab_def['query_start']:ab_def['query_end']] for k, v in struc.items()}

    new_struc.update(
        chain_id = chain_id,
        numbering=ab_def['numbering'],
        region_index=ab_def['region_index'])

    return new_struc

def process(code, args):
    logging.info('mmcif_parse: processing %s', code)

    mmcif_file = os.path.join(args.mmcif_dir, f'{code}.cif')

    try:
        parsing_result = mmcif_parse(file_id=code, mmcif_file=mmcif_file, model_id=0)
    except PDBConstructionException as e:
        logging.warning('mmcif_parse: %s {%s}', mmcif_file, str(e))
    except Exception as e:
        logging.warning('mmcif_parse: %s {%s}', mmcif_file, str(e))
        raise Exception('...') from e

    if not parsing_result.mmcif_object:
        logging.warning('mmcif_parse: mmcif_object is empty {%s}', mmcif_file)
        return

    mmcif_object = parsing_result.mmcif_object

    # identify heavy and light chains seperately
    heavy_chains, light_chains = [], []
    for chain_id, str_seq in mmcif_object.chain_to_seqres.items():
        heavy_chain = make_ab_chain(
            chain_id, str_seq,
            mmcif_object.structure[chain_id], mmcif_object.seqres_to_structure[chain_id],
            chain_type='heavy_chain')

        if heavy_chain is not None:
            heavy_chains.append(heavy_chain)

        light_chain = make_ab_chain(
            chain_id, str_seq,
            mmcif_object.structure[chain_id], mmcif_object.seqres_to_structure[chain_id],
            chain_type='light_chain')

        if light_chain is not None:
            light_chains.append(light_chain)

    pairs = []
    # pair the heavy and light chains
    if len(heavy_chains) == 0:
        logging.warn(f'no heavy chains found in {code}')
        return pairs
    elif len(light_chains) > 0:
        pairs = make_chain_pairs(heavy_chains, light_chains)
        # validate the heavy and light chain ids

        if len(pairs) == 0:
            logging.warn(f'No heavy and light chain pairs found in {code}!')
            pairs = [dict(heavy_chain=h) for h in heavy_chains]
        else:
            if not validate_chain_ids_in_pair(pairs):
                pair_ids = ':'.join([p['heavy_chain']['chain_id'] + ',' + p['light_chain']['chain_id'] for p in pairs])
                logging.warn(f'two heavy chains share the same light chain in {code}. {pair_ids}')
                return
            logging.info(f'{len(pairs)} heavy and light chain pairs found in {code}')

    # extract heavy only chains
    heavy_in_pair_chain_ids = [x['heavy_chain']['chain_id'] for x in pairs]
    heavy_only_chains = [dict(heavy_chain=x) for x in heavy_chains if x ['chain_id'] not in heavy_in_pair_chain_ids]
    pairs.extend(heavy_only_chains)
    logging.info(f'{len(heavy_only_chains)} heavy only chains found in {code}')

    # identify antigens
    antibody_antigen_complexs = make_antibody_antigen_complex(pairs, mmcif_object)

    for antibody_antigen_complex in antibody_antigen_complexs:
        save(code, antibody_antigen_complex, meta=mmcif_object.header, output_base_dir=args.output_base_dir)

    return

def main(args):
    # output dirs
    if not os.path.exists(os.path.join(args.output_base_dir, 'npy')):
        os.mkdir(os.path.join(args.output_base_dir, 'npy'))
    if not os.path.exists(os.path.join(args.output_base_dir, 'heavy_fasta')):
        os.mkdir(os.path.join(args.output_base_dir, 'heavy_fasta'))
    if not os.path.exists(os.path.join(args.output_base_dir, 'heavy_first_fasta')):
        os.mkdir(os.path.join(args.output_base_dir, 'heavy_first_fasta'))
    if not os.path.exists(os.path.join(args.output_base_dir, 'light_first_fasta')):
        os.mkdir(os.path.join(args.output_base_dir, 'light_first_fasta'))
    if not os.path.exists(os.path.join(args.output_base_dir, 'pdb')):
        os.mkdir(os.path.join(args.output_base_dir, 'pdb'))
    if not os.path.exists(os.path.join(args.output_base_dir, 'meta')):
        os.mkdir(os.path.join(args.output_base_dir, 'meta'))

    func = functools.partial(process, args=args)

    with mp.Pool(args.cpus) as p:
        p.starmap(func, parse_list(args.name_idx))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--cpus', type=int, default=1)
  parser.add_argument('--name_idx', type=str, required=True)
  parser.add_argument('--mmcif_dir', type=str, required=True)
  parser.add_argument('--output_base_dir', type=str, required=True)
  parser.add_argument('--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args)
