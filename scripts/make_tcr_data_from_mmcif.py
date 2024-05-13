import os
import argparse
import functools
import multiprocessing as mp
import logging
import itertools
import json
import pandas as pd
import traceback
import sys
sys.path.append('..')
import numpy as np
import pdb
from Bio.PDB.PDBExceptions import PDBConstructionException

from carbonmatrix.common import residue_constants
from carbonmatrix.data.mmcif_parsing import parse as mmcif_parse
from carbonmatrix.common.ab.numbering import renumber_ab_seq, get_ab_regions, get_tcr_regions

def parse_list(path):
    with open(path) as f:
        names = [n.strip() for n in f]
    
    df = pd.read_csv(path, sep='\t')
    df = df[df['method'].isin(['X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY'])]
    
    logging.info(f'all pairs: {df.shape[0]}')
    
    df = df.fillna({'Bchain':'', 'Achain':''})
    df = df[df['Bchain'] != '']
    df = df[df['TCRtype'] == 'abTCR']

    logging.info(f'number of B chains: {df.shape[0]}')
    df.loc[df['antigen_type'] != 'peptide', 'antigen_chain'] = ''
    df.loc[df['mhc_type'] != 'MH1', 'mhc_chain1'] = ''
    # df = df[df['model'] == 0]
    
    # logging.info(f'number of model 0: {df.shape[0]}')
   
   
    for code, rr in df.groupby(by='pdb'):            
        chain_list = list(zip(rr['Bchain'], rr['Achain'], rr['antigen_chain'], rr['mhc_chain1']))       
        yield (code, chain_list)

def make_feature(str_seq, seq2struc, structure, chain):
    chain_name = {
        'B': 1,
        'A': 2,
        'G': 3,
        'M': 4
    }
    prefix_name = {
        'B': 'beta',
        'A': 'alpha',
        'G': 'antigen',
        'M': 'mhc'
    }
    n = len(str_seq)
    assert n > 0
    coords = np.zeros((n, 14, 3), dtype=np.float32)
    coord_mask = np.zeros((n, 14), dtype=bool)
    chain_id = np.ones((n,), dtype=np.float32) * chain_name[chain]
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
    
    feature = dict(str_seq=str_seq,
            coords=coords,
            coord_mask=coord_mask,
            chain_id=chain_id)
    prefix = prefix_name[chain]

    feature = {f'{prefix}_{k}': v for k, v in feature.items()}

    return feature

def make_npz(beta_data, alpha_data, antigen_data, mhc_data):
    def _make_domain(feature, chain_id):
        allow = ['B'] if chain_id == 'B' else ['A']
        prefix = 'beta' if chain_id == 'B' else 'alpha'
        anarci_res = renumber_ab_seq(feature[f'{prefix}_str_seq'], allow=allow, scheme='imgt')
        domain_numbering, domain_start, domain_end = map(anarci_res.get, ['domain_numbering', 'start', 'end'])

        assert domain_numbering is not None
        
        cdr_def = get_tcr_regions(domain_numbering, chain_id=chain_id)
        
        updated_feature = {k : v[domain_start:domain_end] for k, v in feature.items()}
        domain_numbering = ','.join([''.join([str(xx) for xx in x]).strip() for x in domain_numbering])
        prefix = 'alpha' if chain_id == 'A' else 'beta'

        cdr_dict = dict(
                cdr_def=cdr_def, 
                numbering=domain_numbering
                )
        cdr_dict = {f'{prefix}_{k}' : v for k, v in cdr_dict.items()}
        updated_feature.update(cdr_dict)
        
        return updated_feature
    
    feature = {}
    
    if beta_data:
        str_seq, seq2struc, struc = map(beta_data.get, ['str_seq', 'seqres_to_structure', 'struc'])
        beta_feature = make_feature(str_seq, seq2struc, struc, 'B')
        beta_feature = _make_domain(beta_feature, 'B')
        feature.update(beta_feature)
    
    if alpha_data:
        str_seq, seq2struc, struc = map(alpha_data.get, ['str_seq', 'seqres_to_structure', 'struc'])
        alpha_feature = make_feature(str_seq, seq2struc, struc, 'A')
        alpha_feature = _make_domain(alpha_feature, 'A')
        feature.update(alpha_feature)

    if antigen_data:
        str_seq, seq2struc, struc = map(antigen_data.get, ['str_seq', 'seqres_to_structure', 'struc'])
        antigen_feature = make_feature(str_seq, seq2struc, struc, 'G')
        feature.update(antigen_feature)

    if mhc_data:
        str_seq, seq2struc, struc = map(mhc_data.get, ['str_seq', 'seqres_to_structure', 'struc'])
        mhc_feature = make_feature(str_seq, seq2struc, struc, 'M')
        feature.update(mhc_feature)

    return feature

def save_feature(feature, code, beta_chain_id, alpha_chain_id, antigen_chain_id, mhc_chain_id, output_dir):
    np.savez(os.path.join(output_dir, f'{code}_{beta_chain_id}_{alpha_chain_id}_{antigen_chain_id}_{mhc_chain_id}.npz'), **feature)
    
    with open(os.path.join(output_dir, f'{code}_{beta_chain_id}_{alpha_chain_id}_{antigen_chain_id}_{mhc_chain_id}.fasta'), 'w') as fw:
        if 'beta_numbering' in feature:
            fw.write(f'>{code}_{beta_chain_id}\n{feature["beta_str_seq"]}\n')
        if 'alpha_numbering' in feature:
            fw.write(f'>{code}_{alpha_chain_id}\n{feature["alpha_str_seq"]}\n')
        if 'antigen_chain_id' in feature:
            fw.write(f'>{code}_{antigen_chain_id}\n{feature["antigen_str_seq"]}\n')
        if 'mhc_chain_id' in feature:
            fw.write(f'>{code}_{mhc_chain_id}\n{feature["mhc_str_seq"]}\n')

    with open(os.path.join(output_dir, f'{code}_{beta_chain_id}_{alpha_chain_id}_{antigen_chain_id}_{mhc_chain_id}_test.fasta'), 'w') as fw:
        str = []
        chain_id = []
        if 'beta_numbering' in feature:
            str.append(feature["beta_str_seq"])
        chain_id.append(beta_chain_id)
        if 'alpha_numbering' in feature:
            str.append(feature["alpha_str_seq"])
        chain_id.append(alpha_chain_id)
        if 'antigen_chain_id' in feature:
            str.append(feature["antigen_str_seq"])
        chain_id.append(antigen_chain_id)
        if 'mhc_chain_id' in feature:
            str.append(feature["mhc_str_seq"])
        chain_id.append(mhc_chain_id)
        chain_id = '_'.join(chain_id)
        str_seq = ':'.join(str)
        fw.write(f'>{code}_{chain_id}\n{str_seq}\n')
        
    with open(os.path.join(output_dir, f'{code}_{beta_chain_id}_{alpha_chain_id}_{antigen_chain_id}_{mhc_chain_id}_test_ab.fasta'), 'w') as fw:
        str = []
        chain_id = []
        if 'beta_numbering' in feature:
            str.append(feature["beta_str_seq"])
        chain_id.append(beta_chain_id)
        if 'alpha_numbering' in feature:
            str.append(feature["alpha_str_seq"])
        chain_id.append(alpha_chain_id)
        # if 'antigen_chain_id' in feature:
        #     str.append(feature["antigen_str_seq"])
        chain_id.append(antigen_chain_id)
        # if 'mhc_chain_id' in feature:
        #     str.append(feature["mhc_str_seq"])
        chain_id.append(mhc_chain_id)
        chain_id = '_'.join(chain_id)
        str_seq = ':'.join(str)
        fw.write(f'>{code}_{chain_id}\n{str_seq}\n')
    
    with open(os.path.join(output_dir, f'{code}_{beta_chain_id}_{alpha_chain_id}_{antigen_chain_id}_{mhc_chain_id}_ab.fasta'), 'w') as fw:
        if 'beta_numbering' in feature:
            fw.write(f'>{code}_{beta_chain_id}\n{feature["beta_str_seq"]}\n')
        if 'alpha_numbering' in feature:
            fw.write(f'>{code}_{alpha_chain_id}\n{feature["alpha_str_seq"]}\n')

    return

def save_header(header, file_path):
    with open(file_path, 'w') as fw:
        json.dump(header, fw)

def process(code, chain_ids, args):
    logging.info(f'processing {code}, {",".join(["_".join(x) for x in chain_ids])}')
    mmcif_file = os.path.join(args.mmcif_dir, f'{code}.cif')
    try:
        parsing_result = mmcif_parse(file_id=code, mmcif_file=mmcif_file)
    except PDBConstructionException as e:
        logging.warning('mmcif_parse: %s {%s}', mmcif_file, str(e))
    except Exception as e:
        logging.warning('mmcif_parse: %s {%s}', mmcif_file, str(e))
        raise Exception('...') from e
    if not parsing_result.mmcif_object:
        return
    
    # save_header(parsing_result.mmcif_object.header, 
    #         os.path.join(args.output_dir, f'{code}.json'))

    struc = parsing_result.mmcif_object.structure 
    
    def _parse_chain_id(beta_chain_id, alpha_chain_id):
        if beta_chain_id.islower() and beta_chain_id.upper() == alpha_chain_id:
            beta_chain_id = beta_chain_id.upper()
        elif alpha_chain_id.islower() and alpha_chain_id.upper() == beta_chain_id:
            alpha_chain_id = alpha_chain_id.upper()
        return beta_chain_id, alpha_chain_id

    for orig_beta_chain_id, orig_alpha_chain_id, origin_antigen_chain_id, origin_mhc_chain_id in chain_ids:
        beta_chain_id, alpha_chain_id = _parse_chain_id(orig_beta_chain_id, orig_alpha_chain_id)

        if ((beta_chain_id and beta_chain_id not in parsing_result.mmcif_object.chain_to_seqres) or
            (alpha_chain_id and alpha_chain_id not in parsing_result.mmcif_object.chain_to_seqres)):
            logging.warn(f'{code} {beta_chain_id} {alpha_chain_id}: chain ids not exist.')
            continue

        if beta_chain_id:
            beta_data = dict(
                    str_seq = parsing_result.mmcif_object.chain_to_seqres[beta_chain_id],
                    seqres_to_structure = parsing_result.mmcif_object.seqres_to_structure[beta_chain_id],
                    struc = struc[beta_chain_id])
        else:
            beta_data = None
        
        if alpha_chain_id:
            alpha_data = dict(
                    str_seq = parsing_result.mmcif_object.chain_to_seqres[alpha_chain_id],
                    seqres_to_structure = parsing_result.mmcif_object.seqres_to_structure[alpha_chain_id],
                    struc = struc[alpha_chain_id])
        else:
            alpha_data = None
        
        if origin_antigen_chain_id:
            antigen_data = dict(
                    str_seq = parsing_result.mmcif_object.chain_to_seqres[origin_antigen_chain_id],
                    seqres_to_structure = parsing_result.mmcif_object.seqres_to_structure[origin_antigen_chain_id],
                    struc = struc[origin_antigen_chain_id])
        else:
            antigen_data = None
        
        if origin_mhc_chain_id:
            mhc_data = dict(
                    str_seq = parsing_result.mmcif_object.chain_to_seqres[origin_mhc_chain_id],
                    seqres_to_structure = parsing_result.mmcif_object.seqres_to_structure[origin_mhc_chain_id],
                    struc = struc[origin_mhc_chain_id])
        else:
            mhc_data = None

        try:
            feature = make_npz(beta_data, alpha_data, antigen_data, mhc_data)
            save_feature(feature, code, orig_beta_chain_id, orig_alpha_chain_id, origin_antigen_chain_id, origin_mhc_chain_id, args.output_dir)
            logging.info(f'succeed: {mmcif_file} {orig_beta_chain_id} {orig_alpha_chain_id}')
        except Exception as e:
            traceback.print_exc()
            logging.error(f'make structure: {mmcif_file} {orig_beta_chain_id} {orig_alpha_chain_id} {str(e)}')

def main(args):
    func = functools.partial(process, args=args)
    
    with mp.Pool(args.cpus) as p:
        p.starmap(func, parse_list(args.summary_file))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--cpus', type=int, default=1)
  parser.add_argument('--summary_file', type=str, required=True)
  parser.add_argument('--mmcif_dir', type=str, required=True)
  parser.add_argument('--output_dir', type=str, required=True)
  parser.add_argument('--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args)
