import os
import argparse
import functools
import multiprocessing as mp
import logging
import itertools
import json
import pandas as pd
import traceback

import numpy as np

from Bio.PDB.PDBExceptions import PDBConstructionException

from abfold.common import residue_constants
from abfold.data.mmcif_parsing import parse as mmcif_parse
from abfold.preprocess.numbering import renumber_ab_seq, get_ab_regions

def parse_list(path):
    with open(path) as f:
        names = [n.strip() for n in f]
    
    df = pd.read_csv(path, sep='\t')
    df = df[df['pdb'].isin(['7pi3'])]
    
    logging.info(f'all pairs: {df.shape[0]}')
    df = df.fillna({'Hchain':'', 'Lchain':''})
    logging.info(f'number of complete pairs: {df.shape[0]}')
   
    for code, r in df.groupby(by='pdb'):
        chain_list = list(zip(r['Hchain'], r['Lchain']))
        yield (code, chain_list)

def make_feature(str_seq, seq2struc, structure):
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
    
    feature = dict(str_seq=str_seq,
            coords=coords,
            coord_mask=coord_mask)

    return feature

def make_npz(heavy_data, light_data):
    def _make_domain(feature, chain_id):
        allow = ['H'] if chain_id == 'H' else ['K', 'L']

        anarci_res = renumber_ab_seq(feature['str_seq'], allow=allow, scheme='imgt')
        domain_numbering, domain_start, domain_end = map(anarci_res.get, ['domain_numbering', 'start', 'end'])

        assert domain_numbering is not None
        
        cdr_def = get_ab_regions(domain_numbering, chain_id=chain_id)
        
        updated_feature = {k : v[domain_start:domain_end] for k, v in feature.items()}
        domain_numbering = ','.join([''.join([str(xx) for xx in x]).strip() for x in domain_numbering])

        updated_feature.update(dict(cdr_def=cdr_def, numbering=domain_numbering))
        
        prefix = 'heavy' if chain_id == 'H' else 'light'

        return {f'{prefix}_{k}' : v for k, v in updated_feature.items()}
    
    feature = {}
    
    if heavy_data:
        str_seq, seq2struc, struc = map(heavy_data.get, ['str_seq', 'seqres_to_structure', 'struc'])
        heavy_feature = make_feature(str_seq, seq2struc, struc)
        heavy_feature = _make_domain(heavy_feature, 'H')
        feature.update(heavy_feature)
    
    if light_data:
        str_seq, seq2struc, struc = map(light_data.get, ['str_seq', 'seqres_to_structure', 'struc'])
        light_feature = make_feature(str_seq, seq2struc, struc)
        light_feature = _make_domain(light_feature, 'L')
        feature.update(light_feature)

    return feature

def save_feature(feature, code, heavy_chain_id, light_chain_id, output_dir):
    np.savez(os.path.join(output_dir, f'{code}_{heavy_chain_id}_{light_chain_id}.npz'), **feature)
    
    with open(os.path.join(output_dir, f'{code}_{heavy_chain_id}_{light_chain_id}.fasta'), 'w') as fw:
        if 'heavy_numbering' in feature:
            fw.write(f'>{code}_H {feature["heavy_numbering"]}\n{feature["heavy_str_seq"]}\n')
        if 'light_numbering' in feature:
            fw.write(f'>{code}_L {feature["light_numbering"]}\n{feature["light_str_seq"]}\n')
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
    
    save_header(parsing_result.mmcif_object.header, 
            os.path.join(args.output_dir, f'{code}.json'))

    struc = parsing_result.mmcif_object.structure 
    
    def _parse_chain_id(heavy_chain_id, light_chain_id):
        if heavy_chain_id.islower() and heavy_chain_id.upper() == light_chain_id:
            heavy_chain_id = heavy_chain_id.upper()
        elif light_chain_id.islower() and light_chain_id.upper() == heavy_chain_id:
            light_chain_id = light_chain_id.upper()
        return heavy_chain_id, light_chain_id

    for orig_heavy_chain_id, orig_light_chain_id in chain_ids:
        heavy_chain_id, light_chain_id = _parse_chain_id(orig_heavy_chain_id, orig_light_chain_id)

        if ((heavy_chain_id and heavy_chain_id not in parsing_result.mmcif_object.chain_to_seqres) or
            (light_chain_id and light_chain_id not in parsing_result.mmcif_object.chain_to_seqres)):
            logging.warn(f'{code} {heavy_chain_id} {light_chain_id}: chain ids not exist.')
            continue

        if heavy_chain_id:
            heavy_data = dict(
                    str_seq = parsing_result.mmcif_object.chain_to_seqres[heavy_chain_id],
                    seqres_to_structure = parsing_result.mmcif_object.seqres_to_structure[heavy_chain_id],
                    struc = struc[heavy_chain_id])
        else:
            heavy_data = None
        
        if light_chain_id:
            light_data = dict(
                    str_seq = parsing_result.mmcif_object.chain_to_seqres[light_chain_id],
                    seqres_to_structure = parsing_result.mmcif_object.seqres_to_structure[light_chain_id],
                    struc = struc[light_chain_id])
        else:
            light_data = None

        try:
            feature = make_npz(heavy_data, light_data)
            save_feature(feature, code, orig_heavy_chain_id, orig_light_chain_id, args.output_dir)
            logging.info(f'succeed: {mmcif_file} {orig_heavy_chain_id} {orig_light_chain_id}')
        except Exception as e:
            traceback.print_exc()
            logging.error(f'make structure: {mmcif_file} {orig_heavy_chain_id} {orig_light_chain_id} {str(e)}')

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
