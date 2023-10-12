import os
import argparse
import functools
import multiprocessing as mp
import logging
import itertools
import json

import numpy as np

from Bio.PDB.PDBExceptions import PDBConstructionException

from abfold.common import residue_constants
from abfold.data.mmcif_parsing import parse as mmcif_parse

def parse_list(path):
    with open(path) as f:
        names = [n.strip() for n in f]
    for code, full_code in itertools.groupby(names, key=lambda x:x[:4]):
        yield (code, list(full_code))

def make_npz(code, chain_id, str_seq, seq2struc, structure, args):
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

    np.savez(os.path.join(args.output_dir, f'{code}_{chain_id}.npz'), **feature)

    return

def save_header(header, file_path):
    with open(file_path, 'w') as fw:
        json.dump(header, fw)

def process(code, full_code, args):
    logging.info(f'{code}, {",".join(full_code)}')
    mmcif_file = os.path.join(args.pdb_dir, f'{code}.cif')
    try:
        parsing_result = mmcif_parse(file_id=code, mmcif_file=mmcif_file)
    except PDBConstructionException as e:
        logging.warning('mmcif_parse: %s {%s}', mmcif_file, str(e))
    except Exception as e:
        logging.warning('mmcif_parse: %s {%s}', mmcif_file, str(e))
        raise Exception('...') from e
    if not parsing_result.mmcif_object:
        return
    if args.header_dir:
        save_header(parsing_result.mmcif_object.header,
                os.path.join(args.header_dir, f'{code}.json'))

    chain_ids = [c.split('_')[1] for c in full_code]
    struc = parsing_result.mmcif_object.structure
    for chain_id in chain_ids:
        if chain_id not in parsing_result.mmcif_object.chain_to_seqres:
            continue
        str_seq = parsing_result.mmcif_object.chain_to_seqres[chain_id]
        seqres_to_structure = parsing_result.mmcif_object.seqres_to_structure[chain_id]
        try:
            make_npz(code, chain_id, str_seq, seqres_to_structure, struc[chain_id], args)
        except Exception as e:
            logging.error(f'make structure: {mmcif_file} {chain_id} {str(e)}')

def main(args):
    names = parse_list(args.name_idx)

    func = functools.partial(process, args=args)

    with mp.Pool(args.cpus) as p:
        p.starmap(func, parse_list(args.name_idx))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--cpus', type=int, default=1)
  parser.add_argument('--name_idx', type=str, required=True)
  parser.add_argument('--pdb_dir', type=str, required=True)
  parser.add_argument('--output_dir', type=str, required=True)
  parser.add_argument('--header_dir', type=str)
  parser.add_argument('--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args)
