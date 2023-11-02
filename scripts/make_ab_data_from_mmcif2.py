import os
import sys
import argparse
import functools
import multiprocessing as mp
import logging
import itertools
import json
import traceback

sys.path.insert(0, '../carbonmatrix')

import numpy as np

from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBIO import PDBIO

from carbonmatrix.common import residue_constants
from carbonmatrix.data.mmcif_parsing import parse as mmcif_parse

def parse_list(path):
    with open(path) as f:
        names = [n.strip() for n in f]
    for code in names:
        yield (code,)

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

    return feature

def save_meta(meta, file_path):
    with open(file_path, 'w') as fw:
        json.dump(meta, fw, indent=2)

    return

def save_fasta(header, str_seq, file_path):
    with open(file_path, 'w') as fw:
        fw.write('>' + header + '\n')
        fw.write(str_seq + '\n')
    
    return

def save_pdb(struc, file_path):
    pdb = PDBIO()
    pdb.set_structure(struc)
    pdb.save(file_path)

    return

def save(code, chain_id, output_base_dir,
         feature, str_seq, struc, meta):
    name = code + '_' + chain_id

    npz_file = os.path.join(output_base_dir, 'npz', name + '.npz')
    fasta_file = os.path.join(output_base_dir, 'fasta', name + '.fasta')
    pdb_file = os.path.join(output_base_dir, 'pdb', name + '.pdb')
    meta_file = os.path.join(output_base_dir, 'meta', name + '.json')
    
    np.savez(npz_file, **feature)
    save_fasta(name, str_seq, fasta_file)
    try:
        save_pdb(struc, pdb_file)
    except:
        logging.error(f'save pdb {pdb_file}')

    save_meta(meta, meta_file)

    return 

def process(code, args):
    logging.info('mmcif_parse: processing %s', code)

    mmcif_file = os.path.join(args.mmcif_dir, f'{code}.cif')

    try:
        parsing_result = mmcif_parse(file_id=code, mmcif_file=mmcif_file)
    except PDBConstructionException as e:
        logging.warning('mmcif_parse: %s {%s}', mmcif_file, str(e))
    except Exception as e:
        logging.warning('mmcif_parse: %s {%s}', mmcif_file, str(e))
        raise Exception('...') from e

    if not parsing_result.mmcif_object:
        logging.warning('mmcif_parse: mmcif_object is empty {%s}', mmcif_file)
        return

    header = parsing_result.mmcif_object.header
    struc = parsing_result.mmcif_object.structure
    
    for chain_id, str_seq in parsing_result.mmcif_object.chain_to_seqres.items():
        seqres_to_structure = parsing_result.mmcif_object.seqres_to_structure[chain_id]

        try:
            feature = make_npz(code, chain_id, str_seq, seqres_to_structure, struc[chain_id], args)
            feature.update(resolution=np.float32(header['resolution']))

            meta = header.copy()
            meta.update(
                str_seq = str_seq,
                seq_len = len(str_seq),
                seq_coverage = len(str_seq) - str_seq.count('X'),
                struc_coverage = int(np.sum(feature['coord_mask'][:,1])),
            )

            save(code, chain_id, args.output_base_dir,
                 feature=feature,
                 str_seq=str_seq,
                 struc=struc[chain_id],
                 meta=meta,
                 )
        
        except Exception as e:
            logging.error(f'make structure: {code} {chain_id} {str(e)}')
            logging.error(traceback.format_exc())

def main(args):
    # output dirs
    if not os.path.exists(os.path.join(args.output_base_dir, 'npz')):
        os.mkdir(os.path.join(args.output_base_dir, 'npz'))
    if not os.path.exists(os.path.join(args.output_base_dir, 'fasta')):
        os.mkdir(os.path.join(args.output_base_dir, 'fasta'))
    if not os.path.exists(os.path.join(args.output_base_dir, 'pdb')):
        os.mkdir(os.path.join(args.output_base_dir, 'pdb'))
    if not os.path.exists(os.path.join(args.output_base_dir, 'meta')):
        os.mkdir(os.path.join(args.output_base_dir, 'meta'))

    names = parse_list(args.name_idx)
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
