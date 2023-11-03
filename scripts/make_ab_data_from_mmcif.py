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

sys.path.insert(0, '../carbonmatrix')

import numpy as np

from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB.Structure import Structure as PDBStructure

from carbonmatrix.common import residue_constants
from carbonmatrix.data.mmcif_parsing import parse as mmcif_parse, MmcifObject
from carbonmatrix.data.antibody.seq import make_ab_numbering, calc_epitope
from carbonmatrix.data.antibody import antibody_constants
from carbonmatrix.data.pdbio import make_gt_chain

'''
npz format
{
    "heavy_chain" : {
        "coord" : array,
        "coord_mask": array,
        "seq_str": array,
        "residx": array,
    },
    "light_chain" : {},
    "antigens": [
        (chain1, {}),
        (chain2, {}),
    ]
}

'''

@dataclass(frozen=True)
class ComplexItem:
    code: str
    heavy_chain_id: str
    light_chain_id: str
    antigen_chain_ids: field(default_factory=[])
    antigen_types: field(default_factory=[])

# parse the sabdab summary file
def parse_list(path):
    with open(path) as f:
        names = [n.strip() for n in f]

    df = pd.read_csv(path, sep='\t')
    df = df[df['method'].isin(['X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY'])]

    logging.info(f'all pairs: {df.shape[0]}')

    df = df.fillna({'Hchain':'', 'Lchain':'', 'antigen_chain':'', 'antigen_type': ''})
    df = df[df['Hchain'] != '']
    logging.info(f'number of H chains: {df.shape[0]}')

    print(df['model'].value_counts())

    # df = df[df['model'] != 0]
    # logging.info(f'number of model 0: {df.shape[0]}')

    def _make_item(x):
        heavy_chain_id, light_chain_id = x['Hchain'], x['Lchain']
        if heavy_chain_id.islower() and heavy_chain_id.upper() == light_chain_id:
            heavy_chain_id = heavy_chain_id.upper()
        elif light_chain_id.islower() and light_chain_id.upper() == heavy_chain_id:
            light_chain_id = light_chain_id.upper()

        return ComplexItem(
            code = x['pdb'],
            heavy_chain_id = heavy_chain_id,
            light_chain_id = light_chain_id,
            antigen_chain_ids = [a.strip() for a in x['antigen_chain'].split('|')],
            antigen_types = [a.strip() for a in x['antigen_type'].split('|')],
            )

    for (code, model_id), g in df.groupby(by=['pdb', 'model']):
        complex_items = []

        for i, r in g.iterrows():
            complex_items.append(_make_item(r))

        yield (code, model_id, complex_items)

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
    print(meta)
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

    for i, (chain_id, chain_feat) in enumerate(feature['antigens']):
        residue_ids = [(i, ' ') for i in chain_feat['epitope_index']]
        new_chain_id = chr(ord('M') + i)
        chain = make_gt_chain(new_chain_id, chain_feat['seq'], chain_feat['coords'], chain_feat['coord_mask'], residue_ids)
        model.add(chain)

    structure = PDBStructure(id='ab')
    structure.add(model)

    io = PDBIO(use_model_flag=1)
    io.set_structure(structure)

    io.save(file_path)

    return

def save(complex_item : ComplexItem, feature, meta, output_base_dir):
    name = complex_item.code + '_' + complex_item.heavy_chain_id + '_' + complex_item.light_chain_id

    npy_file = os.path.join(output_base_dir, 'npy', name + '.npy')
    np.save(npy_file, feature, allow_pickle=True)

    heavy_seq = feature['heavy_chain']['seq']
    light_seq = ''
    if 'light_chain' in feature:
        light_seq = feature['light_chain']['seq']

    fasta_file = os.path.join(output_base_dir, 'heavy_first_fasta', name + '.fasta')
    save_fasta(name, heavy_seq + ':' + light_seq, fasta_file)

    fasta_file = os.path.join(output_base_dir, 'light_first_fasta', name + '.fasta')
    save_fasta(name, light_seq + ':' + heavy_seq, fasta_file)

    cdrh3_len = np.sum(feature['heavy_chain']['region_index'] == antibody_constants.region_type_order['CDR-H3'])
    meta.update(
        name = name,
        code = complex_item.code,
        heavy_seq = heavy_seq,
        light_seq = light_seq,
        cdrh3_len = int(cdrh3_len),
    )
    meta_file = os.path.join(output_base_dir, 'meta', name + '.json')
    save_meta(meta, meta_file)

    pdb_file = os.path.join(output_base_dir, 'pdb', name + '.pdb')

    save_pdb(feature, pdb_file)

    return

def make_complex(complex_item: ComplexItem, mmcif_object:MmcifObject, args):
    ret = {}

    def _make_chain(chain_id, allow):
        str_seq = mmcif_object.chain_to_seqres[chain_id]
        struc = make_struc(
            str_seq,
            mmcif_object.seqres_to_structure[chain_id],
            mmcif_object.structure[chain_id],
            )

        ab_def = make_ab_numbering(str_seq, allow)

        new_struc = {k : v[ab_def['query_start']:ab_def['query_end']] for k, v in struc.items()}

        new_struc.update(
            numbering=ab_def['numbering'],
            region_index=ab_def['region_index'])

        return new_struc

    def _make_antigen(ab_coords, ab_coord_mask, chain_id):
        str_seq = mmcif_object.chain_to_seqres[chain_id]
        struc = make_struc(
            str_seq,
            mmcif_object.seqres_to_structure[chain_id],
            mmcif_object.structure[chain_id],
            )

        antigen_coords, antigen_coord_mask = struc['coords'], struc['coord_mask']
        epitope_index, epitope_dist = calc_epitope(ab_coords, ab_coord_mask, antigen_coords, antigen_coord_mask, dist_thres=8.0)

        if epitope_index is None:
            return None

        new_struc = dict(
            seq = ''.join([str_seq[i] for i in epitope_index]),
            coords = antigen_coords[epitope_index],
            coord_mask = antigen_coord_mask[epitope_index],
            epitope_index = epitope_index,
            epitope_dist = epitope_dist,
        )

        return new_struc

    ret['heavy_chain'] = _make_chain(complex_item.heavy_chain_id, allow=['H'])

    ab_coords, ab_coord_mask = ret['heavy_chain']['coords'], ret['heavy_chain']['coord_mask']

    if complex_item.light_chain_id:
        ret['light_chain'] = _make_chain(complex_item.light_chain_id, allow=['K','L'])
        ab_coords = np.concatenate([ab_coords, ret['light_chain']['coords']], axis=0)
        ab_coord_mask = np.concatenate([ab_coord_mask, ret['light_chain']['coord_mask']], axis=0)

    antigens = []
    for chain_id, antigen_type in zip(complex_item.antigen_chain_ids, complex_item.antigen_types):
        if antigen_type not in ['protein', 'peptide']:
            continue

        epitope = _make_antigen(ab_coords, ab_coord_mask, chain_id)
        antigens.append((chain_id, epitope))

    ret.update(antigens=antigens)

    return ret

def process(code, model_id, complex_items, args):
    logging.info('mmcif_parse: processing %s', code)

    mmcif_file = os.path.join(args.mmcif_dir, f'{code}.cif')

    try:
        parsing_result = mmcif_parse(file_id=code, mmcif_file=mmcif_file, model_id=model_id)
    except PDBConstructionException as e:
        logging.warning('mmcif_parse: %s {%s}', mmcif_file, str(e))
    except Exception as e:
        logging.warning('mmcif_parse: %s {%s}', mmcif_file, str(e))
        raise Exception('...') from e

    if not parsing_result.mmcif_object:
        logging.warning('mmcif_parse: mmcif_object is empty {%s}', mmcif_file)
        return

    for complex_item in complex_items:
        try:
            ret = make_complex(complex_item, parsing_result.mmcif_object, args)
            save(complex_item, ret, parsing_result.mmcif_object.header, args.output_base_dir)
        except Exception:
            logging.warn(f'code= {complex_item.code}, heavy= {complex_item.heavy_chain_id} light= {complex_item.light_chain_id} error')
            traceback.print_exc()

def main(args):
    # output dirs
    if not os.path.exists(os.path.join(args.output_base_dir, 'npz')):
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
