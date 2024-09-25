import os
import argparse
import functools
import multiprocessing as mp
import logging
import itertools
import json
import pandas as pd
import traceback
import pickle
import numpy as np
import pdb
from Bio.PDB.PDBExceptions import PDBConstructionException
import sys
sys.path.append('..')

from immunefold.common import residue_constants
from immunefold.data.mmcif_parsing import parse as mmcif_parse, MmcifObject
from immunefold.data.antibody.seq import make_ab_numbering, calc_epitope
from immunefold.data.antibody import antibody_constants
from immunefold.data.pdbio import make_gt_chain
from immunefold.common.ab.numbering import renumber_ab_seq, get_ab_regions, get_tcr_regions


def parse_list(path):
    with open(path) as f:
        names = [n.strip() for n in f]
    
    df = pd.read_csv(path, sep='\t')
    df = df[df['method'].isin(['X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY'])]
    
    logger.info(f'all pairs: {df.shape[0]}')
    
    df = df.fillna({'Hchain':'', 'Lchain':''})
    df = df[df['Hchain'] != '']
    logger.info(f'number of H chains: {df.shape[0]}')
    df['date'] = pd.to_datetime(df['date'])
    # df['date'] = df['date'].to_datetime()
    df = df[df['model'] == 0]
    logger.info(f'number of model 0: {df.shape[0]}')
    
    # df = df[df['antigen_chain'].notna()]
    # df = df[df['antigen_type'].str.contains('protein|peptide')]

    contains_peptide = df['antigen_type'].str.contains('protein|peptide', na=False)
    df['antigen_type'] = df['antigen_type'].fillna('NaN')
    df['antigen_type'] = df['antigen_type'].apply(lambda x: x if x in ['protein', 'peptide'] else 'NaN')
    
    df = df[contains_peptide | (df['antigen_type'] == 'NaN')]

    logger.info(f'number of antigen: {df.shape[0]}')

    for index, row in df.iterrows():
        code = row['pdb']
        heavy_chain = row['Hchain']
        light_chain = row['Lchain']
        antigen_chains = row['antigen_chain'] if type(row['antigen_chain'])!=float else ''
    
        antigen_chains = antigen_chains.split('|')
        antigen_chains = [s.replace(" ", "") for s in antigen_chains]

        for i in range(len(antigen_chains)):
            chain_list = [(row['Hchain'], row['Lchain'], antigen_chains[i])]
            if row['date']  < pd.to_datetime('2023-01-01'):
                logger.info(f'train_set@{code}_{heavy_chain}_{light_chain}_{antigen_chains[i]}')
            else:
                logger.info(f'test_set@{code}_{heavy_chain}_{light_chain}_{antigen_chains[i]}')
            yield (code, chain_list)

def continuous_flag_to_range(flag):
    first = (np.arange(0, flag.shape[0])[flag]).min().item()
    last = (np.arange(0, flag.shape[0])[flag]).max().item()
    return first, last

def Patch_idx(a, b, mask_a, mask_b, k=10):
    assert len(a.shape) == 3 and len(b.shape) == 3
    diff = a[:, np.newaxis, :, np.newaxis, :] - b[np.newaxis, :, np.newaxis, :, :]
    mask = mask_a[:, np.newaxis, :, np.newaxis] * mask_b[np.newaxis, :, np.newaxis, :]
    distance = np.where(mask, np.linalg.norm(diff, axis=-1), 1e+10)
    distance = np.min(distance.reshape(a.shape[0], b.shape[0], -1), axis=(-1,-2))
    # import pdb
    # pdb.set_trace()
    # patch_idx = np.argwhere(distance < 16).reshape(-1)
    patch_idx = np.argpartition(distance,k)[:k]
    # expanded_patch_idx = [i for j in patch_idx for i in range(j-48, j+48)]
    expanded_patch_idx = sorted(list(set(patch_idx)))
    logger.info(f"Antigen idx length@ {len(expanded_patch_idx)}")
    return expanded_patch_idx


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

def make_antigen(features):
    for i, data in enumerate(features):
        chain_id = np.full((len(data['str_seq'])), 4)
        res_nb = np.arange(0,len(data['str_seq']))
        data.update(dict(res_nb=res_nb))
        data.update(dict(chain_id=chain_id))
    chain_ids = np.concatenate([data['chain_id'] for data in features],axis=0)   
    res_nb = np.concatenate([data['res_nb'] for data in features],axis=0)         
    str_seq = ''.join([data['str_seq'] for data in features])
    coord_mask = np.concatenate([data['coord_mask'] for data in features],axis=0)
    coords = np.concatenate([data['coords'] for data in features],axis=0)
    features = dict(antigen_str_seq=str_seq,
            antigen_coords=coords,
            antigen_coord_mask=coord_mask,
            antigen_chain_id=chain_ids,
            antigen_res_nb=res_nb)
    return features
    

def PatchAroundAnchor(data, antigen_feature):
    cdr_str_to_enum = {
        'H1': 1,
        'H2': 3,
        'H3': 5,
        'L1': 8,
        'L2': 10,
        'L3': 12,
    }
    def anchor_flag_generate(data, antigen_feature):
        heavy_cdr_flag = data['heavy_cdr_def']
        heavy_anchor_flag = np.zeros((len(data['heavy_str_seq'])))

        if 'light_cdr_def' in data:
            light_cdr_flag = data['light_cdr_def']
            light_anchor_flag = np.zeros((len(data['light_str_seq'])))

        idx = []
        anchor_pos_list = []
        anchor_mask_list = []
        for sele in cdr_str_to_enum.keys():
            if sele in ['H1', 'H2', 'H3']:
                cdr_to_mask_flag = (heavy_cdr_flag == cdr_str_to_enum[sele])
                cdr_fist, cdr_last = continuous_flag_to_range(cdr_to_mask_flag)
                left_idx = max(0, cdr_fist - 1)
                right_idx = min(cdr_last + 1, len(data['heavy_str_seq'])-1)
                heavy_anchor_flag[left_idx] = cdr_str_to_enum[sele]
                heavy_anchor_flag[right_idx] = cdr_str_to_enum[sele]
                anchor_pos = data['heavy_coords'][[left_idx, right_idx]]
                anchor_mask = data['heavy_coord_mask'][[left_idx, right_idx]]
                anchor_pos_list.append(anchor_pos)
                anchor_mask_list.append(anchor_mask)        
            
            elif sele in ['L1', 'L2', 'L3'] and 'light_cdr_def' in data:
                cdr_to_mask_flag = (light_cdr_flag == cdr_str_to_enum[sele])
                cdr_fist, cdr_last = continuous_flag_to_range(cdr_to_mask_flag)
                left_idx = max(0, cdr_fist - 1)
                right_idx = min(cdr_last + 1, len(data['light_str_seq'])-1)
                light_anchor_flag[left_idx] = cdr_str_to_enum[sele]
                light_anchor_flag[right_idx] = cdr_str_to_enum[sele]
                anchor_pos = data['light_coords'][[left_idx, right_idx]]
                anchor_mask = data['light_coord_mask'][[left_idx, right_idx]]
                anchor_pos_list.append(anchor_pos)
                anchor_mask_list.append(anchor_mask)  
            anchor_pos = np.concatenate(anchor_pos_list, axis=0)
            anchor_mask = np.concatenate(anchor_mask_list, axis=0)
            antigen_pos = antigen_feature['antigen_coords']
            antigen_mask = antigen_feature['antigen_coord_mask']
            init_patch_idx = Patch_idx(antigen_pos, anchor_pos, antigen_mask, anchor_mask)
            idx = (init_patch_idx)
        
        mask = antigen_feature['antigen_coord_mask'][...,residue_constants.atom_order['CA']]
        mask_idx = np.argwhere(mask).reshape(-1).tolist()
        antigen_idx = sorted(list(set(idx).intersection(set(mask_idx))))
        # antigen_coords = antigen_feature['antigen_coords'][antigen_idx]
        # antigen_coords_mask = antigen_feature['antigen_coord_mask'][antigen_idx]
        # antigen_chain_ids = antigen_feature['antigen_chain_id'][antigen_idx]
        # antigen_str_seq = [antigen_feature['antigen_str_seq'][idx] for idx in antigen_idx]
        # antigen_residx = antigen_feature['antigen_res_nb'][antigen_idx]
        # antigen_str_seq = ''.join(antigen_str_seq) 
        # try:       
        #     assert len(antigen_idx) > 0
        # except:
        #     return None, None, None, None, None

        return antigen_idx
    
    # antigen_anchor_coords, antigen_anchor_coords_mask, antigen_anchor_str_seq, antigen_anchor_chain_ids, antigen_residx = anchor_flag_generate(data, antigen_feature)
    antigen_idx = anchor_flag_generate(data, antigen_feature)
    antigen_contact_idx = antigen_feature['antigen_res_nb'][antigen_idx]

    # if antigen_anchor_coords is not None:
    # data.update(dict(
    #                 antigen_coords = antigen_anchor_coords,
    #                 antigen_coord_mask = antigen_anchor_coords_mask,
    #                 antigen_str_seq = antigen_anchor_str_seq,
    #                 antigen_chain_id = antigen_anchor_chain_ids,
    #                 antigen_residx = antigen_residx
    #                 ))
    data.update(antigen_feature)
    data.update(dict(antigen_contact_idx = antigen_contact_idx))
    

    


def make_npz(heavy_data, light_data, antigen_data):
    def _make_domain(feature, chain_id):
        allow = ['H'] if chain_id == 'H' else ['K', 'L']

        anarci_res = renumber_ab_seq(feature['str_seq'], allow=allow, scheme='imgt')
        domain_numbering, domain_start, domain_end = map(anarci_res.get, ['domain_numbering', 'start', 'end'])
        # print(f"anarci_res: {anarci_res}")
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
        # pdb.set_trace()
        chain_ids = np.full((len(heavy_feature['heavy_str_seq'])), 1)
        heavy_feature.update(
            {'heavy_chain_id': chain_ids}
        )
        feature.update(heavy_feature)
    
    if light_data:
        str_seq, seq2struc, struc = map(light_data.get, ['str_seq', 'seqres_to_structure', 'struc'])
        # pdb.set_trace()
        light_feature = make_feature(str_seq, seq2struc, struc)
        light_feature = _make_domain(light_feature, 'L')
        chain_ids = np.full((len(light_feature['light_str_seq'])), 2)
        light_feature.update(
            {'light_chain_id': chain_ids}
        )
        feature.update(light_feature)
    if antigen_data:
        antigen_features = []
        for i, antigen_item in enumerate(antigen_data):
            str_seq, seq2struc, struc = map(antigen_item.get, ['str_seq', 'seqres_to_structure', 'struc'])
            antigen_feature = make_feature(str_seq, seq2struc, struc)
            antigen_features.append(antigen_feature)
        antigen_feature = make_antigen(antigen_features)

        PatchAroundAnchor(feature, antigen_feature)
    return feature

def save_feature(feature, code, heavy_chain_id, light_chain_id, antigen_chain_ids, output_dir):
    antigen_chain_id = ''.join(antigen_chain_ids)
    np.savez(os.path.join(output_dir, f'{code}_{heavy_chain_id}_{light_chain_id}_{antigen_chain_id}.npz'), **feature)
    with open(os.path.join(output_dir, f'{code}_{heavy_chain_id}_{light_chain_id}_{antigen_chain_id}.fasta'), 'w') as fw:
        if 'heavy_numbering' in feature:
            fw.write(f'>{code}_{heavy_chain_id} \n{feature["heavy_str_seq"]}\n')
        if 'light_numbering' in feature:
            fw.write(f'>{code}_{light_chain_id} \n{feature["light_str_seq"]}\n')
        if 'antigen_str_seq' in feature:
            # antigen_res = ','.join(map(str, feature['antigen_res_nb']))
            fw.write(f'>{code}_{antigen_chain_id} \n{feature["antigen_str_seq"]}\n')
    with open(os.path.join(output_dir, f'{code}_{heavy_chain_id}_{light_chain_id}_{antigen_chain_id}_ab.fasta'), 'w') as fw:
        if 'heavy_numbering' in feature:
            fw.write(f'>{code}_{heavy_chain_id} \n{feature["heavy_str_seq"]}\n')
        if 'light_numbering' in feature:
            fw.write(f'>{code}_{light_chain_id} \n{feature["light_str_seq"]}\n')
    return

def save_header(header, file_path):
    with open(file_path, 'w') as fw:
        json.dump(header, fw)

def process(code, chain_ids, args):
    
    logger.info(f'processing {code}, {",".join(["_".join(x) for x in chain_ids])}')
    mmcif_file = os.path.join(args.mmcif_dir, f'{code}.cif')
    try:
        parsing_result = mmcif_parse(file_id=code, mmcif_file=mmcif_file)
    except PDBConstructionException as e:
        logger.warning('mmcif_parse: %s {%s}', mmcif_file, str(e))
    except Exception as e:
        logger.warning('mmcif_parse: %s {%s}', mmcif_file, str(e))
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


    for orig_heavy_chain_id, orig_light_chain_id, orig_antigen_chain_id in chain_ids:
        # print(f"code: {code}, chain_ids: {chain_ids}")
        if os.path.exists(os.path.join(args.output_dir, f'{code}_{orig_heavy_chain_id}_{orig_light_chain_id}_{orig_antigen_chain_id}.npz')):
            logger.info(f'{code}_{orig_heavy_chain_id}_{orig_light_chain_id}_{orig_antigen_chain_id} already exists.')
            return
        heavy_chain_id, light_chain_id = _parse_chain_id(orig_heavy_chain_id, orig_light_chain_id)

        if ((heavy_chain_id and heavy_chain_id not in parsing_result.mmcif_object.chain_to_seqres) or
            (light_chain_id and light_chain_id not in parsing_result.mmcif_object.chain_to_seqres)):
            logger.warning(f'{code} {heavy_chain_id} {light_chain_id}: chain ids not exist.')
            continue
        
        flag = 0
        if orig_antigen_chain_id != '':
            antigen_chain_ids = [orig_antigen_chain_id]
            for antigen_chain_id in antigen_chain_ids:
                if antigen_chain_id not in parsing_result.mmcif_object.chain_to_seqres:
                    logger.warning(f"antigen id: {antigen_chain_id} not exist")
                    flag += 1
                    continue
            if flag > 0:
                continue
        else:
            antigen_chain_ids = [orig_antigen_chain_id]

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
        if orig_antigen_chain_id != '':
            antigen_data = []
            for antigen_chain_id in antigen_chain_ids:
                if antigen_chain_id not in parsing_result.mmcif_object.chain_to_seqres:
                    continue
                antigen_data.append(dict(
                    str_seq = parsing_result.mmcif_object.chain_to_seqres[antigen_chain_id],
                    seqres_to_structure = parsing_result.mmcif_object.seqres_to_structure[antigen_chain_id],
                    struc = struc[antigen_chain_id])
                )
        else:
            antigen_data = None

        try:
            feature = make_npz(heavy_data, light_data, antigen_data)
            save_feature(feature, code, orig_heavy_chain_id, orig_light_chain_id, antigen_chain_ids, args.output_dir)
            logger.info(f'succeed: {mmcif_file} {orig_heavy_chain_id} {orig_light_chain_id}')
        except Exception as e:
            # traceback.print_exc()
            logger.error(f'make structure: {mmcif_file} {orig_heavy_chain_id} {orig_light_chain_id} {str(e)}')

def main(args):
    func = functools.partial(process, args=args)
    
    with mp.Pool(args.cpus) as p:
        p.starmap(func, parse_list(args.summary_file))

    # for code, chain_ids in parse_list(args.summary_file):
    #     process(code, chain_ids, args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cpus', type=int, default=1)
    parser.add_argument('--summary_file', type=str, required=True)
    parser.add_argument('--mmcif_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    log_file = os.path.join(args.output_dir,'make_abfold_data.log')
    logger.info(f"log_file: {log_file}")
    handler_test = logging.FileHandler(log_file) # stdout to file
    handler_control = logging.StreamHandler()    # stdout to console

    selfdef_fmt = '%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(selfdef_fmt)
    handler_test.setFormatter(formatter)
    handler_control.setFormatter(formatter)
    logger.setLevel('DEBUG')           #设置了这个才会把debug以上的输出到控制台
    logger.addHandler(handler_test)    #添加handler
    logger.addHandler(handler_control)
    
    main(args)
