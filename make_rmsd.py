import os
import argparse
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd

from carbonmatrix.common.ab.metrics import calc_ab_metrics
from carbonmatrix.common import residue_constants

from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1
# from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

import pdb

def read_fasta(fasta_file):
    with open(fasta_file) as f:
        lines = f.readlines()
    return lines[1].strip(), lines[3].strip()

def make_pred_coords(pdb_file, alg_type, heavy_len=0, light_len=0):
    sep_pad_num = 48
    parser = PDBParser(QUIET=1)
    model = parser.get_structure('pdb', pdb_file)[0]

    
    if alg_type in ['imb']:
        residues = list(model['B'].get_residues()) + list(model['A'].get_residues())
    elif alg_type in ['esmfold', 'alphafold']:
        residues = list(model['A'].get_residues()) + list(model['B'].get_residues())
    elif alg_type in ['omegafold']:
        residues = list(model.get_residues())
        residues = residues[:heavy_len] + residues[heavy_len+sep_pad_num:heavy_len+sep_pad_num+light_len]

    coords = np.zeros((len(residues), 3))
    # pdb.set_trace()
    # for i, r in enumerate(residues):
    #     coords[i] = r['CA'].get_coord()

    coords = np.zeros((len(residues), 14, 3))
    coord_mask = np.zeros((len(residues), 14), dtype=bool)
    for i, residue in enumerate(residues):
        if residue.resname not in residue_constants.restype_name_to_atom14_names.keys():
            break
        res_atom14_list = residue_constants.restype_name_to_atom14_names[residue.resname]
        for atom in residue.get_atoms():
            if atom.id not in res_atom14_list:
                continue
            atom14idx = res_atom14_list.index(atom.id)
            coords[i, atom14idx] = atom.get_coord()
            coord_mask[i, atom14idx]= True

    return coords, coord_mask

def make_coords(pdb_file, mode):
    parser = PDBParser(QUIET=1)
    model = parser.get_structure('pdb', pdb_file)[0]
    name = pdb_file.split('/')[-1].split('.')[0]
    chain_ids = name.split('_')[1:]
    if mode == 'unbound':
        chain_ids = chain_ids[:2]
    else:
        chain_ids = chain_ids
    residues = []
    single_codes = []
    for chain_id in chain_ids:
        if chain_id:
            residue = list(model[chain_id].get_residues())
            residues.append(residue)
            single_codes.append([seq1(r.get_resname()) for r in residue if r.get_resname() in residue_constants.restype_3to1.keys()])
        else:
            residues.append([])
            single_codes.append([])
    str_seq_list = [''.join(sc) for sc in single_codes]
    coords_list = []
    coord_mask_list = []
    for r in residues:
        if r:
            
            coords = np.zeros((len(r), 14, 3))
            coord_mask = np.zeros((len(r), 14), dtype=bool)
            # pdb.set_trace()            
            for i, residue in enumerate(r):
                res_atom14_list = residue_constants.restype_name_to_atom14_names[residue.resname]
                
                for atom in residue.get_atoms():
                    if atom.id not in res_atom14_list:
                        continue
                    atom14idx = res_atom14_list.index(atom.id)
                    coords[i, atom14idx] = atom.get_coord()
                    coord_mask[i, atom14idx]= True
            # pdb.set_trace()
            coords = coords[:i+1]
            coord_mask = coord_mask[:i]
            coords_list.append(coords)
            coord_mask_list.append(coord_mask)
        else:
            coords_list.append(np.zeros((0, 14, 3)))
            coord_mask_list.append(np.zeros((0, 14), dtype=bool))
    # pdb.set_trace()

    return coords_list, coord_mask_list, str_seq_list


def make_one(name, gt_npz_file, pred_file, alg_type, pdb_dir, ig='tcr', mode='unbound'):
    gt_fea = np.load(gt_npz_file, allow_pickle=True)
    
    coord_mask = []
    coords = []
    gt_ab_coords = []
    gt_ab_coord_mask = []
    cdr_def = []
    chains = []
    if ig == 'ab':
        code, heavy_chain, light_chain, antigen_chain = name.split('_')
        heavy_str_seq, heavy_chain_id, heavy_coords, heavy_coord_mask, heavy_cdr_def = gt_fea.get('heavy_str_seq'), gt_fea.get('heavy_chain_id'), gt_fea.get('heavy_coords'), gt_fea.get('heavy_coord_mask'), gt_fea.get('heavy_cdr_def')
        gt_ab_coords.append(heavy_coords)
        gt_ab_coord_mask.append(heavy_coord_mask)
        coords.append(heavy_coords)
        coord_mask.append(heavy_coord_mask)
        cdr_def.append(heavy_cdr_def)
        chains = [heavy_chain, light_chain, antigen_chain]
        if 'light_str_seq' in gt_fea.files():
            light_str_seq, light_chain_id, light_coords, light_coord_mask, light_cdr_def = gt_fea.get('light_str_seq'), gt_fea.get('light_chain_id'), gt_fea.get('light_coords'), gt_fea.get('light_coord_mask'), gt_fea.get('light_cdr_def')
            gt_ab_coords.append(light_coords)
            gt_ab_coord_mask.append(light_coord_mask)
            coords.append(light_coords)
            coord_mask.append(light_coord_mask)
            cdr_def.append(light_cdr_def)

        if 'antigen_str_seq' in gt_fea.files():
            antigen_str_seq, antigen_chain_id, antigen_coords, antigen_coord_mask = gt_fea.get('antigen_str_seq'), gt_fea.get('antigen_chain_id'), gt_fea.get('antigen_coords'), gt_fea.get('antigen_coord_mask')
            coords.append(antigen_coords)
            coord_mask.append(antigen_coord_mask)
    else:
        code, heavy_chain, light_chain, antigen_chain, mhc_chain = name.split('_')
        heavy_str_seq, heavy_chain_id, heavy_coords, heavy_coord_mask, heavy_cdr_def = gt_fea.get('beta_str_seq'), gt_fea.get('beta_chain_id'), gt_fea.get('beta_coords'), gt_fea.get('beta_coord_mask'), gt_fea.get('beta_cdr_def')
        gt_ab_coords.append(heavy_coords)
        gt_ab_coord_mask.append(heavy_coord_mask)
        coords.append(heavy_coords)
        coord_mask.append(heavy_coord_mask)
        cdr_def.append(heavy_cdr_def)
        chains = [heavy_chain, light_chain, antigen_chain, mhc_chain]

        if 'alpha_str_seq' in gt_fea.files:
            light_str_seq, light_chain_id, light_coords, light_coord_mask, light_cdr_def = gt_fea.get('alpha_str_seq'), gt_fea.get('alpha_chain_id'), gt_fea.get('alpha_coords'), gt_fea.get('alpha_coord_mask'), gt_fea.get('alpha_cdr_def')
            gt_ab_coords.append(light_coords)
            gt_ab_coord_mask.append(light_coord_mask)
            coords.append(light_coords)
            coord_mask.append(light_coord_mask)
            cdr_def.append(light_cdr_def)

        if 'antigen_str_seq' in gt_fea.files:
            antigen_str_seq, antigen_chain_id, antigen_coords, antigen_coord_mask = gt_fea.get('antigen_str_seq'), gt_fea.get('antigen_chain_id'), gt_fea.get('antigen_coords'), gt_fea.get('antigen_coord_mask')
            coords.append(antigen_coords)
            coord_mask.append(antigen_coord_mask)
        if 'mhc_str_seq' in gt_fea.files:
            mhc_str_seq, mhc_chain_id, mhc_coords, mhc_coord_mask = gt_fea.get('mhc_str_seq'), gt_fea.get('mhc_chain_id'), gt_fea.get('mhc_coords'), gt_fea.get('mhc_coord_mask')
            coords.append(mhc_coords)
            coord_mask.append(mhc_coord_mask)

    if alg_type == 'tcrfold':
        pred_coords_list, pred_coord_mask_list, pred_str_seq_list = make_coords(pred_file, args.mode)
    elif alg_type in ['imb', 'esmfold', 'alphafold']:
        pred_coords_list, pred_coord_mask_list = make_pred_coords(pred_file, alg_type)
    elif alg_type in ['omegafold']:
        heavy_len, light_len = len(str(heavy_str_seq)), len(str(light_str_seq))
        pred_coords_list, pred_coord_mask_list = make_pred_coords(pred_file, alg_type, heavy_len, light_len)

    if alg_type == 'tcrfold':
        pred_coords = np.concatenate(pred_coords_list, axis=0)
        pred_coord_mask = np.concatenate(pred_coord_mask_list, axis=0)
        if light_chain != None:
            pred_ab_coords = np.concatenate(pred_coords_list[:2], axis=0)
        else:
            pred_ab_coords = pred_coords_list[0]
    elif alg_type in ['imb', 'esmfold', 'omegafold', 'alphafold']:
        pred_coords = pred_coords_list
        pred_coord_mask = pred_coord_mask_list
        pred_ab_coords = pred_coords
    
    gt_coords = np.concatenate(coords, axis=0)
    gt_coord_mask = np.concatenate(coord_mask, axis=0)
    cdr_def = np.concatenate(cdr_def, axis=0)

    gt_ab_coords = np.concatenate(gt_ab_coords, axis=0)
    gt_ab_coord_mask = np.concatenate(gt_ab_coord_mask, axis=0)

    ca_mask = gt_coord_mask[:, 1]
    gt_ca = gt_coords[:,1]
    gt_ab_ca = gt_ab_coords[:,1]
    ab_ca_mask = gt_ab_coord_mask[:,1]
    pred_ab_ca = pred_ab_coords[:,1]

    N = len(str(heavy_str_seq)) + len(str(light_str_seq))
    assert(N==gt_ab_ca.shape[0] and N==ab_ca_mask.shape[0] and N==cdr_def.shape[0])
    assert (N == pred_ab_ca.shape[0])
    ab_metrics = calc_ab_metrics(gt_coords, pred_coords, gt_ab_ca, pred_ab_ca, ca_mask, ab_ca_mask, cdr_def, remove_middle_residues=True)
    return ab_metrics

def main(args):
    names = list(pd.read_csv(args.name_idx, names=['name'], header=None)['name'])

    metrics = []
    for i, n in enumerate(names):
        gt_file = os.path.join(args.gt_dir, n + '.npz')
        pred_file = os.path.join(args.pred_dir, n + '.pdb')
        # import pdb
        # pdb.set_trace()
        if os.path.exists(gt_file) and os.path.exists(pred_file):
            one_metric = OrderedDict({'name' : n})
            rmsd_metric = make_one(n, gt_file, pred_file, args.alg_type, args.pdb_dir, args.ig, args.mode)
            one_metric.update(rmsd_metric)
            metrics.append(one_metric)

    columns = metrics[0].keys()
    metrics = zip(*map(lambda x:x.values(), metrics))

    df = pd.DataFrame(dict(zip(columns,metrics)))
    # print('all', df.shape)
    # print(df['heavy_cdr3_coverage'].describe())
    # print(f"df: {df}")
    len_thres=3
    df = df[df['heavy_cdr3_len'] >=len_thres]
    # print('length >= ', len_thres, df.shape)
    
    df = df[df['heavy_cdr3_coverage'] >= 1.0]
    # print('coverage=1.0', df.shape)
    
    df.to_csv(args.output, sep='\t', index=False)
    df = df.dropna()

    def _average_all_framework_regions(x):
        return np.mean([x['heavy_fr1_rmsd'], x['heavy_fr2_rmsd'], x['heavy_fr3_rmsd'], x['heavy_fr4_rmsd'], x['light_fr1_rmsd'], x['light_fr2_rmsd'], x['light_fr3_rmsd'], x['light_fr4_rmsd']])

    def _average_all_cdr_gegions(x):
        return np.mean([x['heavy_cdr1_rmsd'], x['heavy_cdr2_rmsd'], x['light_cdr1_rmsd'], x['light_cdr2_rmsd'], x['light_cdr3_rmsd']])

    df['avg_fr_rmsd'] = df.apply(_average_all_framework_regions, axis=1)
    df['avg_other_cdr_rmsd'] = df.apply(_average_all_cdr_gegions, axis=1)
    print(f"+++"*20)
    print(f"Algorithm: {args.alg_type}")
    print('final total', df.shape[0])
    for r in df.columns:
        if r.endswith('rmsd'):
            rmsd = df[r].values
            mean = np.mean(rmsd)
            std = np.std(rmsd)
            max_ = np.max(rmsd)
            print(f'{r:15s} {mean:6.2f} {std:6.2f} {max_:6.2f}')
        
    # print(df['heavy_cdr3_len'].describe())
    # print(df['heavy_cdr3_rmsd'].describe())
    # print(df['full_rmsd'].describe())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--name_idx', type=str, required=True)
    parser.add_argument('-t', '--alg_type', type=str, choices=['imb', 'tcrmodel', 'tcrfold', 'omegafold', 'esmfold', 'alphafold'], required=True)
    parser.add_argument('-g', '--gt_dir', type=str, required=True)
    parser.add_argument('-p', '--pred_dir', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-m', '--mode', type=str, choices=['unbound', 'bound'], default='unbound')
    parser.add_argument('-i', '--ig', type=str, choices=['ab', 'tcr'], default='tcr')
    parser.add_argument('-d', '--pdb_dir', type=str, required=True)
    args = parser.parse_args()

    main(args)

