import os
import argparse
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd
import random
from carbonmatrix.common.ab.metrics import calc_ab_metrics
from carbonmatrix.common import residue_constants

from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces, group_chains, get_all_chain_maps, count_chain_combinations, progress_map, print_results, format_mapping_string
from functools import lru_cache, partial

import sys
import pdb
import itertools
from parallelbar import progress_map


def make_one(name, gt_pdb_file, pred_file, alg_type, ig='tcr', mode='unbound'):
    
    if ig == 'ab':
        code, heavy_chain, light_chain, antigen_chain = name.split('_')
        chains = [heavy_chain, light_chain, antigen_chain]
    else:
        code, heavy_chain, light_chain, antigen_chain, mhc_chain = name.split('_')
        chains = [heavy_chain, light_chain, antigen_chain, mhc_chain]
    if alg_type in ['esmfold']: 
        target_chain_ids = ['A', 'B', 'C', 'D']
    elif alg_type in ['imb']:
        target_chain_ids = ['B', 'A']
    elif alg_type in ['alphafold']:
        target_chain_ids = ['B', 'C', 'D', 'E']
    elif alg_type in ['tcrmodel']:
        target_chain_ids = ['E', 'D', 'C', 'A']
    else:
        target_chain_ids = chains

    if mode == 'unbound':
        native_chain_ids = chains[:2]
        target_chain_ids = target_chain_ids[:2]
    else:
        native_chain_ids = chains[:3]
        target_chain_ids = target_chain_ids[:3]
    # print(f"Process: {pred_file} Native: {gt_pdb_file}")

    if alg_type == 'tcrfold':
        chain_map = {target_chain_ids[i]: target_chain_ids[i] for i in range(len(target_chain_ids))}
    elif alg_type in ['alphafold', 'esmfold', 'imb', 'tcrmodel']:
        chain_map = {native_chain_ids[i]: target_chain_ids[i] for i in range(len(target_chain_ids))}
    else:
        raise ValueError('Unknown algorithm type {}'.format(alg_type))
    gt_dock_pdb = load_PDB(gt_pdb_file, chains=native_chain_ids)
    gt_dock_pdb.id = gt_pdb_file
    pred_dock_pdb = load_PDB(pred_file, chains=target_chain_ids)
    pred_dock_pdb.id = pred_file
    # print(f"chain map: {chain_map}")
    dock_metrics = make_DockQ(gt_dock_pdb, pred_dock_pdb, chain_map, mode, ig, alg_type)
    return dock_metrics


def make_DockQ(native_structure, model_structure, chain_map, mode, ig, alg_type):
    
    # print(f"gt_dock_pdb: {gt_dock_pdb}")
    # print(f"pred_dock_pdb: {pred_dock_pdb}")
    # print(f"chain_map: {chain_map}")
    dockq = run_on_all_native_interfaces(model_structure, native_structure, chain_map=chain_map) 
        # check user-given chains are in the structures

    # initial_mapping = chain_map
    # model_chains = list(chain_map.values())
    # native_chains = list(chain_map.keys())
    # model_chains = [c.id for c in model_structure] if not model_chains else model_chains
    # native_chains = (
    #     [c.id for c in native_structure] if not native_chains else native_chains
    # )

    # if len(model_chains) < 2 or len(native_chains) < 2:
    #     print("Need at least two chains in the two inputs\n")
    #     sys.exit()

    # # permute chains and run on a for loop
    # best_dockq = -1
    # best_result = None
    # best_mapping = None

    # model_chains_to_combo = [
    #     mc for mc in model_chains if mc not in initial_mapping.values()
    # ]
    # native_chains_to_combo = [
    #     nc for nc in native_chains if nc not in initial_mapping.keys()
    # ]

    # chain_clusters, reverse_map = group_chains(
    #     model_structure,
    #     native_structure,
    #     model_chains_to_combo,
    #     native_chains_to_combo,
    #     args.allowed_mismatches,
    # )

    # chain_maps = get_all_chain_maps(
    #     chain_clusters,
    #     initial_mapping,
    #     reverse_map,
    #     model_chains_to_combo,
    #     native_chains_to_combo,
    # )

    # num_chain_combinations = count_chain_combinations(
    #     chain_clusters)
 
    # # copy iterator to use later
    # chain_maps, chain_maps_ = itertools.tee(chain_maps)

    # low_memory = num_chain_combinations > 100
    # run_chain_map = partial(
    #     run_on_all_native_interfaces,
    #     model_structure,
    #     native_structure,
    #     no_align=args.no_align,
    #     capri_peptide=args.capri_peptide,
    #     low_memory=low_memory,
    # )

    # if num_chain_combinations > 1:
    #     cpus = min(num_chain_combinations, args.n_cpu)
    #     chunk_size = min(args.max_chunk, max(1, num_chain_combinations // cpus))

    #     # for large num_chain_combinations it should be possible to divide the chain_maps in chunks
    #     result_this_mappings = progress_map(
    #         run_chain_map,
    #         chain_maps,
    #         total=num_chain_combinations,
    #         n_cpu=cpus,
    #         chunk_size=chunk_size,
    #     )

    #     for chain_map, (result_this_mapping, total_dockq) in zip(
    #         chain_maps_, result_this_mappings
    #     ):

    #         if total_dockq > best_dockq:
    #             best_dockq = total_dockq
    #             best_result = result_this_mapping
    #             best_mapping = chain_map

    #     if low_memory:  # retrieve the full output by rerunning the best chain mapping
    #         best_result, total_dockq = run_on_all_native_interfaces(
    #             model_structure,
    #             native_structure,
    #             best_mapping,
    #             args.no_align,
    #             args.capri_peptide,
    #             low_memory=False,
    #         )

    # else:  # skip multi-threading for single jobs (skip the bar basically)
    #     best_mapping = next(chain_maps)
    #     best_result, best_dockq = run_chain_map(best_mapping)

    # info = dict()
    # info["model"] = model_structure
    # info["native"] = native_structure
    # info["best_dockq"] = best_dockq
    # info["best_result"] = best_result
    # info["GlobalDockQ"] = best_dockq / len(best_result)
    # info["best_mapping"] = best_mapping
    # info["best_mapping_str"] = f"{format_mapping_string(best_mapping)}"
    # results = best_result[tuple(native_chains)]


    # pdb.set_trace()
    native_chain_ids = tuple(chain_map.keys())
    # import pdb
    # pdb.set_trace()
    res = {'DockQ_F1': 0, 'DockQ': 0, 'F1': 0, 'irms': 0, 'Lrms': 0, 'fnat': 0, 'nat_correct': 0, 'nat_total': 0, 'fnonnat': 0, 'nonnat_count': 0, 'model_total': 0, 'clashes': 0}
    
    # results = dockq[0][native_chain_ids]
    for k,v in dockq[0].items():
        for key, value in res.items():
            res[key] = value + v[key]
    for k, v in res.items():
        res[k] = res[k] / len(dockq[0].keys())
    # DockQ, Irms, Lrms, Fnat = results['DockQ'], results['irms'], results['Lrms'], results['fnat']
    # print(f"Finsihed")
    DockQ, Irms, Lrms, Fnat = res['DockQ'], res['irms'], res['Lrms'], res['fnat']
    # pdb.set_trace()
    return {'DockQ':DockQ, 'Irms':Irms, 'Lrms':Lrms, 'Fnat':Fnat}


def main(args):
    names = list(pd.read_csv(args.name_idx, names=['name'], header=None)['name'])
    # random.shuffle(names)
    # names = ['7n2q_G_E_J_H']
    metrics = []
    for i, n in enumerate(names):
        if args.mode == 'unbound':
            gt_file = os.path.join(args.gt_dir, n + '_ab.pdb')
        else:
            gt_file = os.path.join(args.gt_dir, n + '.pdb')
        if args.alg_type != 'alphafold':
            pred_file = os.path.join(args.pred_dir, n + '.pdb')
        else:
            pred_file = os.path.join(args.pred_dir, n + '.cif')
        # print(f"pred_file: {pred_file}")
        if os.path.exists(gt_file) and os.path.exists(pred_file):
            one_metric = OrderedDict({'name' : n})

            dockq_metric = make_one(n, gt_file, pred_file, args.alg_type, args.ig, args.mode)
            # try:
            one_metric.update(dockq_metric)
            metrics.append(one_metric)
            # except:
            #     continue

    columns = metrics[0].keys()
    metrics = zip(*map(lambda x:x.values(), metrics))

    df = pd.DataFrame(dict(zip(columns,metrics)))
    df.to_csv(args.output, sep='\t', index=False)
    df = df.dropna()

    print(f"+++"*20)
    # print(f"Algorithm: {args.alg_type}")
    print('final total', df.shape[0])
    for r in df.columns:
        if r == 'name':
            continue
        value = df[r].values
        mean = np.mean(value)
        std = np.std(value)
        max_ = np.max(value)
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


    # DockQ
    parser.add_argument(
        "--capri_peptide",
        default=False,
        action="store_true",
        help="use version for capri_peptide \
        (DockQ cannot not be trusted for this setting)",
    )
    parser.add_argument(
        "--short", default=True, action="store_true", help="Short output"
    )
    parser.add_argument(
        "--verbose", "-v", default=False, action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--no_align",
        default=False,
        action="store_true",
        help="Do not align native and model using sequence alignments, but use the numbering of residues instead",
    )
    parser.add_argument(
        "--n_cpu",
        default=8,
        type=int,
        metavar="CPU",
        help="Number of cores to use",
    )
    parser.add_argument(
        "--max_chunk",
        default=512,
        type=int,
        metavar="CHUNK",
        help="Maximum size of chunks given to the cores, actual chunksize is min(max_chunk,combos/cpus)",
    )
    parser.add_argument(
        "--optDockQF1",
        default=False,
        action="store_true",
        help="Optimize on DockQ_F1 instead of DockQ",
    )
    parser.add_argument(
        "--allowed_mismatches",
        default=0,
        type=int,
        help="Number of allowed mismatches when mapping model sequence to native sequence.",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        metavar="MODELCHAINS:NATIVECHAINS",
        help="""Specify a chain mapping between model and native structure.
            If the native contains two chains "H" and "L"
            while the model contains two chains "A" and "B",
            and chain A is a model of native chain
            H and chain B is a model of native chain L,
            the flag can be set as: '--mapping AB:HL'.
            This can also help limit the search to specific native interfaces.
            For example, if the native is a tetramer (ABCD) but the user is only interested
            in the interface between chains B and C, the flag can be set as: '--mapping :BC'
            or the equivalent '--mapping *:BC'.""",
    )

    args = parser.parse_args()



    main(args)

