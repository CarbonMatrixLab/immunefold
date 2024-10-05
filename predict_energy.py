import os
import argparse
import functools
import multiprocessing as mp
import logging
import pandas as pd
import csv
from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.teaching import *
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V'
}

init('-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res \
    -ignore_zero_occupancy false -load_PDB_components true -relax:default_repeats 2 -no_fconfig', silent=True)

def parse_energy_log(energy_log_path):
    fields = ['Energy']
    data = list()
    def _parse_line(line):
        flags = line.split(':')
        if len(flags) >= 2 and any(map(lambda x: flags[-2].endswith(x), fields)):
            k = (flags[-2].split('- ')[-1]).split('@')[0]
            v = float(flags[-1].strip())
            item = {'PDB': k, 'Energy': v}
            data.append(item)
    with open(energy_log_path) as f:
        for line in f:
            _parse_line(line)
    return data

def parse_list(name_idx, pdb_dir):
    names = list(pd.read_csv(name_idx, names=['name'], header=None)['name'])
    for name in names:
        pdb_file = f"{pdb_dir}/{name}.pdb"
        yield pdb_file


def relax_pose(pose, pdb_path):
    """Performs a full-atom relax on the input pose."""
    scorefxn = pyrosetta.create_score_function('ref2015')
    relax = rosetta.protocols.relax.FastRelax(scorefxn)
    relax.apply(pose)
    pdb_dir = '/'.join(pdb_path.split('/')[:-1])
    pdb_name = pdb_path.split('/')[-1].split('.pdb')[0]
    pose.dump_pdb(f'{pdb_dir}/{pdb_name}_relaxed.pdb')

def combine_chains(pose, chains_indices):
    """Combine specified chains into a single pose."""
    combined_pose = rosetta.core.pose.Pose()
    for i, chain_index in enumerate(chains_indices):
        chain_pose = pose.split_by_chain()[chain_index]
        if i == 0:
            combined_pose = chain_pose
        else:
            combined_pose.append_pose_by_jump(chain_pose, combined_pose.total_residue())
    return combined_pose


def pyrosetta_interface_energy(pdb_path, chain_ids):
    logger.info(f"Calculate Rosetta Interface Energy for {pdb_path}")
    pose = pyrosetta.pose_from_pdb(pdb_path)
    tcr_chain_ids = chain_ids[:2]
    antigen_chain_ids = chain_ids[2:]
    interface_tcr = ''.join(tcr_chain_ids)
    interface_antigen = ''.join(antigen_chain_ids)
    interface = interface_tcr+'_'+interface_antigen
    relax_pose(pose, pdb_path)

    mover = InterfaceAnalyzerMover()
    mover.set_interface(interface)
    mover.set_scorefunction(pyrosetta.create_score_function('ref2015'))
    mover.apply(pose)
    return pose.scores['dG_separated']

def pyrosetta_whole_energy(pdb_file, chain_ids):
    logger.info(f"Calculate Rosetta Whole Energy for {pdb_file}")
    peptide_chain = 3
    if len(chain_ids) == 4:
        TCR_MHC_chain = [1,2,4]
    elif len(chain_ids) == 3:
        TCR_MHC_chain = [1,2]
    pose = pyrosetta.pose_from_file(pdb_file)
    relax_pose(pose, pdb_file)

    chains = pose.split_by_chain()
    peptide_pose = chains[peptide_chain]
    TCR_MHC_pose = combine_chains(pose, TCR_MHC_chain)
    
    combined_pose = rosetta.core.pose.Pose(peptide_pose)
    combined_pose.append_pose_by_jump(TCR_MHC_pose, combined_pose.total_residue())
    scorefxn = rosetta.core.scoring.get_score_function()
    total_energy = scorefxn(combined_pose)
    energy_chain_C = scorefxn(peptide_pose)
    energy_chains_ABD = scorefxn(TCR_MHC_pose)
    binding_energy = total_energy - (energy_chain_C + energy_chains_ABD)
    return binding_energy



def process(pdb_file, args):
    pdb_name = pdb_file.split('/')[-1].split('.pdb')[0]
    chain_ids = pdb_name.split('_')[1:]
    if args.mode == 'interface':
        energy = pyrosetta_interface_energy(pdb_file, chain_ids)
    elif args.mode == 'whole':
        energy = pyrosetta_whole_energy(pdb_file, chain_ids)
    logger.info(f"{pdb_name}@Energy: {energy}")


def main(args):
    func = functools.partial(process, args=args)

    with mp.Pool(args.cpus) as p:
        p.starmap(process, [(item, args) for item in parse_list(args.name_idx, args.pdb_dir)])

def data_process(log_file, args):
    energy_data = parse_energy_log(log_file)

    with open(args.output_file, 'w', newline='') as file:
        fieldsname = ['PDB', 'Energy']
        writer = csv.DictWriter(file, fieldnames=fieldsname)
        writer.writeheader()  
        for row in energy_data:
            writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pdb_dir', type=str, required=True)
    parser.add_argument('-n', '--name_idx', type=str, required=True)
    parser.add_argument('-o', '--output_file', type=str, required=True)
    parser.add_argument('-c', '--cpus', type=int, default=1)
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['whole', 'interface'])
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    log_file = os.path.join(args.pdb_dir,f'energy_calculation_{args.mode}.log')
    handler_test = logging.FileHandler(log_file) 
    handler_control = logging.StreamHandler()    

    selfdef_fmt = '%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(selfdef_fmt)
    handler_test.setFormatter(formatter)
    handler_control.setFormatter(formatter)
    logger.setLevel('DEBUG')          
    logger.addHandler(handler_test)    
    logger.addHandler(handler_control)
    main(args)
    data_process(log_file, args)