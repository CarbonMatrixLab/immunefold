import os
import argparse
import functools
import multiprocessing as mp
import logging
import pandas as pd
import traceback
from Bio.PDB.PDBExceptions import PDBConstructionException

from carbonmatrix.data.mmcif_parsing import parse as mmcif_parse

from Bio import PDB
import shutil
import pdb
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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
    
    
    def _parse_chain_id(heavy_chain_id, light_chain_id):
        if heavy_chain_id.islower() and heavy_chain_id.upper() == light_chain_id:
            heavy_chain_id = heavy_chain_id.upper()
        elif light_chain_id.islower() and light_chain_id.upper() == heavy_chain_id:
            light_chain_id = light_chain_id.upper()
        return heavy_chain_id, light_chain_id

    for orig_heavy_chain_id, orig_light_chain_id, antigen_chain_id, mhc_chain_id in chain_ids:
        # antigen_chain_ids = orig_antigen_chain_id.split('|')
        # antigen_chain_ids = [s.replace(" ", "") for s in antigen_chain_ids]
        
        # heavy_chain_id, light_chain_id = _parse_chain_id(orig_heavy_chain_id, orig_light_chain_id)
        heavy_chain_id, light_chain_id = orig_heavy_chain_id, orig_light_chain_id
        if (heavy_chain_id and heavy_chain_id not in parsing_result.mmcif_object.chain_to_seqres):
            logging.warning(f'{code} {heavy_chain_id} {light_chain_id}: chain ids not exist.')
            continue
        
        
        flag = 0
        # for antigen_chain_id in antigen_chain_ids:
        #     if antigen_chain_id not in parsing_result.mmcif_object.chain_to_seqres:
        #         logging.warning(f"antigen id: {antigen_chain_id} not exist")
        #         flag += 1
        #         continue
        # if flag > 0:
        #     continue
        chain_ids = [heavy_chain_id, light_chain_id] + [antigen_chain_id, mhc_chain_id]
        # pdb.set_trace()
        try:
            split_antibody_antigen_complex(mmcif_file, code, chain_ids, args.output_dir)
            # save_feature(feature, code, orig_heavy_chain_id, orig_light_chain_id, antigen_chain_ids, args.output_dir)
            logging.info(f'succeed: {mmcif_file} {orig_heavy_chain_id} {orig_light_chain_id}')
        except Exception as e:
            traceback.print_exc()
            logging.error(f'make structure: {mmcif_file} {orig_heavy_chain_id} {orig_light_chain_id} {str(e)}')



def get_multiple_chains_pdb_from_complex(input_pdb, pdb_name, chain_ids, output_pdb):
    # Create a PDB parser object
    parser = PDB.MMCIFParser()
    chain_ids = [c for c in chain_ids if c != '']
    try:
        structure = parser.get_structure(pdb_name, input_pdb)
    except PDBConstructionException as e:
        logging.warning('mmcif_parse: %s {%s}', input_pdb, str(e))
    except Exception as e:
        logging.warning('mmcif_parse: %s {%s}', input_pdb, str(e))
        raise Exception('...') from e
    # Create a PDB io object
    io = PDB.PDBIO()
    # Set the structure object to the io object
    io.set_structure(structure)
    # Create an empty list to store the selected chains
    selected_chains = []
    # Loop through the chain ids
    for chain_id in chain_ids:
        chain = structure[0][chain_id]
        selected_chains.append(chain)
    # pdb.set_trace()
    # Create a select class that only accepts the selected chains
    class Select(PDB.Select):
        def accept_chain(self, chain_select):
            if chain_select in selected_chains:
                return True
            else:
                return False

    # Save the selected chains to the output pdb file using the select class
    io.save(output_pdb, Select())


def get_single_chain_pdb_from_complex(input_pdb, pdb_name, chain_id, output_pdb):
    parser = PDB.MMCIFParser()
    try:
        structure = parser.get_structure(pdb_name, input_pdb)
    except PDBConstructionException as e:
        logging.warning('mmcif_parse: %s {%s}', input_pdb, str(e))
    except Exception as e:
        logging.warning('mmcif_parse: %s {%s}', input_pdb, str(e))
        raise Exception('...') from e
    chain = None

    for model in structure:
        for c in model:
            if c.get_id() == chain_id:
                chain = c
                break
        if chain is not None:
            break
    if chain is None:
        raise Exception("Error: could not find chain with name", chain_id)

    io = PDB.PDBIO()

    class ChainSelect(PDB.Select):
        def accept_chain(self, c):
            return c == chain

    io.set_structure(structure)
    io.save(output_pdb, ChainSelect())


def split_antibody_antigen_complex(input_pdb_path, pdb_name, chain_ids, out_dir):
    """
    split complex:
    1. save complex pdb.
    2. save antibody heavy chain pdb, light chain pdb and heavy - light chain complex pdb.
    3. save antigen chain pdb.
    all pdbs are saved into different folder.
    :param input_pdb_path: input complex pdb file
    :param pdb_name: as its name.
    :param chain_ids: Attention! the order is heavy light antigen.
    :param out_dir: output dir, function will make subdir in it.
    """
    make_dir(out_dir)
    antigen_chain_id, mhc_chain_id = chain_ids[2:]
    pdb_out_name = f'{pdb_name}_{chain_ids[0]}_{chain_ids[1]}_{antigen_chain_id}_{mhc_chain_id}'
    pdb_out_dir = os.path.join(out_dir, pdb_out_name)
    make_dir(pdb_out_dir)
    # input
    pdb_path = os.path.join(pdb_out_dir, f"{pdb_name}.cif")
    shutil.copy(input_pdb_path, pdb_path)

    # complex
    tcr_pMHC_path = os.path.join(pdb_out_dir, f"{pdb_name}_{'_'.join(chain_ids)}.pdb")
    ab_path = os.path.join(pdb_out_dir, f"{pdb_name}_{'_'.join(chain_ids[:2])}.pdb")
    # pdb.set_trace()
    # single chain
    # h_path = os.path.join(pdb_out_dir, f"{pdb_name}_{chain_ids[0]}_hc.pdb")
    # l_path = os.path.join(pdb_out_dir, f"{pdb_name}_{chain_ids[1]}_lc.pdb")
    # ag_path = os.path.join(pdb_out_dir, f"{pdb_name}_{''.join(chain_ids[2:])}_ag.pdb")

    get_multiple_chains_pdb_from_complex(pdb_path, pdb_name, chain_ids, tcr_pMHC_path)
    get_multiple_chains_pdb_from_complex(pdb_path, pdb_name, chain_ids[:2], ab_path)
    # get_multiple_chains_pdb_from_complex(pdb_path, pdb_name, chain_ids[2:], ag_path)

    # get_single_chain_pdb_from_complex(pdb_path, pdb_name, chain_ids[0], h_path)
    # get_single_chain_pdb_from_complex(pdb_path, pdb_name, chain_ids[1], l_path)


def main(args):
    func = functools.partial(process, args=args)
    
    with mp.Pool(args.cpus) as p:
        p.starmap(func, parse_list(args.summary_file))
    # for code, chain_ids in parse_list(args.summary_file):
    #     process(code, chain_ids, args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--cpus', type=int, default=50)
  parser.add_argument('--summary_file', type=str, required=True)
  parser.add_argument('--mmcif_dir', type=str, required=True)
  parser.add_argument('--output_dir', type=str, required=True)
  parser.add_argument('--verbose', action='store_true', help='verbose')
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  main(args)

