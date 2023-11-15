import os
import sys
import numpy as np
import pandas as pd

from carbonmatrix.data.antibody import antibody_constants
from carbonmatrix.data.antibody.seq import is_in_framework

def get_framework_struc(feature):
    print(antibody_constants.region_type_order)
    N = 130
    
    def _calc_chain_feat(feat, domain_type):
        coords = np.zeros((N, 3))
        coord_mask = np.zeros((N, ), dtype=np.bool_)

        for i, (resn, icode) in enumerate(feat['numbering']):
            if icode != ' ':
                continue
            
            if not is_in_framework(domain_type, (resn, icode)):
                continue

            coords[resn, :] = feat['coords'][i,1]
            coord_mask[resn] = feat['coord_mask'][i, 1]

        return coords, coord_mask

    heavy_coords, heavy_coord_mask = _calc_chain_feat(feature['heavy_chain'], domain_type='H')
    light_coords, light_coord_mask = _calc_chain_feat(feature['light_chain'], domain_type='L')

    return np.concatenate([heavy_coords, light_coords], axis=0), np.concatenate([heavy_coord_mask, light_coord_mask], axis=0)
    
def main():
    summary_file = '~/neo/carbon/antibody/sabdab/20231102/sabdab_summary_all.tsv'
    meta_file = '/home/zhanghaicang/neo/carbon/antibody/sabdab/20231102/all.meta.tsv'

    npy_dir = '/home/zhanghaicang/neo/carbon/antibody/sabdab/20231102/npy' 
    
    df = pd.read_csv(meta_file, sep='\t')
    df = df[df['name'] == '1hzh_H_L']
    print(df.shape[0])

    df = df[(df['resolution'] > 0.1) & (df['resolution'] < 3.0)]
    print(df.shape[0])

    df = df[~df['light_seq'].isna()]
    print(df.shape[0])

    fr_coors = []
    fr_coord_mask = []

    for n, g in df.groupby(by='code'):
        r = g.iloc[0,:]
        x = np.load(os.path.join(npy_dir, r['name'] + '.npy'), allow_pickle=True).item()

        struc = get_framework_struc(x)
        print(struc)

        break

if __name__ == '__main__':
    main()
    
