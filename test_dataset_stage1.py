import json
import argparse

import torch
import pandas as pd

from abfold.trainer import dataset_stage as dataset
from abfold.model.lm.pretrained import load_model_and_alphabet_local

def main(args):
    
    device = torch.device('cuda:0')

    with open(args.model_features, 'r', encoding='utf-8') as f:
        feats = json.load(f)
        for i in range(len(feats)):
            feat_name, feat_args = feats[i]
            if 'device' in feat_args and feat_args['device'] == '%(device)s':
                feat_args['device'] = device
                feats[i] = (feat_name, feat_args)
    
    name_idx = list(pd.read_csv(args.name_idx, header=None, names=['name'])['name'])[:2]
    print(name_idx[-1])

    train_loader = dataset.load(
            data_dir = args.data_dir,
            name_idx = name_idx,
            feats = feats,
            batch_size = 2)
    
    if args.restore_model_ckpt is not None:
        ckpt = torch.load(args.restore_model_ckpt)
        model, _, esm_cfg = load_model_and_alphabet_local(args.restore_model_ckpt)
        model.to(device)
        

    for batch in train_loader:
        for k in batch:
            print(k)
        r = model(tokens = batch['seq'], index = batch['residx'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--name_idx', type=str, required=True)
    parser.add_argument('--restore_model_ckpt', type=str, required=True)
    parser.add_argument('--model_features', type=str, required=True)
    args = parser.parse_args()
    main(args)
