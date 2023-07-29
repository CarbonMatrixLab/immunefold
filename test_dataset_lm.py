import json
import argparse

import torch

from abfold.trainer import dataset_lm as dataset

def main(args):
    
    device = torch.device('cuda:0')

    with open(args.model_features, 'r', encoding='utf-8') as f:
        feats = json.load(f)
        for i in range(len(feats)):
            feat_name, feat_args = feats[i]
            if 'device' in feat_args and feat_args['device'] == '%(device)s':
                feat_args['device'] = device
                feats[i] = (feat_name, feat_args)
    

    train_loader = dataset.load(
            fasta_file = args.fasta_file,
            feats = feats,
            batch_size = 3)

    for data in train_loader:
        print(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_file', type=str, required=True)
    parser.add_argument('--model_features', type=str, required=True)
    args = parser.parse_args()
    main(args)
