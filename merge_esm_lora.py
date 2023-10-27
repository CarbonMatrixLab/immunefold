import os
import json
import torch
import argparse
from collections import OrderedDict

from ml_collections.config_dict import ConfigDict
import torch

def main(args):
    # only works for esm2
    cfg = {
         'model': {
              "encoder_layers" : 36,
              "encoder_embed_dim": 2560,  
              "encoder_attention_heads": 40,
              "token_dropout" : True
         }
    }
    
    cfg = ConfigDict(cfg)
    
    scaling = args.scaling
    
    data = torch.load(args.input_our_esm_ckpt, map_location='cpu')

    model_state_dict = data['model_state_dict']
    merged_model_state_dict = OrderedDict()
    
    lora_keys = {}
    for k in model_state_dict:
        if k.endswith('lora_A'):
            prefix = k.split('lora_A')[0]
            k_lora_B = prefix + 'lora_B'
            k_w = prefix + 'weight'
            lora_keys[k_w] = (k, k_lora_B)
    
    
    for k, v in model_state_dict.items():
            if k.endswith('lora_A') or k.endswith('lora_B'):
                 continue
            if k in lora_keys:
                k_lora_A, k_lora_B = lora_keys[k]
                print(model_state_dict[k].shape, model_state_dict[k_lora_A].shape, model_state_dict[k_lora_B].shape)
                print(model_state_dict[k_lora_A].device, 'device')
                delta = torch.matmul(model_state_dict[k_lora_A], model_state_dict[k_lora_B]).transpose(1,0) * scaling
                merged_model_state_dict[k] = model_state_dict[k] + delta
                print(k, torch.sqrt(torch.sum(torch.square(delta))), torch.sqrt(torch.sum(torch.square(model_state_dict[k]))))
            else:
                merged_model_state_dict[k] = v
    
    out_file = args.input_our_esm_ckpt.split('.ckpt')[0] + '_lora_merged.ckpt'

    torch.save(dict(model=merged_model_state_dict, cfg=cfg), out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_our_esm_ckpt', type=str, required=True)
    parser.add_argument('--scaling', type=float, default=1.0)
    args = parser.parse_args()

    main(args)