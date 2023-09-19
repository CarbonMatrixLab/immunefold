from collections import OrderedDict

import torch

in_file = '../abdata_2023/trained_models/stage1/step_20000.ckpt'
orig_file = '../abdata_2023/esm2/esm2_t36_3B_UR50D.pt'

x = torch.load(in_file, map_location='cpu')
orig_x = torch.load(orig_file, map_location='cpu')

print(x.keys())

model_state_dict = x['model_state_dict']
new_model_state_dict = OrderedDict()

lora_keys = {}
for k in model_state_dict:
    if k.endswith('lora_A'):
        prefix = k.split('lora_A')[0]
        k_lora_B = prefix + 'lora_B'
        k_w = prefix + 'weight'
        lora_keys[k_w] = (k, k_lora_B)

lora_r = 32

scaling = 1. / lora_r

for k, v in model_state_dict.items():
        if k.endswith('lora_A') or k.endswith('lora_B'):
             continue
        
        if k in lora_keys:
            k_lora_A, k_lora_B = lora_keys[k]
            print(model_state_dict[k].shape, model_state_dict[k_lora_A].shape, model_state_dict[k_lora_B].shape)
            print(model_state_dict[k_lora_A].device, 'device')
            new_model_state_dict[k] = model_state_dict[k] + torch.matmul(model_state_dict[k_lora_B], model_state_dict[k_lora_A]) * scaling
        else:
            new_model_state_dict[k] = v

out_file = in_file.split('.ckpt')[0] + '_lora_merged.ckpt'

torch.save(dict(model=new_model_state_dict, cfg=orig_x['cfg']), out_file)
