from collections import OrderedDict

import torch

def main():
    in_file = '../carbon_data/trained_models/carbonfold_diffuser_v2/step_0.ckpt'
    in_file = '../abdata_2023/trained_models/stage2/step_58000.ckpt'

    x = torch.load(in_file, map_location='cpu')
    print(x.keys())
    model_state_dict = x['model_state_dict']
    new_model_state_dict = OrderedDict()
    
    lora_keys = {}
    for k in model_state_dict:
        print(k)
        if k.endswith('lora_A'):
            prefix = k.split('lora_A')[0]
            k_lora_B = prefix + 'lora_B'
            k_w = prefix + 'weight'
            lora_keys[k_w] = (k, k_lora_B)
    
    scaling = 1.
    
    for k, v in model_state_dict.items():
            if k.endswith('lora_A') or k.endswith('lora_B'):
                 continue
            if k in lora_keys:
                k_lora_A, k_lora_B = lora_keys[k]
                print(model_state_dict[k].shape, model_state_dict[k_lora_A].shape, model_state_dict[k_lora_B].shape)
                print(model_state_dict[k_lora_A].device, 'device')
                delta = torch.matmul(model_state_dict[k_lora_A], model_state_dict[k_lora_B]) * scaling
                new_model_state_dict[k] = model_state_dict[k] + torch.transpose(delta, 0, 1)
                print(k,
                        torch.sqrt(torch.sum(torch.square(delta))),
                        torch.sqrt(torch.sum(torch.square(model_state_dict[k_lora_A]))),
                        torch.sqrt(torch.sum(torch.square(model_state_dict[k_lora_B]))),
                        torch.sqrt(torch.sum(torch.square(model_state_dict[k]))))
            else:
                new_model_state_dict[k] = v
    
    out_file = in_file.split('.ckpt')[0] + '_lora_merged.ckpt'

    x.update(model_state_dict=new_model_state_dict)
    
    torch.save(x, out_file)

if __name__ == '__main__':
    main()
