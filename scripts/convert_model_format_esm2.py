import torch

y = torch.load('../abdata_2023/esm2/esm2_t36_3B_UR50D.pt', map_location='cpu')

x = torch.load('../abdata_2023/trained_models/stage1/step_20000.ckpt', map_location='cpu')
x.update(model=x['model_state_dict'], cfg=y['cfg'])
del x['model_state_dict']

torch.save(x, '../abdata_2023/trained_models/stage1/step_20000_v2.ckpt')

print(x['model']['layers.33.self_attn.k_proj.weight'])

print(y['model']['encoder.sentence_encoder.layers.33.self_attn.k_proj.weight'])

print(torch.sum(torch.square(x['model']['lm_head.dense.weight']-y['model']['encoder.lm_head.dense.weight'])))


