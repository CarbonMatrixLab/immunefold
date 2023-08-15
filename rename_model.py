import torch
from collections import OrderedDict

x = torch.load('../abdata_2023/esm2/abfold_from_esmfold.ckpt')

print(x.keys())
for k, v in x.items():
    print(k, type(v))


state_dict = OrderedDict()

for k, v in x['model_state_dict'].items():
    n = k
    if k.startswith('impl.seqformer'):
        n = 'seqformer_module' + k[14:] 
    elif k.startswith('impl.'):
        n = k[5:]
        print(n)
    state_dict[n] = v 

x.update(model_state_dict=state_dict)

torch.save(x, '../abdata_2023/esm2/esmfold_no_esm2.ckpt')
