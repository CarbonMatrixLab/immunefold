import hydra

import torch
from torch.utils.data.distributed import DistributedSampler

from carbonmatrix.trainer.base_dataset import collate_fn_struc
from carbonmatrix.trainer.dataset import StructureDatasetNpzIO as StructureDataset 
from carbonmatrix.trainer import utils
from carbonmatrix.data.base_dataset import  TransformedDataLoader
    
def test_dataset(cfg):
    utils.setup_ddp()
    world_rank = utils.get_world_rank()
    local_rank = utils.get_local_rank()
    device = utils.get_device()

    dataset = StructureDataset(cfg.train_data, cfg.train_name_idx, cfg.max_seq_len)
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)

    dataloader = TransformedDataLoader(
            dataset, feats=cfg.transforms, device=2,
            collate_fn = collate_fn_struc,
            sampler=sampler,
            batch_size=2,
            )

    for epoch in range(2):
        dataloader.set_epoch(epoch) 
        for i, x in enumerate(dataloader):
            print('batch', local_rank, x['name'])
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    print(k, v.shape)
            if i == 2:
                break

@hydra.main(version_base=None, config_path="config", config_name="test")
def main(cfg):
    test_dataset(cfg)

if __name__ == '__main__':
    main()
