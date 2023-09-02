base='../abdata_2023'

python predict.py  \
    --restore_model_ckpt ../abdata_2023/esm2/esmfold_no_esm2.ckpt \
    --restore_esm2_model ../abdata_2023/esm2/esm2_t36_3B_UR50D.pt \
    --model_features ./config/config_data_stage1.json \
    --restore_model_ckpt ../abdata_2023/esm2/esmfold_no_esm.ckpt \
    --name_idx  ${base}/examples/test.idx
    --data_dir ../abdata_2023/examples/ \
    --output_dir ./tmp  \
    --mode general
