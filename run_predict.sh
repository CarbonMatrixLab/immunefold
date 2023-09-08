base='../abdata_2023'
#    --restore_model_ckpt ../abdata_2023/esm2/esmfold_no_esm2.ckpt \

python predict.py  \
    --gpu_list 0 \
    --restore_model_ckpt ../abdata_2023/esm2/carbonfold_from_esmfold.ckpt \
    --restore_esm2_model ../abdata_2023/esm2/esm2_t36_3B_UR50D.pt \
    --model_features ./config/data_carbonfold_predict.json \
    --name_idx  ${base}/examples/test2.idx \
    --data_dir ../abdata_2023/examples/ \
    --output_dir ./tmp  \
    --mode general
