base='../abdata_2023'

python predict.py  \
    --model ../abdata_2023/esm2/abfold_from_esmfold.ckpt \
    --model_features ./config/config_data_pair.json \
    --name_idx  ${base}/examples/test.idx
    --data_dir ../abdata_2023/examples/ \
    --output_dir ./tmp  \
    --mode general
