base=../abdata_2023/examples
model_file=../abdata_2023/esm2/abfold_from_esmfold.ckpt

python predict.py  \
    --model ${model_file} \
    --model_features ./config/config_data_pair.json \
    --name_idx  ${base}/test.idx \
    --data_dir  ${base}/npz \
    --output_dir ${base}/pred  \
    --mode general

for i in `cat ${base}/test.idx`
do
    echo ${i}
    TMalign ${base}/pred/$i.pdb ${base}/esmfold/$i.pdb
    cat ${base}/pred/$i.pdb | grep -w NH1 | head -n 1
    cat ${base}/esmfold/$i.pdb | grep -w NH1 | head -n 1
done
