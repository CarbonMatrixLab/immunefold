source /home/zhutian/anaconda3/etc/profile.d/conda.sh
conda activate diffab

base=../compare_ab_2023/

name_idx=/home/zhutian/data/tcr_data/test_tcr.idx
metric_dir=${base}/metrics
fasta_dir=/home/zhutian/data/tcr_data/test.fasta
gt_dir=/home/zhutian/Git_repo/carbonmatrix/pred/gt_tcr
tcrfold=/home/zhutian/Git_repo/carbonmatrix/pred/tcrfold_3000
imb=/home/zhutian/Git_repo/ImmuneBuilder/tcr_pred
esmfold=/home/zhutian/Git_repo/esm/ESMFold_tcr
omegafold=/home/zhutian/Git_repo/OmegaFold/OmegaFold_model1_tcr
alphafold=/home/zhutian/Git_repo/carbonmatrix/pred/afm
alphafold3=/home/zhutian/Git_repo/carbonmatrix/pred/af3

output_dir=/home/zhutian/Git_repo/carbonmatrix/pred

mkdir -p ${output_dir}

evaluate_abfold() {
    python ./make_dockq.py  \
        --name_idx ${name_idx} \
        --alg_type tcrfold \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${tcrfold} \
        --output ${output_dir}/tcrfold_3000_dockq.csv \
        --ig tcr \
        --mode unbound
}

evaluate_alphafold3() {
    python ./make_dockq.py  \
        --name_idx ${name_idx} \
        --alg_type alphafold \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${alphafold3} \
        --output ${output_dir}/af3_dockq.csv \
        --ig tcr \
        --mode unbound
}


evaluate_imb() {
    python ./make_dockq.py  \
        --name_idx ${name_idx} \
        --alg_type imb \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${imb} \
        --output ${output_dir}/tcr_esm_dockq.csv \
        --ig tcr \
        --mode unbound
}

evaluate_esmfold() {
    python ./make_dockq.py  \
        --name_idx ${name_idx} \
        --alg_type esmfold \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${esmfold} \
        --output ${output_dir}/esmfold_dockq.csv \
        --ig tcr \
        --mode unbound
}

evaluate_omegafold() {
    python ./make_dockq.py  \
        --name_idx ${name_idx} \
        --alg_type omegafold \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${omegafold} \
        --output ${output_dir}/omegafold_dockq.csv \
        --ig tcr \
        --mode unbound
}

evaluate_igfold() {
    pred_dir=${base}/igfold
    
    python ./make_dockq.py  \
        --name_idx ${name_idx} \
        --fasta_dir ${fasta_dir} \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${pred_dir} \
        --output ${metric_dir}/igfold_dockq.csv
}
evaluate_alphafold() {
    python ./make_dockq.py  \
        --name_idx ${name_idx} \
        --alg_type alphafold \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${alphafold} \
        --output ${output_dir}/alphafold_dockq.csv \
        --ig tcr \
        --mode unbound
}


# evaluate_abfold
# evaluate_alphafold3

# evaluate_imb
# evaluate_esmfold
# evaluate_omegafold
evaluate_alphafold
# evaluate_igfold
