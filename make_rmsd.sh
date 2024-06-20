source /home/zhutian/anaconda3/etc/profile.d/conda.sh
conda activate esm_flash_attn

base=../compare_ab_2023/

name_idx=/home/zhutian/data/tcr_data/idx/test.idx
metric_dir=${base}/metrics
fasta_dir=/home/zhutian/data/tcr_data/test.fasta
gt_dir=/home/zhutian/data/tcr_data/npz
gt_pdb=/home/zhutian/data/tcr_data/test_pdb
# tcrfold=/home/zhutian/Git_repo/carbonmatrix/pred/tcrfold_20000
# tcrfold=/home/zhutian/Git_repo/carbonmatrix/pred/tcrfold_peptide
tcrfold=/home/zhutian/Git_repo/carbonmatrix/pred/tcrfold_25000

imb=/home/zhutian/Git_repo/ImmuneBuilder/tcr_pred
esmfold=/home/zhutian/Git_repo/esm/ESMFold_tcr
omegafold=/home/zhutian/Git_repo/OmegaFold/OmegaFold_model1_tcr
alphafold=/home/zhutian/Git_repo/carbonmatrix/pred/afm
alphafold3=/home/zhutian/Git_repo/carbonmatrix/pred/af3
tcrmodel=/home/zhutian/Git_repo/carbonmatrix/pred/tcrmodel2_new_model1
# tcrmodel=/home/zhutian/Git_repo/carbonmatrix/pred/tcrmodel2_pMHC

output_dir=/home/zhutian/Git_repo/carbonmatrix/pred

mkdir -p ${output_dir}

evaluate_abfold() {
    echo '*******************************************************************'
    echo 'ImmuneFold'
    python ./make_rmsd.py  \
        --name_idx ${name_idx} \
        --alg_type tcrfold \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${tcrfold} \
        --output ${output_dir}/ImmuneFold_25000_tcr.csv \
        --pdb_dir ${gt_pdb} \
        --ig tcr \
        --mode unbound
    echo ''
}

evaluate_tcrmodel() {
    echo '*******************************************************************'
    echo 'TCRmodel2'
    python ./make_rmsd.py  \
        --name_idx ${name_idx} \
        --alg_type tcrmodel \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${tcrmodel} \
        --output ${output_dir}/tcrmodel_tcr.csv \
        --pdb_dir ${gt_pdb} \
        --ig tcr \
        --mode unbound
    echo ''
}


evaluate_alphafold3() {
    python ./make_rmsd.py  \
        --name_idx ${name_idx} \
        --alg_type alphafold \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${alphafold3} \
        --output ${output_dir}/af3.csv \
        --pdb_dir ${gt_pdb} \
        --ig tcr \
        --mode unbound
    echo ''
}


evaluate_imb() {
    echo '*******************************************************************'
    echo 'ImmuneBuilder'
    python ./make_rmsd.py  \
        --name_idx ${name_idx} \
        --alg_type imb \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${imb} \
        --output ${output_dir}/immunebuilder_tcr.csv \
        --pdb_dir ${gt_pdb} \
        --ig tcr \
        --mode unbound
    echo ''
}

evaluate_esmfold() {
    echo '*******************************************************************'
    echo 'ESMFold'
    python ./make_rmsd.py  \
        --name_idx ${name_idx} \
        --alg_type esmfold \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${esmfold} \
        --output ${output_dir}/esmfold_tcr.csv \
        --pdb_dir ${gt_pdb} \
        --ig tcr \
        --mode unbound
    echo ''
}

evaluate_omegafold() {
    echo '*******************************************************************'
    echo 'OmegaFold'
    python ./make_rmsd.py  \
        --name_idx ${name_idx} \
        --alg_type omegafold \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${omegafold} \
        --output ${output_dir}/omegafold_tcr.csv \
        --pdb_dir ${gt_pdb} \
        --ig tcr \
        --mode unbound
}

evaluate_igfold() {
    pred_dir=${base}/igfold
    
    python ./make_rmsd.py  \
        --name_idx ${name_idx} \
        --fasta_dir ${fasta_dir} \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${pred_dir} \
        --output ${metric_dir}/igfold_tcr.csv
}
evaluate_alphafold() {
    python ./make_rmsd.py  \
        --name_idx ${name_idx} \
        --alg_type alphafold \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${alphafold} \
        --output ${output_dir}/alphafold_tcr.csv \
        --pdb_dir ${gt_pdb} \
        --ig tcr \
        --mode unbound
}


evaluate_abfold
# evaluate_alphafold3

# evaluate_imb
# evaluate_esmfold
# evaluate_omegafold
# evaluate_alphafold
# evaluate_igfold
evaluate_tcrmodel