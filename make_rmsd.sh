
base=../compare_ab_2023/

name_idx=${base}/S3_test_0.99.idx
metric_dir=${base}/metrics
fasta_dir=${base}/fasta
gt_dir=../sabdab_20230511/npz/

mkdir -p ${metric_dir}

evaluate_abfold() {
    python ./make_rmsd.py  \
        --name_idx ${name_idx} \
        --fasta_dir ${fasta_dir} \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${base}/ab_release1_esm_cle_v3_epoch590 \
        --output ${metric_dir}/abfold_590.csv
}

evaluate_esmfold() {
    pred_dir=${base}/esmfold
    renum_pdb_dir=${base}/renum_esmfold
    mkdir -p ${renum_pdb_dir}

    python ./renum_pred_pdb.py \
        --type ig \
        --name_idx ${name_idx} \
        --fasta_dir ${fasta_dir} \
        --pred_dir ${pred_dir} \
        --output_dir ${renum_pdb_dir}

    python ./make_rmsd.py  \
        --name_idx ${name_idx} \
        --fasta_dir ${fasta_dir} \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${base}/${renum_pdb_dir} \
        --output ${metric_dir}/esmfold.csv
}

evaluate_omegafold() {
    pred_dir=${base}/omegafold
    renum_pdb_dir=${base}/renum_omegafold
    mkdir -p ${renum_pdb_dir}

    python ./renum_pred_pdb.py \
        --type ig \
        --name_idx ${name_idx} \
        --fasta_dir ${fasta_dir} \
        --pred_dir ${pred_dir} \
        --output_dir ${renum_pdb_dir}

    python ./make_rmsd.py  \
        --name_idx ${name_idx} \
        --fasta_dir ${fasta_dir} \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${base}/${renum_pdb_dir} \
        --output ${metric_dir}/omegafold.csv
}

evaluate_igfold() {
    pred_dir=${base}/igfold
    
    python ./make_rmsd.py  \
        --name_idx ${name_idx} \
        --fasta_dir ${fasta_dir} \
        --gt_dir  ${gt_dir} \
        --pred_dir  ${pred_dir} \
        --output ${metric_dir}/igfold.csv
}


 evaluate_abfold
# evaluate_esmfold
# evaluate_omegafold
# evaluate_igfold
