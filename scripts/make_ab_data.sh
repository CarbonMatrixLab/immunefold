#!/bin/bash

base='../sabdab_20230511/'

#python make_ab_data_from_mmcif.py \
#    --summary_file ../sabdab_20230511/sabdab_summary_all.tsv  \
#    --mmcif_dir ../sabdab_20230511/mmcif \
#    --output_dir ../sabdab_20230511/npz/
#
out_dir=${base}/list
mkdir -p ${out_dir}
python make_ab_data_filter_list.py \
    --data_dir ${base}/npz \
    --out_dir ${out_dir} \
    --summary_file ${base}/sabdab_summary_all.tsv

data_dir=${out_dir}
out_dir=${data_dir}/mmseqs/
tmp_dir=${data_dir}/tmp
rm -rf ${tmp_dir}
mkdir -p ${out_dir}
seq_id=0.99
#mmseqs easy-cluster \
#    --min-seq-id ${seq_id} \
#    --cov-mode 2 \
#    ${data_dir}/S1_train_concat.fasta ${out_dir}/train ${tmp_dir}
#
#python make_ab_data_prepare_cluster.py ${out_dir}/train_cluster.tsv ${base}/list/S2_train_cluster${seq_id}.idx 

#mmseqs easy-search --min-seq-id 0.99 \
#    ${data_dir}/S1_test_concat.fasta ${data_dir}/S1_train_concat.fasta ${out_dir}/test_against_train_0.99 ${tmp_dir}
#cut -f 1 ${out_dir}/test_against_train_0.99 | sort | uniq  > ${out_dir}/test_against_train_0.99_overlap
#python scripts/filter_fasta_overlap.py ${data_dir}/S1_test_concat.fasta ${data_dir}/S2_test_concat.fasta ${out_dir}/test_against_train_0.99_overlap

seq_id=0.99
#mmseqs easy-cluster \
#    --min-seq-id ${seq_id} \
#    --cov-mode 2 \
#    ${data_dir}/S2_test_concat.fasta ${out_dir}/S2_test_clust_${seq_id} ${tmp_dir}

python scripts/format_test_list.py ${data_dir}/S1_summary_filter.csv ${out_dir}/S2_test_clust_${seq_id}_cluster.tsv ${data_dir}/S3_test_0.99.idx
