scripts_dir=/home/zhanghaicang/neo/hh-suite/scripts
sto_dir=/home/zhanghaicang/neo/Tranception/ProteinGym_substitution_sto_files

reformat=${scripts_dir}/reformat.pl

parse_a3m='python ./scripts/bayesmvp/preprocess_a3m.py'

process_one(){
    name=$1
    sto_file=${sto_dir}/${name}.sto
    a3m_file=${sto_dir}/${name}.a3m

    # ${reformat} sto a3m ${sto_file} ${a3m_file}
    # hhfilter -maxseq 1000000 -cov 75 -id 99 -i ${a3m_file} -o ${sto_dir}/${name}.filterd.a3m 
    ${parse_a3m} --in_a3m_file ${sto_dir}/${name}.filterd.a3m --out_fasta_file ${sto_dir}/${name}.filterd.fasta
    
}

name_list=/home/zhanghaicang/neo/Tranception/msa.list

for i in `cat ${name_list}`
do
    process_one $i
done
