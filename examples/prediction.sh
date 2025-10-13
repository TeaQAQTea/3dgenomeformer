#!/bin/bash

python /local1/home/zhongjiaxin/C.Origami/src/corigami/inference/prediction.py \
    --chr $1 \
    --celltype "k562" \
    --start $2  \
    --model $3 \
    --seq "/local1/home/zhongjiaxin/corigami_data/data/hg38/dna_sequence" \
    --ctcf "/local1/home/zhongjiaxin/corigami_data/data/hg38/k562/genomic_features/CTCF_K562_hg38.bigWig.bw" \
    --atac "/local1/home/zhongjiaxin/corigami_data/data/hg38/k562/genomic_features/kas_k562_hg38_final.bigwig"\
    --out "/local1/home/zhongjiaxin/corigami_data/result_hicpro_1d"
    
