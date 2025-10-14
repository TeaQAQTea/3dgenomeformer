#!/bin/bash

python /local2/zjx/corigami/code/C.Origami/src/corigami/inference/prediction.py \
    --chr $1 \
    --celltype "k562" \
    --start $2  \
    --model $3 \
    --seq "/local2/zjx/corigami/data/data/hg38/dna_sequence" \
    --ctcf "/local2/zjx/corigami/data/data/hg38/k562/genomic_features/k562_ctcf_sns.bw" \
    --atac "/local2/zjx/corigami/data/data/hg38/k562/genomic_features/kas_k562_hg38_ip.bigwig"\
    --out "/local2/zjx/corigami/data/data/result_hicpro_1d"
    
