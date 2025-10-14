#!/bin/bash

python /local1/home/zhongjiaxin/C.Origami/src/corigami/inference/screening.py \
    --chr "chr2" \
    --celltype "k562" \
    --model "/local1/home/zhongjiaxin/corigami_data/result/k562/models/epoch=26-step=32129-v1.ckpt" \
    --seq "/local1/home/zhongjiaxin/corigami_data/data/hg38/dna_sequence" \
    --ctcf "/local1/home/zhongjiaxin/corigami_data/data/hg38/k562/genomic_features/K562_rep1_peak_hg38.bw" \
    --atac "/local1/home/zhongjiaxin/corigami_data/data/hg38/k562/genomic_features/K562_KAS-IP.rep1.ext150.hg38.bw"\
    --screen-start 1250000 \
    --screen-end 3250000 \
    --perturb-width 1000 \
    --step-size 1000 \
    --plot-impact-score \
    --save-pred --save-perturbation --save-diff --save-bedgraph

