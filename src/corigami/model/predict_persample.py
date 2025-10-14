import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

import inference.utils.inference_utils as infer
from inference.utils import plot_utils

WINDOW = 2097152  # 固定窗口大小，与你注释一致


def read_bed(bed_path: str) -> List[Tuple[str, int, int]]:
    regions = []
    with open(bed_path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith(("#", "track", "browser")):
                continue
            toks = ln.split()
            if len(toks) < 3:
                continue
            chrom, start, end = toks[0], int(toks[1]), int(toks[2])
            regions.append((chrom, start, end))
    if not regions:
        raise ValueError(f"No valid regions in BED: {bed_path}")
    return regions


def safe_mkdir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def predict_one(output_path: str,
                celltype: Optional[str],
                chr_name: str,
                start: int,
                model_path: str,
                seq_path: str,
                ctcf_path: str,
                atac_path: str,
                do_plot: bool = True) -> np.ndarray:
    """
    与你原 single_prediction 等价，但可选择不画图，仅返回矩阵。
    """
    # 加载区域数据（utils 内部负责切片/标准化）
    seq_region, ctcf_region, atac_region = infer.load_region(
        chr_name, start, seq_path, ctcf_path, atac_path
    )
    # 预测
    pred = infer.prediction(seq_region, ctcf_region, atac_region, model_path)

    # 画图（与原脚本保持一致的调用）
    if do_plot:
        plot = plot_utils.MatrixPlot(output_path, pred, 'prediction',
                                     celltype, chr_name, start)
        plot.plot()

    return pred


def main():
    parser = argparse.ArgumentParser(description='C.Origami Prediction Module (single or BED).')

    # 原有参数（保持不变）
    parser.add_argument('--out', dest='output_path', default='outputs',
                        help='output path for storing results (default: %(default)s)')
    parser.add_argument('--celltype', dest='celltype',
                        help='Sample cell type for prediction, used for output separation')
    parser.add_argument('--chr', dest='chr_name',
                        help='Chromosome for prediction')
    parser.add_argument('--start', dest='start', type=int,
                        help=f'Starting point for prediction (width is {WINDOW} bp)')
    parser.add_argument('--model', dest='model_path', required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--seq', dest='seq_path', required=True,
                        help='Path to folder where sequence .fa(.gz) files are stored')
    parser.add_argument('--ctcf', dest='ctcf_path', required=True,
                        help='Path to folder where the CTCF ChIP-seq .bw files are stored')
    parser.add_argument('--atac', dest='atac_path', required=True,
                        help='Path to folder where the ATAC-seq .bw files are stored')

    # 新增参数（批量/设备/输出控制）
    parser.add_argument('--bed', dest='bed_path',
                        help='BED file with columns: chr start end (each line one region). If set, overrides --chr/--start.')
    parser.add_argument('--gpu', dest='gpu', default='0',
                        help='CUDA_VISIBLE_DEVICES (default: 0). Set to "" to force CPU.')
    parser.add_argument('--no-plot', dest='no_plot', action='store_true',
                        help='Do not render plots, only save matrices if requested.')
    parser.add_argument('--save-npy', dest='save_npy', action='store_true',
                        help='Save each region prediction as .npy under output folder.')
    parser.add_argument('--save-h5', dest='save_h5', action='store_true',
                        help='Save all predictions into a single HDF5 file.')
    parser.add_argument('--h5-name', dest='h5_name', default='predictions.h5',
                        help='HDF5 filename (used when --save-h5).')

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    # 设备设置（默认单卡）
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 输出目录结构
    base_out = Path(args.output_path)
    safe_mkdir(base_out)

    # 选择单区间 or BED 多区间
    regions: List[Tuple[str, int, int]] = []
    if args.bed_path:
        regions = read_bed(args.bed_path)
    else:
        # 单窗口模式：需要 --chr 与 --start
        if not args.chr_name or args.start is None:
            parser.error("Either provide --bed, or both --chr and --start.")
        regions = [(args.chr_name, int(args.start), int(args.start) + WINDOW)]

    # H5 准备
    if args.save_h5:
        try:
            import h5py
            h5_path = base_out / args.h5_name
            if h5_path.exists():
                h5_path.unlink()
            h5f = h5py.File(h5_path, 'w')
            h5_chr = h5f.create_dataset('chr', (len(regions),), dtype=h5py.string_dtype('utf-8'))
            h5_start = h5f.create_dataset('start', (len(regions),), dtype='i8')
            h5_end = h5f.create_dataset('end', (len(regions),), dtype='i8')
            # 预测矩阵可能大小一致（若 utils 固定窗口），则可一次性建数组；否则逐个 group 存
            # 这里稳妥起见：每个区间一个 group
            h5_grp = h5f.create_group('pred')
        except Exception as e:
            print(f"[WARN] Failed to prepare HDF5 ({e}), fallback to no H5 saving.")
            args.save_h5 = False
            h5f = None
    else:
        h5f = None

    # 逐区间预测
    for i, (chrom, start, end) in enumerate(regions):
        # 与 utils 接口一致，窗口宽度固定从 start 开始
        start_aligned = start  # 这里直接用 start，必要时可加边界检查/对齐逻辑
        out_dir = base_out / f"{chrom}_{start_aligned}"
        safe_mkdir(out_dir)

        try:
            pred = predict_one(
                output_path=str(out_dir),
                celltype=args.celltype,
                chr_name=chrom,
                start=start_aligned,
                model_path=args.model_path,
                seq_path=args.seq_path,
                ctcf_path=args.ctcf_path,
                atac_path=args.atac_path,
                do_plot=(not args.no_plot)
            )
        except Exception as e:
            print(f"[ERROR] {chrom}:{start_aligned}-{start_aligned+WINDOW} failed: {e}")
            continue

        # 可选保存 .npy
        if args.save_npy:
            np.save(out_dir / "prediction.npy", pred)

        # 可选写入 H5
        if h5f is not None:
            h5_chr[i] = chrom
            h5_start[i] = start_aligned
            h5_end[i] = start_aligned + WINDOW
            # 逐个区间存成 pred/{i}
            h5_grp.create_dataset(str(i), data=pred)

        print(f"[OK] {chrom}:{start_aligned}-{start_aligned+WINDOW} → {out_dir}")

    if h5f is not None:
        h5f.close()
        print(f"[H5] Saved to {base_out / args.h5_name}")

    print("All done.")


if __name__ == '__main__':
    main()