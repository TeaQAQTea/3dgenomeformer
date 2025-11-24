import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

import inference.utils.inference_utils as infer
from inference.utils import plot_utils
import h5py

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
                log= [None, None],
                do_plot: bool = True) -> np.ndarray:
    """
    与你原 single_prediction 等价，但可选择不画图，仅返回矩阵。
    """
    # 加载区域数据（utils 内部负责切片/标准化）

    print(f"Loading data for {chr_name}:{start}-{start + WINDOW}...")
    seq_region, ctcf_region, atac_region = infer.load_region(
        chr_name, start, seq_path, ctcf_path, atac_path, log=log
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
    parser.add_argument('--log',dest='feature_log', nargs='+',default=['log','log'])

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
        if not args.chr_name or args.start is None:
            parser.error("Either provide --bed, or both --chr and --start.")
        regions = [(args.chr_name, int(args.start), int(args.start) + WINDOW)]

    # ---------- H5：改为可增长的单一数据集 ----------
    h5f = None
    d_pred = d_chr = d_start = d_end = None
    pred_hw = None
    written_n = 0

    if args.save_h5:
        try:
            h5_path = base_out / args.h5_name
            if h5_path.exists():
                h5_path.unlink()
            h5f = h5py.File(h5_path, 'w')

            # 先建空数据集，占位，第一次写入时再根据 (H,W) 重建
            d_pred = h5f.create_dataset(
                "predict", shape=(0, 1, 1), maxshape=(None, None, None),
                dtype="f4", chunks=True, compression="gzip", compression_opts=4
            )
            d_chr = h5f.create_dataset(
                "chr", shape=(0,), maxshape=(None,),
                dtype=h5py.string_dtype('utf-8'), chunks=True
            )
            d_start = h5f.create_dataset(
                "start", shape=(0,), maxshape=(None,), dtype='i8', chunks=True
            )
            d_end = h5f.create_dataset(
                "end", shape=(0,), maxshape=(None,), dtype='i8', chunks=True
            )
        except Exception as e:
            print(f"[WARN] Failed to prepare HDF5 ({e}), fallback to no H5 saving.")
            args.save_h5 = False
            h5f = None

    # 逐区间预测
    for (chrom, start, end) in regions:
        # 与 utils 接口一致
        start_aligned = start
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
                do_plot=(not args.no_plot),
                log=args.feature_log
            )
        except Exception as e:
            print(f"[ERROR] {chrom}:{start_aligned}-{start_aligned+WINDOW} failed: {e}")
            continue

        # 可选保存 .npy
        if args.save_npy:
            np.save(out_dir / "prediction.npy", pred)

        # H5 追加写入（可增长）
        if h5f is not None:
            pred = np.asarray(pred)
            if pred_hw is None:
                # 第一次写入时，确定 (H,W)，重建 predict 数据集的尾维
                H, W = pred.shape[-2], pred.shape[-1]
                pred_hw = (H, W)
                # 删除占位并重建（固定尾维方便后续追加）
                del h5f["predict"]
                d_pred = h5f.create_dataset(
                    "predict", shape=(0, H, W), maxshape=(None, H, W),
                    dtype="f4", chunks=(1, H, W), compression="gzip", compression_opts=4
                )

            # 统一扩容 + 赋值
            s, e = written_n, written_n + 1
            d_pred.resize((e, *pred_hw))
            d_chr.resize((e,))
            d_start.resize((e,))
            d_end.resize((e,))

            d_pred[s:e, :, :] = pred[np.newaxis, ...]
            d_chr[s:e] = np.array([chrom])
            d_start[s:e] = np.array([start_aligned], dtype=np.int64)
            d_end[s:e] = np.array([start_aligned + WINDOW], dtype=np.int64)

            written_n = e

        print(f"[OK] {chrom}:{start_aligned}-{start_aligned+WINDOW} → {out_dir}")

    if h5f is not None:
        h5f.flush()
        h5f.close()
        print(f"[H5] Saved to {base_out / args.h5_name}  (N={written_n})")

    print("All done.")

if __name__ == '__main__':
    main()