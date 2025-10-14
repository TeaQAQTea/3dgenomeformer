import sys
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks

import corigami_models as corigami_models
import genome_dataset_test as genome_dataset
from torchinfo import summary
import os
from blocks import EncoderSplit, TransformerLayer, diDecoder
from concat_test import GenomeModel

# 加载模型、数据处理等代码与之前的相同...

# 新增加载全模型权重的测试初始化
def init_testing(args):
    # 设置GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 初始化模型实例 (使用 GenomeModel)，并加载全模型的权重
    pl_module = GenomeModel(
        encoder=EncoderSplit(num_epi=1, output_size=256),
        transformer=TransformerLayer(d_model=256, nhead=8),
        decoder=diDecoder(hidden_dim=256),
        num_splits=args.num_splits,
        learning_rate=args.lr,
        trainer_max_epochs=args.trainer_max_epochs,
        args=args
    )

    # 加载训练好的模型权重
    checkpoint_path = args.checkpoint_path
    pl_module.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))['state_dict'])
    print(f"Loaded full model weights from {checkpoint_path}.")

    # 初始化测试 Trainer
    pl_trainer = pl.Trainer(
        accelerator="gpu", 
        devices=args.trainer_num_gpu  # 使用 GPU 进行推理
    )

    # 获取测试数据加载器
    testloader = pl_module.get_dataloader(args, 'test')

    # 开始测试
    pl_trainer.test(pl_module, testloader)

def init_parser():
    parser = argparse.ArgumentParser(description='Model Training Module.')
    # 新增预训练权重的路径
    parser.add_argument('--encoder-weights', default=None, help='Path to the pre-trained encoder weights.')
    parser.add_argument('--transformer-weights', default=None, help='Path to the pre-trained transformer weights.')
  # Data and Run Directories
    parser.add_argument('--seed', dest='run_seed', default=2077,
                            type=int,
                            help='Random seed for training')
    parser.add_argument('--save_path', dest='run_save_path', default='checkpoints',
                            help='Path to the model checkpoint')

    # Data directories
    parser.add_argument('--data-root', dest='dataset_data_root', default='data',
                            help='Root path of training data', required=True)
    parser.add_argument('--assembly', dest='dataset_assembly', default='hg38',
                            help='Genome assembly for training data')
    parser.add_argument('--celltype', dest='dataset_celltype', default='imr90',
                            help='Sample cell type for prediction, used for output separation')

    # Model parameters
    parser.add_argument('--model-type', dest='model_type', default='ConvTransModel',
                            help='CNN with Transformer')

    # Training Parameters
    parser.add_argument('--patience', dest='trainer_patience', default=800,
                            type=int,
                            help='Epoches before early stopping')
    parser.add_argument('--max-epochs', dest='trainer_max_epochs', default=1000,
                            type=int,
                            help='Max epochs')
    parser.add_argument('--save-top-n', dest='trainer_save_top_n', default=20,
                            type=int,
                            help='Top n models to save')
    parser.add_argument('--num-gpu', dest='trainer_num_gpu', default=1,
                            type=int,
                            help='Number of GPUs to use')

    # Dataloader Parameters
    parser.add_argument('--batch-size', dest='dataloader_batch_size', default=8, 
                            type=int,
                            help='Batch size')
    parser.add_argument('--ddp-disabled', dest='dataloader_ddp_disabled',
                            action='store_false',
                            help='Using ddp, adjust batch size')
    parser.add_argument('--num-workers', dest='dataloader_num_workers', default=20,
                            type=int,
                            help='Dataloader workers')
    parser.add_argument('--gpu-number', dest='gpu', default='2',
                            type=str,
                            help='gpu id')
    parser.add_argument('--epigenomic-features', nargs='+', dest='epigenomic_features', default=['ctcf_log2fc', 'ro'],)

    parser.add_argument('--log',dest='feature_log', nargs='+',default=['log','log'])

    parser.add_argument('--lr', dest='lr', default=2e-4,
                            type=float,
                            help='Learning rate')
    parser.add_argument('--di_weight',dest='di_weight',default=0.5,type=float,help='di loss weight')
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', default='checkpoints/epoch=999.ckpt', help='Path to the model checkpoint')
    
    parser.add_argument('--num_splits', dest='num_splits', default=8, type=int, help='Number of splits')
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    for i in range (len(args.feature_log)):
        if args.feature_log[i]=='None':
                args.feature_log[i]=None
    return args
# 修改主函数用于测试
def main():
    args = init_parser()
    init_testing(args)

if __name__ == "__main__":
    main()
