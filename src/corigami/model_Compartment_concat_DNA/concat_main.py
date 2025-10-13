import sys
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import os
import corigami_models
import genome_dataset
from blocks import diDecoder,AttnModule,Encoder
from torch.utils.data import DataLoader
import torch.nn as nn
from torchinfo import summary

class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1):
        super(DilatedResidualBlock, self).__init__()
        
        # 使用 nn.Sequential 构建第一个卷积块（含膨胀卷积、BatchNorm 和 ReLU）
        self.scale = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

        # 使用 nn.Sequential 构建残差块（两层卷积、BatchNorm）
        self.res = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
        )

        # 如果输入和输出通道数不同，使用 1x1 卷积匹配通道数

        # 激活函数 ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        # 主路径：通过 scale 卷积块
        out = self.scale(x)

        # 残差路径
        residual = out

        # 将残差和 res 卷积块的输出相加
        out = self.res(out) + residual

        # 通过 ReLU 激活函数
        out = self.relu(out)
        return out
    
# 自定义卷积层 (处理拼接后的特征)
class FeatureConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1):
        super(FeatureConv, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)

# 数据处理流程


def load_partial_weights(model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if 'state_dict' in checkpoint:
            # 只加载模型的 state_dict，而忽略与模型不匹配的层
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"Loaded partial pre-trained weights from {checkpoint_path}.")
        else:
            # 如果是直接的 state_dict 文件，则直接加载
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded pre-trained weights from {checkpoint_path}.")

# Lightning 模型类
class GenomeModel(pl.LightningModule):
    def __init__(self, encoder, transformer, decoder, num_splits, learning_rate, trainer_max_epochs=500, encoder_weights=None, transformer_weights=None,alpha=0):
        super(GenomeModel, self).__init__()
        self.encoder = encoder
        self.transformer = transformer
        self.decoder = decoder
        self.num_splits = num_splits
        self.learning_rate = learning_rate
        self.trainer_max_epochs = trainer_max_epochs
        self.alpha = alpha

        # 卷积层：将输入通道从 2560 转换为 256
        self.block1 = DilatedResidualBlock(in_channels=256, out_channels=128, kernel_size=3, dilation=1, stride=2)  # 序列长度从 2560 减少
        self.block2 = DilatedResidualBlock(in_channels=128, out_channels=256, kernel_size=3, dilation=2, stride=2)  # 进一步减少序列长度
        self.block3 = DilatedResidualBlock(in_channels=256, out_channels=256, kernel_size=3, dilation=2, stride=2)
        
        # 额外的卷积来达到目标维度
        self.conv_final = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)

        # 加载部分预训练的权重 (encoder和transformer)
        if encoder_weights:
            self.encoder.load_state_dict(torch.load(encoder_weights), strict=False)
            print(f"Loaded pre-trained encoder weights from {encoder_weights}.")
        
        if transformer_weights:
            self.transformer.load_state_dict(torch.load(transformer_weights), strict=False)
            print(f"Loaded pre-trained transformer weights from {transformer_weights}.")

    def forward(self, x):
        # 将输入数据沿第三维度分割成 self.num_splits 块，逐块处理
        split_data = torch.chunk(x, self.num_splits, dim=1)  # 不进行 transpose

        # 初始化空的输出列表来保存每个块的输出
        encoded_features = []

        for block in split_data:
            # 注意：每个块的维度需要手动进行 transpose，这样可以逐块处理
            block = torch.transpose(block, 1, 2).contiguous()  # 仅对块操作
            print(block.shape)
            encoded_block = self.encoder(block)  # 每块经过 encoder，形状为 (4, 256, 256)

            # 将每个块的结果添加到输出列表中
            encoded_features.append(encoded_block)

            # 显式释放 GPU 内存
            del encoded_block
            torch.cuda.empty_cache()

        # 将所有编码后的特征块拼接起来 (拼接维度是第 2 维度)

        concatenated_features = torch.cat(encoded_features, dim=2)  # 形状为 (4, 256, num_splits * 256)

        # 将拼接后的特征输入到 transformer 中进行处理
        conv_output = self.block1(concatenated_features)
        conv_output = self.block2(conv_output)
        conv_output = self.block3(conv_output)
        conv_output = self.conv_final(conv_output)
        print(conv_output.shape)
        transformed_features = self.transformer(conv_output.permute(0, 2, 1))

        # 转置拼接后的特征，使其适合 Conv1d
        transformed_features = transformed_features.permute(0, 2, 1)  

        # 通过卷积层，将通道从 2560 变为 256
         # 形状为 (4, 256, 256)

        # 最终通过 decoder 进行解码
        # transformed_features = torch.concat([transformed_features,conv_output],dim=1)
        print(transformed_features.shape)
        output = self.decoder(transformed_features)

        return output

    def proc_batch(self, batch):
        seq, di, start, end, chr_name, chr_idx = batch
        # 拼接 sequence 和 features
        # features = torch.cat([feat.unsqueeze(2) for feat in features], dim=2)
        inputs = seq.float()
        di = di.float()
        return inputs, di

    def training_step(self, batch, batch_idx):
        inputs, di = self.proc_batch(batch)  # 假设 proc_batch 返回输入数据和回归标签

        # 根据 di 创建分类标签
        labels = (di > 0).long()  # 如果 di 为负，则 label 为 1；否则为 0

        # 模型前向计算
        outputs = self(inputs)

        # 计算回归损失（MSELoss）
        criterion_di = torch.nn.MSELoss()
        regression_loss = criterion_di(outputs, di)

        # 计算分类损失（CrossEntropyLoss）
        # 首先需要将输出转换为符号分类的概率
        classification_logits = outputs.squeeze()  # 假设输出是 [batch_size, 1]
        criterion_class = torch.nn.BCELoss()  # 使用BCEWithLogitsLoss处理二分类问题

        classification_loss = criterion_class(classification_logits, labels.float().squeeze())  # 将标签转换为浮点型

        # 总损失：加权组合回归和分类损失
        alpha = 1  # 可调整的权重系数
        beta = 1 - alpha
        loss = alpha * regression_loss + beta * classification_loss

        # 记录损失
        metrics = {
            'train_step_loss': loss,
            'regression_loss': regression_loss,
            'classification_loss': classification_loss
        }
        self.log_dict(metrics, batch_size=inputs.shape[0], prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def _shared_eval_step(self, batch, batch_idx):
    # 处理输入数据和回归标签
        inputs, di = self.proc_batch(batch)

        # 生成分类标签：将 di 为负的变为 1，非负的变为 0
        labels = (di >0).long()  # 将布尔值转换为整数类型

        # 模型前向计算，得到输出
        outputs = self(inputs)

        # 计算回归损失（MSELoss）
        criterion_di = torch.nn.MSELoss()
        regression_loss = criterion_di(outputs, di)

        # 计算分类损失（BCEWithLogitsLoss）
        classification_logits = outputs.squeeze()  # 假设输出为 [batch_size, 1]
        labels = labels.squeeze()  # 将标签转换为浮点型
        criterion_class = torch.nn.BCELoss()  # 使用BCEWithLogitsLoss处理二分类问题
        print(classification_logits.shape)
        print(labels.shape)
        classification_loss = criterion_class(classification_logits, labels.float())  # 将标签转换为浮点型

        # 总损失：加权组合回归和分类损失
        alpha= 1  # 可调整的权重系数
        beta = 1 - alpha
        loss = alpha * regression_loss + beta * classification_loss

        # 记录损失
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_regression_loss", regression_loss, prog_bar=True)
        self.log("val_classification_loss", classification_loss, prog_bar=True)

        return loss


    # 记录每个 epoch 的统计信息
    def training_epoch_end(self, step_outputs):
        step_outputs = [out['loss'] for out in step_outputs]
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'train_loss': ret_metrics['loss']}
        self.log_dict(metrics, prog_bar=True)

    def validation_epoch_end(self, step_outputs):
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'val_loss': ret_metrics['loss']}
        self.log_dict(metrics, prog_bar=True)

    def _shared_epoch_end(self, step_outputs):
        loss = torch.tensor(step_outputs).mean()
        return {'loss': loss}

    def configure_optimizers(self):
        # 配置 Adam 优化器和 CosineAnnealingLR 学习率调度器
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr = self.learning_rate,
                                     weight_decay = 0)

        import pl_bolts
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=self.trainer_max_epochs)
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': 'WarmupCosineAnnealing',
        }
        return {'optimizer' : optimizer, 'lr_scheduler' : scheduler_config}

    def get_dataset(self, args, mode):
        # 获取数据集的路径
        celltype_root = f'{args.dataset_data_root}/{args.dataset_assembly}/{args.dataset_celltype}'
        
        # 动态生成 genomic_features 字典
        genomic_features = {}
        for i in range(len(args.epigenomic_features)):
            genomic_features[args.epigenomic_features[i]] = {
                'file_name': f'{args.epigenomic_features[i]}',
                'norm': args.feature_log[i]
            }

        # 将生成的 genomic_features 传入 genome_dataset
        dataset = genome_dataset.GenomeDataset(
            celltype_root, args.dataset_assembly, genomic_features, mode=mode,
            include_sequence=True, include_genomic_features=False
        )
        return dataset

    def get_dataloader(self, args, mode):
        dataset = self.get_dataset(args, mode)
        shuffle = mode == 'train'
        batch_size = args.dataloader_batch_size
        num_workers = args.dataloader_num_workers

        if not args.dataloader_ddp_disabled:
            gpus = args.trainer_num_gpu
            batch_size = int(args.dataloader_batch_size / gpus)
            num_workers = int(args.dataloader_num_workers / gpus)

        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=shuffle, batch_size=batch_size,
            num_workers=num_workers, pin_memory=True, prefetch_factor=1,
            persistent_workers=True
        )
        return dataloader

    def get_model(self, args):
        model_name = args.model_type
        num_genomic_features = len(args.epigenomic_features)
        ModelClass = getattr(corigami_models, model_name)
        model = ModelClass(num_genomic_features, mid_hidden=256)
        return model


# 初始化参数解析
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
    
    parser.add_argument('--num_splits', dest='num_splits', default=8, type=int, help='Number of splits')
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    for i in range (len(args.feature_log)):
        if args.feature_log[i]=='None':
                args.feature_log[i]=None
    return args

# 数据集生成


# 训练初始化
def init_training(args):

    # Early_stopping
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 设置早停回调函数
    early_stop_callback = callbacks.EarlyStopping(
        monitor='val_loss', 
        min_delta=0.00, 
        patience=args.trainer_patience,  # 早停的 patience
        verbose=False,
        mode="min"  # 最小化 val_loss
    )

    # 设置时间戳
    import time
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    
    # 模型检查点保存路径
    args.run_save_path = f'{args.run_save_path}/{timestamp}'
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=f'{args.run_save_path}_models',  # 保存模型的路径
        save_top_k=args.trainer_save_top_n,  # 保留 top N 的最佳模型
        monitor='val_loss'  # 基于 val_loss 保存最佳模型
    )

    # 学习率监控回调
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')

    # 使用 TensorBoard 记录训练日志
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=f'{args.run_save_path}/tb')


    # 设置随机种子
    pl.seed_everything(args.run_seed, workers=True)

    # 初始化模型实例 (使用 GenomeModel)
    pl_module = GenomeModel(
        encoder=Encoder(in_channel=5),
        transformer=AttnModule(hidden=256),
        decoder=diDecoder(hidden_dim=256),
        num_splits=args.num_splits,
        learning_rate=args.lr,
        trainer_max_epochs=args.trainer_max_epochs,
        encoder_weights=args.encoder_weights, 
        transformer_weights=args.transformer_weights
    )

    # 初始化 Trainer
    pl_trainer = pl.Trainer(
        strategy='ddp',  # 使用分布式数据并行
        accelerator="gpu", devices=args.trainer_num_gpu,  # 使用多 GPU
        gradient_clip_val=1,  # 梯度裁剪
        logger=tb_logger, # 记录训练过程
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],  # 早停、检查点保存、LR监控
        max_epochs=args.trainer_max_epochs  # 最大训练 epoch 数
    )

    # 获取训练和验证数据加载器
    trainloader = pl_module.get_dataloader(args, 'train')
    valloader = pl_module.get_dataloader(args, 'val')

    # 开始训练
    pl_trainer.fit(pl_module, trainloader, valloader)


# 主函数
def main():
    args = init_parser()
    init_training(args)

if __name__ == "__main__":
    main()
