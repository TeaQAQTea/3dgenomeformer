import os, re, sys, argparse, torch
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from typing import List, Pattern, Iterable

# === 引入你现有项目的模块 ===
import corigami.model.corigami_models as corigami_models
import genome_dataset
from main import TrainModule  # 假设你把上面的训练脚本命名为 train.py，并且其中定义了 TrainModule

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune C.Origami models.")

    # ====== 基本与数据路径（沿用你的训练脚本命名）======
    parser.add_argument('--data-root', dest='dataset_data_root', required=True, help='Root path of training data')
    parser.add_argument('--assembly', dest='dataset_assembly', default='hg38')
    parser.add_argument('--celltype', dest='dataset_celltype', default='imr90')
    parser.add_argument('--epigenomic-features', nargs='+', dest='epigenomic_features', default=['ctcf_log2fc','ro'])
    parser.add_argument('--log', dest='feature_log', nargs='+', default=['log','log'])

    # ====== 预训练与保存 ======
    parser.add_argument('--pretrained-ckpt', type=str, required=True, help='Path to pretrained .ckpt')
    parser.add_argument('--save-path', dest='run_save_path', default='finetune_ckpts', help='Where to save fine-tuned ckpts')

    # ====== 微调策略 ======
    parser.add_argument('--freeze-patterns', nargs='*', default=[],
                        help='Regex or glob-like substrings to freeze (match on parameter names). e.g. backbone|encoder|conv')
    parser.add_argument('--unfreeze-patterns', nargs='*', default=[],
                        help='Regex/substring patterns to force unfreeze even if frozen earlier.')
    parser.add_argument('--reinit-head', action='store_true',
                        help='Reinit (reset) last-layer(s) commonly used as head. 需要你在下面的函数里按项目习惯补充匹配规则。')
    parser.add_argument('--head-patterns', nargs='*', default=['head', 'out', 'proj', 'classifier', 'fc_out'],
                        help='Parameter name patterns (regex/substring) treated as head when reinitializing.')

    # ====== 训练超参（微调用较小LR）======
    parser.add_argument('--seed', dest='run_seed', type=int, default=2077)
    parser.add_argument('--lr', type=float, default=5e-5, help='New learning rate for fine-tuning')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--batch-size', dest='dataloader_batch_size', type=int, default=8)
    parser.add_argument('--num-workers', dest='dataloader_num_workers', type=int, default=8)

    # ====== Trainer/硬件 ======
    parser.add_argument('--strategy', default='ddp', choices=['ddp','auto','none'])
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--gpu', dest='gpu', default='0', help='CUDA_VISIBLE_DEVICES, e.g. "0" or "0,1"')
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save-top-n', type=int, default=5)

    # ====== 其他 ======
    parser.add_argument('--model-type', dest='model_type', default='ConvTransModel')
    args = parser.parse_args()

    # 对 log 列表中的 "None" 字符串做转换
    for i in range(len(args.log)):
        if args.log[i] == 'None':
            args.log[i] = None
    return args


# ========== 工具函数：参数名匹配/冻结 ==========
def _compile_patterns(patterns: Iterable[str]) -> List[Pattern]:
    compiled = []
    for p in patterns:
        try:
            compiled.append(re.compile(p))
        except re.error:
            # 退化为简单的子串匹配：转义所有非字母数字字符
            compiled.append(re.compile(re.escape(p)))
    return compiled

def _name_match_any(name: str, patterns: List[Pattern]) -> bool:
    return any(bool(p.search(name)) for p in patterns)

def freeze_by_name(model: torch.nn.Module, freeze_patterns: List[str], unfreeze_patterns: List[str]):
    """按参数名冻结/解冻。先冻结匹配 freeze_patterns 的，再强制解冻匹配 unfreeze_patterns 的。"""
    freeze_re = _compile_patterns(freeze_patterns)
    unfreeze_re = _compile_patterns(unfreeze_patterns)

    for n, p in model.named_parameters():
        if _name_match_any(n, freeze_re):
            p.requires_grad = False
    for n, p in model.named_parameters():
        if _name_match_any(n, unfreeze_re):
            p.requires_grad = True

def reinit_head_layers(model: torch.nn.Module, head_patterns: List[str]):
    """尝试将匹配到的“头部层”重新初始化（Linear/Conv/Norm）。"""
    head_re = _compile_patterns(head_patterns)

    def _maybe_reset(m: torch.nn.Module, prefix: str):
        # Linear
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(m.bias, -bound, bound)
        # Conv
        elif isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        # Norm
        elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.GroupNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    import math
    for name, module in model.named_modules():
        if _name_match_any(name, head_re):
            _maybe_reset(module, name)

def count_trainable(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ========== 载入预训练权重到 LightningModule ==========
    # 通过 load_from_checkpoint 调用 TrainModule.__init__(args) 并加载匹配到的权重
    pl.seed_everything(args.run_seed, workers=True)
    ft_module: TrainModule = TrainModule.load_from_checkpoint(
        checkpoint_path=args.pretrained_ckpt,
        args=argparse.Namespace(
            # 下面这些与 TrainModule 期望的字段名保持一致（沿用你的原训练脚本参数命名）
            run_seed=args.run_seed,
            run_save_path=args.run_save_path,
            dataset_data_root=args.dataset_data_root,
            dataset_assembly=args.dataset_assembly,
            dataset_celltype=args.dataset_celltype,
            epigenomic_features=args.epigenomic_features,
            feature_log=args.log,
            model_type=args.model_type,
            trainer_patience=args.patience,
            trainer_max_epochs=args.max_epochs,
            trainer_save_top_n=args.save_top_n,
            trainer_num_gpu=args.devices,
            dataloader_batch_size=args.dataloader_batch_size,
            dataloader_ddp_disabled=False,  # 微调默认仍按DDP缩放
            dataloader_num_workers=args.dataloader_num_workers,
            gpu=args.gpu,
            lr=args.lr,  # 新学习率
        )
    )

    # ========== 可选：重新初始化头部 ==========
    if args.reinit_head:
        reinit_head_layers(ft_module.model, args.head_patterns)

    # ========== 冻结/解冻 ==========
    if args.freeze_patterns or args.unfreeze_patterns:
        freeze_by_name(ft_module.model, args.freeze_patterns, args.unfreeze_patterns)

    # 统计可训练参数数量
    total_p, trainable_p = count_trainable(ft_module.model)
    print(f"[Fine-tune] Total params: {total_p:,} | Trainable: {trainable_p:,}")

    # ========== 重设优化器与调度器（用新LR） ==========
    # 直接覆盖 configure_optimizers 所需的值
    ft_module.args.lr = args.lr

    # ========== 回调与日志 ==========
    import time
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    save_dir = f"{args.run_save_path}/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=args.patience, verbose=True
    )
    ckpt_cb = callbacks.ModelCheckpoint(
        dirpath=f"{save_dir}_models",
        save_top_k=args.save_top_n,
        monitor='val_loss',
        mode='min'
    )
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=f"{save_dir}/tb")
    csv_logger = pl.loggers.CSVLogger(save_dir=f"{save_dir}/csv")

    # ========== DataLoaders ==========
    trainloader = ft_module.get_dataloader(ft_module.args, 'train')
    valloader   = ft_module.get_dataloader(ft_module.args, 'val')

    # ========== Trainer ==========
    # 注意：我们不是 resume_from_checkpoint，而是“用预训练权重初始化后重新训练”
    strategy = None if args.strategy == 'none' else args.strategy
    trainer = pl.Trainer(
        strategy=strategy,
        accelerator="gpu",
        devices=args.devices,
        gradient_clip_val=1.0,
        logger=[tb_logger, csv_logger],
        callbacks=[early_stop, ckpt_cb, lr_monitor],
        max_epochs=args.max_epochs
    )

    trainer.fit(ft_module, trainloader, valloader)


if __name__ == '__main__':
    main()