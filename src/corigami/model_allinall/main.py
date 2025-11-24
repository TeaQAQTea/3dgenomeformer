import sys
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import os
import corigami_models as corigami_models
import genome_dataset
from torchinfo import summary
import pl_bolts
from typing import List, Pattern, Iterable
import torch.nn.functional as F
from tqdm import tqdm
import cell_dataset
def main():
    args = init_parser()
    if args.mode == 'test':
        init_test(args)
    elif args.mode == 'train':
        init_training(args)
    elif args.mode == 'finetune':
        init_finetune(args)

def init_parser():
  parser = argparse.ArgumentParser(description='C.Origami Training Module.')

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
  parser.add_argument('--celltype', dest='dataset_celltype', nargs='+',default=['imr90'],
                        help='Sample cell type for prediction, used for output separation')
  # Model parameters
  parser.add_argument('--model-type', dest='model_type', default='ConvTransModel',
                        help='CNN with Transformer')
  parser.add_argument('--model-path', dest='model_path', default='',
                        help='Path to the pre-trained model')
  parser.add_argument('--mode', dest='mode', default='train',
                        help='train, test or finetune')

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
  parser.add_argument('--num-gpu', dest='trainer_num_gpu', default=4,
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
  parser.add_argument('--gpu-number', dest='gpu', nargs='+',  default=['0'], help='GPU id(s), e.g. --gpu-number 0 1')
  
  parser.add_argument('--epigenomic-features', nargs='+', dest='epigenomic_features', default=['ctcf_log2fc', 'ro'],)

  parser.add_argument('--log',dest='feature_log', nargs='+',default=['log','log'])

  parser.add_argument('--backbone', dest='backbone', default='Mamba',
                        help='Model backbone, choose from Mamba or Transformer')
  parser.add_argument('--resolution', dest='resolution', default=4096,
                        type=int,
                        help='Resolution for Hi-C contact map, choose from 2048 4096 or 8192')
  parser.add_argument('--cool-res', dest='cool_res', default=5000,
                        type=int,
                        help='Resolution for cool file, choose from 3000 5000 or 10000')
  parser.add_argument('--lr', dest='lr', default=2e-4,
                        type=float,
                        help='Learning rate')
  parser.add_argument('--freeze-patterns', nargs='*', default=[],
                        help='Regex or glob-like substrings to freeze (match on parameter names). e.g. backbone|encoder|conv')
  parser.add_argument('--unfreeze-patterns', nargs='*', default=[],
                        help='Regex/substring patterns to force unfreeze even if frozen earlier.')
  parser.add_argument('--reinit-head', action='store_true',
                        help='Reinit (reset) last-layer(s) commonly used as head. 需要你在下面的函数里按项目习惯补充匹配规则。')
  parser.add_argument('--loss-type', dest='loss_type',type=str,default='mse',choices=['mse', 'insulation_combined'],help='Type of loss function to use: mse | distance_decay | insulation_combined')
  parser.add_argument('--target-treatment', dest='target_treatment',type=str,default='clip',choices=['linear','clip','log'],help='How to treat the target Hi-C contact map: log1p | raw')
  parser.add_argument('--weight',dest = 'use_weight',action='store_true',help='Whether to use weighting in loss function.')
  parser.add_argument('--opt-type', dest='opt_type',type=str,default='adam',choices=['adam','adamw'],help='Type of optimizer to use: adam | adamw')
  parser.add_argument('--clip-pc', dest ='clip_pc',type=float,default=95,help='Percentile for clipping the target Hi-C contact map.')
  parser.add_argument('--sample-length', dest='sample_length', type=int,
                    default=2097152,
                    help='Length of the input genomic sequence (e.g., 2Mb = 2097152).')
  args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
  if isinstance(args.gpu, (list, tuple)):
    args.gpu = ",".join(str(i) for i in args.gpu)
  else:
    args.gpu = str(args.gpu)

  for i in range (len(args.feature_log)):
      if args.feature_log[i]=='None':
            args.feature_log[i]=None
  assert len(args.epigenomic_features) == len(args.feature_log), "Number of epigenomic features and log methods should be the same."
  return args

def init_test(args):


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #时间戳

    tb_logger = pl.loggers.TensorBoardLogger(save_dir = f'{args.run_save_path}/tb')
    
    pl_module = TrainModule(args)
    checkpoint_path=args.model_path
    testloader = pl_module.get_dataloader(args, 'test')
    

    model = TrainModule.load_from_checkpoint(checkpoint_path, args=args, strict=False)
    trainer = pl.Trainer(gpus=args.trainer_num_gpu, logger=tb_logger)
    output=trainer.test(model, testloader)
    print(output)


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

def init_finetune(args):
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.lr = args.lr/50
    
    # ========== 载入预训练权重到 LightningModule ==========
    # 通过 load_from_checkpoint 调用 TrainModule.__init__(args) 并加载匹配到的权重
    pl.seed_everything(args.run_seed, workers=True)
    ft_module: TrainModule = TrainModule.load_from_checkpoint(
        checkpoint_path=args.model_path,
        args=args
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
        monitor='val_loss', mode='min', patience=args.trainer_patience, verbose=True
    )
    ckpt_cb = callbacks.ModelCheckpoint(
        dirpath=f"{save_dir}_models",
        save_top_k=args.trainer_save_top_n,
        monitor='val_loss',
        mode='min'
    )
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=f"{save_dir}/tb")
    csv_logger = pl.loggers.CSVLogger(save_dir=f"{save_dir}/csv")

    # ========== DataLoaders ==========
    trainloader = ft_module.get_dataloader(ft_module.args, 'finetune')
    valloader   = ft_module.get_dataloader(ft_module.args, 'val')

    # ========== Trainer ==========
    # 注意：我们不是 resume_from_checkpoint，而是“用预训练权重初始化后重新训练”

    trainer = pl.Trainer(
        strategy='ddp',
        accelerator="gpu",
        devices=args.trainer_num_gpu,
        gradient_clip_val=1.0,
        logger=[tb_logger, csv_logger],
        callbacks=[early_stop, ckpt_cb, lr_monitor],
        max_epochs=args.trainer_max_epochs
    )

    trainer.fit(ft_module, trainloader, valloader)


def init_training(args):

    # Early_stopping

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', 
                                        min_delta=0.00, 
                                        patience=args.trainer_patience,
                                        verbose=False,
                                        mode="min")
    import time
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    
    args.run_save_path = f'{args.run_save_path}/{timestamp}'
    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'{args.run_save_path}_models',
                                        save_top_k=args.trainer_save_top_n, 
                                        monitor='val_loss')

    # LR monitor
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')

    # Logger
    #使用tensorboard记录
    tb_logger = pl.loggers.TensorBoardLogger(save_dir = f'{args.run_save_path}/tb')


    # Assign seed
    pl.seed_everything(args.run_seed, workers=True)
    pl_module = TrainModule(args)
    if args.model_path != '':
        print(f'Loading model weights from {args.model_path} for training...')
        state_dict = torch.load(args.model_path, map_location='cpu')['state_dict']
        pl_module.load_state_dict(state_dict, strict=False)
    trainloader = pl_module.get_dataloader(args, 'train')
    valloader = pl_module.get_dataloader(args, 'val')
    num_visible = torch.cuda.device_count()
    use_ddp = (num_visible >= 2) and (int(args.trainer_num_gpu) >= 2)
    pl_trainer = pl.Trainer(
            accelerator="ddp",           # ✅ 1.6 常用写法；等价写法：strategy="ddp", accelerator="gpu"
            gpus=int(args.trainer_num_gpu),  # ✅ 用 gpus，不用 devices
            gradient_clip_val=1.0,
            logger=tb_logger,
            callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
            max_epochs=int(args.trainer_max_epochs),          # ✅ 可选：混合精度省显存
        )
    pl_trainer.fit(pl_module, trainloader, valloader)

class TrainModule(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.model = self.get_model(args)
        if not os.path.exists(args.run_save_path):
            os.makedirs(args.run_save_path)

        self.args = args
        with open(f'{args.run_save_path}/model_summary.txt', 'w') as f:
            sys.stdout = f
            summary(self.model,input_size=(args.dataloader_batch_size,args.sample_length , len(args.epigenomic_features)+5))
            sys.stdout = sys.__stdout__

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    


    def proc_batch_test(self, batch):
        seq, features, mat, start, end, chr_name, chr_idx, weight = batch
        features = torch.cat([feat.unsqueeze(2) for feat in features], dim = 2)
        inputs = torch.cat([seq, features], dim = 2)
        mat = mat.float()
        return inputs, mat, start, end, chr_name, chr_idx, weight

    def test_step(self, batch, batch_idx):
        import os, h5py, numpy as np, torch

        r = self._shared_test_step(batch, batch_idx)

        # 标量
        loss_val = r['loss'].detach().cpu().item()

        # 转 CPU + numpy
        pred_np  = r['predict'].detach().cpu().numpy()            # (B, H, W)
        tgt_np   = r['target'].detach().cpu().numpy()             # (B, H, W)
        start_np = r['start'].detach().cpu().numpy().reshape(-1)  # (B,)
        end_np   = r['end'].detach().cpu().numpy().reshape(-1)    # (B,)


        if torch.is_tensor(r['chr_name']):
            chr_np = r['chr_name'].detach().cpu().numpy().reshape(-1)
        else:
            # 字符串或其他类型，直接转为 numpy 对象数组
            chr_np = np.array(r['chr_name'], dtype=object).reshape(-1)

        if not hasattr(self, "_h5"):
            out_dir = self.args.run_save_path
            os.makedirs(out_dir, exist_ok=True)
            self._h5_path = os.path.join(out_dir, "output.h5")
            # 覆盖旧文件
            if os.path.exists(self._h5_path):
                os.remove(self._h5_path)
            self._h5 = h5py.File(self._h5_path, "w")

            # 可增长一维元数据
            self._d_start   = self._h5.create_dataset("start",    shape=(0,), maxshape=(None,), dtype="i8", chunks=True)
            self._d_end     = self._h5.create_dataset("end",      shape=(0,), maxshape=(None,), dtype="i8", chunks=True)
            self._d_chrname = self._h5.create_dataset("chr_name", shape=(0,), maxshape=(None,), dtype = h5py.string_dtype(encoding='utf-8'), chunks=True)

            # 可增长矩阵
            pred_tail = pred_np.shape[1:]     # (H, W)
            tgt_tail  = tgt_np.shape[1:]      # (H, W)
            self._d_pred = self._h5.create_dataset(
                "predict", shape=(0, *pred_tail), maxshape=(None, *pred_tail),
                dtype="f4", chunks=(1, *pred_tail)
            )
            self._d_tgt = self._h5.create_dataset(
                "target", shape=(0, *tgt_tail), maxshape=(None, *tgt_tail),
                dtype="f4", chunks=(1, *tgt_tail)
            )
            self._ptr = 0
            self._att_ptr = 0
        B = pred_np.shape[0]
        s, e = self._ptr, self._ptr + B

        self._d_start.resize((e,));   self._d_start[s:e]   = start_np
        self._d_end.resize((e,));     self._d_end[s:e]     = end_np
        self._d_chrname.resize((e,)); self._d_chrname[s:e] = chr_np

        self._d_pred.resize((e, *self._d_pred.shape[1:])); self._d_pred[s:e, ...] = pred_np
        self._d_tgt.resize((e,  *self._d_tgt.shape[1:]));  self._d_tgt[s:e,  ...] = tgt_np

        self._ptr = e
        try: torch.cuda.empty_cache()
        except: pass
        return {"loss": loss_val}

    def get_loss_fn(self, weight=None):
        if self.args.loss_type == 'mse':
            return lambda outputs, mat, batch=None: self.weighted_mse_loss(outputs, mat, weight=weight)
        elif self.args.loss_type == 'insulation_combined':
            return lambda outputs, mat, batch=None: self._loss_insulation_combined(outputs, mat, weight=weight)
        else:
            raise ValueError(f"Unknown loss_type: {self.args.loss_type}")


    def weighted_mse_loss(self, input, target, weight=None, reduction='mean'):
        """
        模拟 F.mse_loss(input, target, weight=...)
        - 支持 weight: 和 input 同 shape 或可广播
        - reduction: 'none', 'mean', 'sum'
        """
        loss = (input - target) ** 2
        if weight is not None:
            loss = loss * weight

        if reduction == 'mean':
            if weight is not None:
                return loss.sum() / weight.sum()
            else:
                return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

    def build_insulation_kernel(self,size=51, scale=1e-3, device=None, dtype=torch.float32):
        assert size % 2 == 1
        mat = torch.ones((size, size), dtype=dtype, device=device)
        h = size // 2
        mat[:h, h+1:] = -1
        mat[h+1:, :h] = -1
        return (mat * scale).unsqueeze(0).unsqueeze(0)  # [1,1,K,K]

    def _loss_insulation_combined(self, outputs: torch.Tensor, mat: torch.Tensor,weight:torch.Tensor):
        if not hasattr(self, 'ins_kernel') or self.ins_kernel is None:
            ksize = getattr(self, 'ins_kernel_size', 51)
            self.register_buffer('ins_kernel', self.build_insulation_kernel(ksize, device=outputs.device))
        k = self.ins_kernel.to(outputs.device)

        mse = self.weighted_mse_loss(outputs, mat, weight=weight)

        y_true = mat.unsqueeze(1)      # [B,1,L,L]
        y_pred = outputs.unsqueeze(1)  # [B,1,L,L]
        conv_true = F.conv2d(y_true, k, padding=0).squeeze(1)  # [B,L',L']
        conv_pred = F.conv2d(y_pred, k, padding=0).squeeze(1)  # [B,L',L']

        diag_true = torch.diagonal(conv_true, dim1=-2, dim2=-1)  # [B,L']
        diag_pred = torch.diagonal(conv_pred, dim1=-2, dim2=-1)  # [B,L']

        ins = F.mse_loss(diag_pred, diag_true)

        beta = getattr(self, 'beta_ins', 10.0)
        return mse + beta * ins

    def pearson_loss(self,pred, target, eps=1e-8):
        pred = pred - pred.mean(dim=(-1,-2), keepdim=True)
        target = target - target.mean(dim=(-1,-2), keepdim=True)
        num = (pred * target).sum(dim=(-1,-2))
        den = torch.sqrt((pred**2).sum(dim=(-1,-2)) * (target**2).sum(dim=(-1,-2)) + eps)
        r = num / den
        return (1 - r).mean()
    
    def test_epoch_end(self, step_outputs):
        import numpy as np

        losses = [o["loss"] for o in step_outputs]
        mean_loss = float(np.mean(losses)) if len(losses) > 0 else float("nan")
        self.log("test_loss", mean_loss, prog_bar=True)
        print(f"[test] mean loss = {mean_loss:.6f}")

        if hasattr(self, "_h5") and self._h5 is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None
            print(f"Output saved to {self._h5_path}")

    def _shared_test_step(self, batch, batch_idx):
        inputs, mat,start, end, chr_name, chr_idx, weight = self.proc_batch_test(batch)
        outputs= self(inputs)
        loss_fn = self.get_loss_fn(weight=weight)
        loss = loss_fn(outputs, mat)                # 计算损失
        return {'loss' : loss,'target':mat,'predict':outputs,'start':start,'end':end,'chr_name':chr_name,'chr_idx':chr_idx}
    
    def proc_batch(self, batch):
        seq, features, mat, start, end, chr_name, chr_idx, weight = batch
        features = torch.cat([feat.unsqueeze(2) for feat in features], dim = 2)
        inputs = torch.cat([seq, features], dim = 2)
        mat = mat.float()
        return inputs, mat, weight

    def training_step(self, batch, batch_idx):
        inputs, mat, weight = self.proc_batch(batch)
        outputs = self(inputs)

        loss_fn = self.get_loss_fn(weight=weight)          # mse 或 insulation_combined
        loss = loss_fn(outputs, mat)                # 计算损失

        # === 计算皮尔逊相关（若你定义过）===
        pearson = self.pearson_loss(outputs, mat) if hasattr(self, 'pearson_loss') else torch.tensor(0.0, device=loss.device)

        # === 记录指标 ===
        metrics = {
            'train_step_loss': loss,
            'train_step_pearson': pearson
        }
        self.log_dict(metrics, batch_size=inputs.shape[0], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)
        return ret_metrics


    def _shared_eval_step(self, batch, batch_idx):
       
        inputs, mat, weight = self.proc_batch(batch)

        # ====== 2. 检查输入 ======
        for name, tensor in zip(["inputs", "mat", "weight"], [inputs, mat, weight]):
            if not torch.isfinite(tensor).all():
                self.print(f"[val] {name} 非有限值 (batch {batch_idx})，跳过该 batch")
                return {"eval_loss": torch.tensor(0.0, device=self.device)}

        # ====== 3. 前向传播 (关 AMP) ======
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            outputs = self(inputs)

            # 检查输出
            if not torch.isfinite(outputs).all():
                self.print(f"[val] outputs 非有限值 (batch {batch_idx})，clamp 修正")
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4)
                outputs = torch.clamp(outputs, -1e4, 1e4)

            # ====== 4. 计算损失 ======
            loss_fn = self.get_loss_fn(weight=weight)
            loss = loss_fn(outputs, mat)

            # ====== 5. 检查损失 ======
            if not torch.isfinite(loss):
                self.print(f"[val] loss NaN/Inf (batch {batch_idx}); "
                        f"outputs∈[{outputs.min().item():.2e},{outputs.max().item():.2e}], "
                        f"mat∈[{mat.min().item():.2e},{mat.max().item():.2e}], "
                        f"weight∈[{weight.min().item():.2e},{weight.max().item():.2e}]")
                loss = torch.tensor(0.0, device=self.device)

        # ====== 6. 记录日志 ======
        self.log(
            'eval_loss', loss,
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
            batch_size=inputs.shape[0]
        )

        return loss

    # Collect epoch statistics
    def training_epoch_end(self, step_outputs):
        step_outputs = [out['loss'] for out in step_outputs]
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'train_loss' : ret_metrics['loss']
                  }
        self.log_dict(metrics, prog_bar=True)

    def validation_epoch_end(self, step_outputs):
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'val_loss' : ret_metrics['loss']
                  }
        self.log_dict(metrics, prog_bar=True)

    def _shared_epoch_end(self, step_outputs):
        loss = torch.tensor(step_outputs).mean()
        return {'loss' : loss}

    def configure_optimizers(self):
        if self.args.opt_type == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                     lr = self.args.lr,
                                     weight_decay = 1e-2)
        else:
            optimizer = torch.optim.Adam(self.parameters(), 
                                        lr = self.args.lr,
                                        weight_decay = 0)

        
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=self.args.trainer_max_epochs)
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

        celltype_root = f'{args.dataset_data_root}/{args.dataset_assembly}'
        genomic_features={}
        for i in range(len(args.epigenomic_features)):
            genomic_features[args.epigenomic_features[i]] = {'file_name' : f'{args.epigenomic_features[i]}',
                                             'norm' : args.feature_log[i]}
        print(genomic_features)
        save_path = f'{args.run_save_path}/input_data.txt'
        
        with open(save_path, 'w') as f:
            f.write(f'genomic_features: {genomic_features}\n')
            f.write(args.__str__())

        dataset = cell_dataset.CellLineDataset(
                                args.dataset_celltype,
                                celltype_root, 
                                args.dataset_assembly,
                                args.sample_length,
                                genomic_features, 
                                mode = mode,
                                include_sequence = True,
                                include_genomic_features = True,
                                resolution = args.resolution,
                                cool_res = args.cool_res,
                                target_treatment = args.target_treatment,
                                use_weight = args.use_weight,
                                clip_pc = args.clip_pc
                                )

        # Record length for printing validation image
        if mode == 'val':
            self.val_length = len(dataset) / args.dataloader_batch_size
            print('Validation loader length:', self.val_length)

        return dataset

    def get_dataloader(self, args, mode):
        dataset = self.get_dataset(args, mode)

        if mode == 'train':
            shuffle = True
        else: # validation and test settings
            shuffle = False
        
        batch_size = args.dataloader_batch_size
        num_workers = args.dataloader_num_workers

        if not args.dataloader_ddp_disabled:
            gpus = args.trainer_num_gpu
            batch_size = int(args.dataloader_batch_size / gpus)
            num_workers = int(args.dataloader_num_workers / gpus) 

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,

            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=True
        )
        return dataloader

    def get_model(self, args):
        model_name =  args.model_type
        num_genomic_features = len(args.epigenomic_features)
        ModelClass = getattr(corigami_models, model_name)
        model = ModelClass(num_genomic_features,args.sample_length,mid_hidden = 256, backbone = args.backbone,resolution = args.resolution)
        return model

if __name__ == '__main__':
    main()
