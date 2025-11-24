import sys
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks

import corigami_models as corigami_models
import genome_dataset
from torchinfo import summary
import os


def main():
    args = init_parser()
    init_test(args)

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
  parser.add_argument('--num-gpu', dest='trainer_num_gpu', default=4,
                        type=int,
                        help='Number of GPUs to use')
  parser.add_argument('--epigenomic-features', nargs='+', dest='epigenomic_features', default=['ctcf_log2fc', 'ro'],)

  parser.add_argument('--log',dest='feature_log', nargs='+',default=['log','log'])
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
  parser.add_argument('--gpu-number', dest='gpu', default=3,
                        type=str,
                        help='gpu id')
  parser.add_argument('--model-path', dest='model_path', default=None,
                        type=str,
                        help='model path')


  args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
  assert len(args.epigenomic_features) == len(args.feature_log), 'Length of epigenomic features and log normalization must be the same'
  args.feature_num=len(args.epigenomic_features)

  for i in range(len(args.feature_log)):
    if args.feature_log[i]=='None':
            args.feature_log[i]=None
  return args

def init_test(args):

    # Early_stopping

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #时间戳
    import time
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S_test", time.localtime())


    # Logger
    #使用tensorboard记录
    print("aaaaaaaa")
    print(args.run_save_path)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir = f'{args.run_save_path}/tb')
    
    pl_module = TrainModule(args)
    checkpoint_path=args.model_path
    # pl_module = pl_module.load_from_checkpoint(checkpoint_path)
    testloader = pl_module.get_dataloader(args, 'test')
    
    model=pl_module.load_from_checkpoint(checkpoint_path, args=args)
    trainer = pl.Trainer(gpus=args.trainer_num_gpu, logger=tb_logger)
    output=trainer.test(model, testloader)
    print(output)
    # #保存成h5
    # import h5py
    # output_path = f'{args.run_save_path}/output_{timestamp}.h5'
    # with h5py.File(output_path, 'w') as f:
    #     f.create_dataset(data=output, name='output')
    # print(f'Output saved to {output_path}')



class TrainModule(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.model = self.get_model(args)
        # summary(self.model,input_size=(4, 2097152, 5+args.feature_num))
        #保存为txt
        self.args = args
        print(self.args.run_save_path)
        print(self.args)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def proc_batch(self, batch):
        seq, features, mat, start, end, chr_name, chr_idx = batch
        features = torch.cat([feat.unsqueeze(2) for feat in features], dim = 2)
        inputs = torch.cat([seq, features], dim = 2)
        mat = mat.float()
        return inputs, mat, start, end, chr_name, chr_idx
    
    def training_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        # print(inputs.shape)
        # print("mat.shape",mat.shape)
        outputs = self(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)

        metrics = {'train_step_loss': loss}
        self.log_dict(metrics, batch_size = inputs.shape[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)
        return ret_metrics



    def _shared_eval_step(self, batch, batch_idx):
        inputs, mat,start, end, chr_name, chr_idx = self.proc_batch(batch)
        outputs,attention = self(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)
        # mean_mat=torch.mean(mat,dim=(1,2))
        # mean_outputs=torch.mean(outputs,dim=(1,2))
        # std_mat=torch.std(mat,dim=(1,2))
        # std_outputs=torch.std(outputs,dim=(1,2))
        # index_not_zero=index_not_zero = torch.nonzero(torch.logical_and( std_mat!= 0, std_outputs != 0)).squeeze()
        # pearson_r=[]

        # if len(index_not_zero)==0:
        #     pearson_r=torch.tensor(0)
        # else:   
        #     for index in index_not_zero:
        #         diff_mat=mat[index,:,:]-mean_mat[index]
        #         diff_outputs=outputs[index,:,:]-mean_outputs[index]
        #         diff_prod=torch.sum(diff_mat*diff_outputs,dim=(0,1))/(torch.numel(diff_mat)-1)
        #         pearson_r_per_sample=diff_prod/(std_mat[index]*std_outputs[index])
        #         pearson_r.append(pearson_r_per_sample)
        #     pearson_r=torch.stack(pearson_r).mean()

        return {'loss' : loss,'target':mat,'predict':outputs,'start':start,'end':end,'chr_name':chr_name,'chr_idx':chr_idx,'attention':attention}

    # Collect epoch statistics
    def training_epoch_end(self, step_outputs):
        step_outputs = [out['loss'] for out in step_outputs]
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'train_loss' : ret_metrics['loss']}
        self.log_dict(metrics, prog_bar=True)

    def validation_epoch_end(self, step_outputs):
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'val_loss' : ret_metrics['loss']
                  }
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_idx):
        import os, h5py, numpy as np, torch

        r = self._shared_eval_step(batch, batch_idx)

        # 标量
        loss_val = r['loss'].detach().cpu().item()

        # 转 CPU + numpy
        pred_np  = r['predict'].detach().cpu().numpy()            # (B, H, W)
        tgt_np   = r['target'].detach().cpu().numpy()             # (B, H, W)
        start_np = r['start'].detach().cpu().numpy().reshape(-1)  # (B,)
        end_np   = r['end'].detach().cpu().numpy().reshape(-1)    # (B,)
        chr_np   = r['chr_name'].detach().cpu().numpy().reshape(-1)  # (B,)
        print(type(r['chr_name']))
        print(r['chr_name'])
        # attention 规范化到 (B, H, W)
        att_np = r['attention']
        if att_np is not None:
            att_np = att_np.detach().cpu().numpy()
            if att_np.ndim == 2:
                att_np = att_np[None, ...]                # (1, H, W)
            elif att_np.ndim > 3:
                tail = att_np.shape[-2:]                  # (H, W)
                att_np = att_np.reshape(-1, *tail)        # (B*, H, W)

        # ========= 首次调用：初始化 HDF5 和数据集 =========
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
            self._d_chrname = self._h5.create_dataset("chr_name", shape=(0,), maxshape=(None,), dtype="i8", chunks=True)

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

            # attention（若有）：按 (B, H, W) 追加
            self._d_att = None
            self._att_tail = None
            if att_np is not None:
                self._att_tail = att_np.shape[1:]   # (H, W)
                self._d_att = self._h5.create_dataset(
                    "attention", shape=(0, *self._att_tail), maxshape=(None, *self._att_tail),
                    dtype="f4", chunks=(1, *self._att_tail)
                )

            # 写指针（样本数累计）
            self._ptr = 0
            self._att_ptr = 0

        # 若首次没遇到 attention，但当前 batch 有 -> 现在创建
        if (att_np is not None) and (getattr(self, "_d_att", None) is None):
            self._att_tail = att_np.shape[1:]
            self._d_att = self._h5.create_dataset(
                "attention", shape=(0, *self._att_tail), maxshape=(None, *self._att_tail),
                dtype="f4", chunks=(1, *self._att_tail)
            )
            self._att_ptr = 0

        # ========= 追加写入 =========
        B = pred_np.shape[0]
        s, e = self._ptr, self._ptr + B

        self._d_start.resize((e,));   self._d_start[s:e]   = start_np
        self._d_end.resize((e,));     self._d_end[s:e]     = end_np
        self._d_chrname.resize((e,)); self._d_chrname[s:e] = chr_np

        self._d_pred.resize((e, *self._d_pred.shape[1:])); self._d_pred[s:e, ...] = pred_np
        self._d_tgt.resize((e,  *self._d_tgt.shape[1:]));  self._d_tgt[s:e,  ...] = tgt_np

        self._ptr = e

        # attention 追加
        if (att_np is not None) and (self._d_att is not None):
            B_att = att_np.shape[0]
            sa, ea = self._att_ptr, self._att_ptr + B_att
            self._d_att.resize((ea, *self._att_tail))
            self._d_att[sa:ea, ...] = att_np
            self._att_ptr = ea

        # 释放显存
        try: torch.cuda.empty_cache()
        except: pass

        # 只返回loss，避免把大张量传回环路
        return {"loss": loss_val}

    def test_epoch_end(self, step_outputs):
        import numpy as np

        # 现在 step_outputs 里只有 {'loss': 标量}，不再累积大张量
        losses = [o["loss"] for o in step_outputs]
        mean_loss = float(np.mean(losses)) if len(losses) > 0 else float("nan")
        self.log("test_loss", mean_loss, prog_bar=True)
        print(f"[test] mean loss = {mean_loss:.6f}")

        # 关闭 HDF5
        if hasattr(self, "_h5") and self._h5 is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None
            print(f"Output saved to {self._h5_path}")

    def _shared_epoch_end(self, step_outputs):
        # print(self.args.run_save_path)
        loss = torch.tensor(step_outputs['loss']).mean()  
        #去除0
        # pearsonr=torch.tensor(step_outputs['pearson_r'])
        target=torch.cat(step_outputs['target'],dim=0)
        # print(target.shape)
        predict=torch.cat(step_outputs['predict'],dim=0)
        # nonzero_indices = torch.nonzero(pearsonr).squeeze()
        # pearsonr= pearsonr[nonzero_indices].mean()
        # pearsonr = torch.tensor(step_outputs['pearson_r'])
        # pearsonr=pearsonr.detach()
        # print({'loss' : loss,'target':target,'predict':predict})
        return {'loss' : loss,'target':target,'predict':predict}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr = 2e-4,
                                     weight_decay = 0)

        import pl_bolts
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

        celltype_root = f'{args.dataset_data_root}/{args.dataset_assembly}/{args.dataset_celltype}'
        genomic_features={}
        for i in range(len(args.epigenomic_features)):
            genomic_features[args.epigenomic_features[i]] = {'file_name' : f'{args.epigenomic_features[i]}',
                                             'norm' : args.feature_log[i]}
        print(genomic_features)
            
        # genomic_features = {'ctcf_log2fc' : {'file_name' : 'CTCF_K562_hg38.bigWig.bw',
        #                                      'norm' : 'log'},
        #                     'ro' : {'file_name' : 'kas_k562_hg38_final.bigwig',
        #                                      'norm' :None}}
        dataset = genome_dataset.GenomeDataset(celltype_root, 
                                args.dataset_assembly,
                                genomic_features, 
                                mode = mode,
                                include_sequence = True,
                                include_genomic_features = True)

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
        num_genomic_features = args.feature_num
        ModelClass = getattr(corigami_models, model_name)
        model = ModelClass(num_genomic_features, mid_hidden = 256,record_attn=True)
        return model

if __name__ == '__main__':
    main()
