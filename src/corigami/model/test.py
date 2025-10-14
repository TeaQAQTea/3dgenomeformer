import sys
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks

import corigami.model.corigami_models as corigami_models
<<<<<<< HEAD
import genome_dataset
from torchinfo import summary
import os
=======
import corigami.data.genome_dataset as genome_dataset
from torchinfo import summary
import os
import h5py
>>>>>>> backup/old-main


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
<<<<<<< HEAD
=======
        self.output_path =f'{self.args.run_save_path}/output.h5'

        if os.path.exists(self.output_path):
            os.remove(self.output_path)
>>>>>>> backup/old-main

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
<<<<<<< HEAD

    def test_step(self, batch, batch_idx):
        # print(self._shared_eval_step(batch, batch_idx)['loss'])
        ret_metrics = {'loss':self._shared_eval_step(batch, batch_idx)['loss'],
                       'target':self._shared_eval_step(batch, batch_idx)['target'],
                       'predict':self._shared_eval_step(batch, batch_idx)['predict']}
        start=self._shared_eval_step(batch, batch_idx)['start']
        end=self._shared_eval_step(batch, batch_idx)['end']
        chr_name=self._shared_eval_step(batch, batch_idx)['chr_name']
        chr_idx=self._shared_eval_step(batch, batch_idx)['chr_idx']
        attention=self._shared_eval_step(batch, batch_idx)['attention']
        # print(ret_metrics)
        return ret_metrics,start,end,chr_name,chr_idx,attention
=======
    def test_step(self, batch, batch_idx):
        # print(self._shared_eval_step(batch, batch_idx)['loss'])
        eval_result = self._shared_eval_step(batch, batch_idx)
        ret_metrics = {'loss': eval_result['loss']}
        target = eval_result['target']
        predict = eval_result['predict']
        start = eval_result['start']
        end = eval_result['end']
        # chr_name = eval_result['chr_name']
        chr_idx = eval_result['chr_idx']
        attention = eval_result['attention']

        data={}
        data['target']=target.detach().cpu().numpy()
        data['predict']=predict.detach().cpu().numpy()
        data['end']=end.detach().cpu().numpy()
        data['start']=start.detach().cpu().numpy()
        # data['chr_name']=chr_name.detach().cpu().numpy()
        data['chr_idx']=chr_idx.detach().cpu().numpy()  
        data['attention']=attention.detach().cpu().numpy()
        #把attention的1 2 维度换一下
        data['attention']=np.transpose(data['attention'],(1,0,2,3))

        with h5py.File(self.output_path, 'a') as f:
            for key in data:
                if key not in f:
            # Create dataset if not exists with unlimited size and chunking
                    maxshape = (None,) + data[key].shape[1:]
                    f.create_dataset(key, data=data[key], maxshape=maxshape, chunks=True)
                else:
                    # Resize dataset and append new data
                    f[key].resize(f[key].shape[0] + data[key].shape[0], axis=0)
                    f[key][-data[key].shape[0]:] = data[key]
        # print(ret_metrics)
        return ret_metrics
>>>>>>> backup/old-main

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

    def test_epoch_end(self, step_outputs):
        step_outputs_dict = {}
<<<<<<< HEAD
        print(step_outputs)
        step_outputs_dict['loss'] = [out[0]['loss'] for out in step_outputs]
        # step_outputs_dict['pearson_r'] = [out['pearson_r'] for out in step_outputs]
        step_outputs_dict['target'] = [out[0]['target'] for out in step_outputs]
        step_outputs_dict['predict'] = [out[0]['predict'] for out in step_outputs]
        start=[out[1] for out in step_outputs]
        start=torch.cat(start,dim=0)
        end=[out[2] for out in step_outputs]
        end=torch.cat(end,dim=0)
        chr_name=[out[3] for out in step_outputs]
        chr_name=torch.cat(chr_name,dim=0)
        attention=[out[5] for out in step_outputs]
        attention=torch.cat(attention,dim=1)
        
        # print(step_outputs_dict['pearsonr'])
        ret_metrics = self._shared_epoch_end(step_outputs_dict)
        # print(ret_metrics)
        #ret_metrics转到cpu
        # # final_metrics = {'test_loss' : ret_metrics['loss']}
        # ret_metrics['target']=ret_metrics['target'].cpu()
        # ret_metrics['predict']=ret_metrics['predict'].cpu()
        print(start)
        import h5py

        output_path =f'{self.args.run_save_path}/output.h5'

        if os.path.exists(output_path):
            os.remove(output_path)

        with h5py.File(output_path, 'w') as f:
            f.create_dataset(data=ret_metrics['target'].detach().cpu().numpy(), name='target')
            f.create_dataset(data=ret_metrics['predict'].detach().cpu().numpy(), name='predict')
            f.create_dataset(data=start.detach().cpu().numpy(), name='start')
            f.create_dataset(data=end.detach().cpu().numpy(), name='end')
            f.create_dataset(data=chr_name.detach().cpu().numpy(), name='chr_name')
            f.create_dataset(data=attention.detach().cpu().numpy(), name='attention')
        print(f'Output saved to {output_path}')
        
        # return ret_metrics
        
        # metrics = {'test_loss' : ret_metrics['loss'],'test_pearsonr':ret_metrics['pearson_r']}
        # self.log_dict(metrics, prog_bar=True, on_epoch=True)

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
=======
    # def test_step(self, batch, batch_idx):
    #     # print(self._shared_eval_step(batch, batch_idx)['loss'])
    #     ret_metrics = {'loss':self._shared_eval_step(batch, batch_idx)['loss'],
    #                    'target':self._shared_eval_step(batch, batch_idx)['target'],
    #                    'predict':self._shared_eval_step(batch, batch_idx)['predict']}
    #     start=self._shared_eval_step(batch, batch_idx)['start']
    #     end=self._shared_eval_step(batch, batch_idx)['end']
    #     chr_name=self._shared_eval_step(batch, batch_idx)['chr_name']
    #     chr_idx=self._shared_eval_step(batch, batch_idx)['chr_idx']
    #     attention=self._shared_eval_step(batch, batch_idx)['attention']
    #     # print(ret_metrics)
    #     return ret_metrics,start,end,chr_name,chr_idx,attention

    # def _shared_eval_step(self, batch, batch_idx):
    #     inputs, mat,start, end, chr_name, chr_idx = self.proc_batch(batch)
    #     outputs,attention = self(inputs)
    #     criterion = torch.nn.MSELoss()
    #     loss = criterion(outputs, mat)
    #     # mean_mat=torch.mean(mat,dim=(1,2))
    #     # mean_outputs=torch.mean(outputs,dim=(1,2))
    #     # std_mat=torch.std(mat,dim=(1,2))
    #     # std_outputs=torch.std(outputs,dim=(1,2))
    #     # index_not_zero=index_not_zero = torch.nonzero(torch.logical_and( std_mat!= 0, std_outputs != 0)).squeeze()
    #     # pearson_r=[]

    #     # if len(index_not_zero)==0:
    #     #     pearson_r=torch.tensor(0)
    #     # else:   
    #     #     for index in index_not_zero:
    #     #         diff_mat=mat[index,:,:]-mean_mat[index]
    #     #         diff_outputs=outputs[index,:,:]-mean_outputs[index]
    #     #         diff_prod=torch.sum(diff_mat*diff_outputs,dim=(0,1))/(torch.numel(diff_mat)-1)
    #     #         pearson_r_per_sample=diff_prod/(std_mat[index]*std_outputs[index])
    #     #         pearson_r.append(pearson_r_per_sample)
    #     #     pearson_r=torch.stack(pearson_r).mean()

    #     return {'loss' : loss,'target':mat,'predict':outputs,'start':start,'end':end,'chr_name':chr_name,'chr_idx':chr_idx,'attention':attention}

    # # Collect epoch statistics
    # def training_epoch_end(self, step_outputs):
    #     step_outputs = [out['loss'] for out in step_outputs]
    #     ret_metrics = self._shared_epoch_end(step_outputs)
    #     metrics = {'train_loss' : ret_metrics['loss']}
    #     self.log_dict(metrics, prog_bar=True)

    # def validation_epoch_end(self, step_outputs):
    #     ret_metrics = self._shared_epoch_end(step_outputs)
    #     metrics = {'val_loss' : ret_metrics['loss']
    #               }
    #     self.log_dict(metrics, prog_bar=True)

    # def test_epoch_end(self, step_outputs):
    #     step_outputs_dict = {}
    #     print(step_outputs)
    #     step_outputs_dict['loss'] = [out[0]['loss'] for out in step_outputs]
    #     # step_outputs_dict['pearson_r'] = [out['pearson_r'] for out in step_outputs]
    #     step_outputs_dict['target'] = [out[0]['target'] for out in step_outputs]
    #     step_outputs_dict['predict'] = [out[0]['predict'] for out in step_outputs]
    #     start=[out[1] for out in step_outputs]
    #     start=torch.cat(start,dim=0)
    #     end=[out[2] for out in step_outputs]
    #     end=torch.cat(end,dim=0)
    #     chr_name=[out[3] for out in step_outputs]
    #     chr_name=torch.cat(chr_name,dim=0)
    #     attention=[out[5] for out in step_outputs]
    #     attention=torch.cat(attention,dim=1)
        
    #     # print(step_outputs_dict['pearsonr'])
    #     ret_metrics = self._shared_epoch_end(step_outputs_dict)
    #     # print(ret_metrics)
    #     #ret_metrics转到cpu
    #     # # final_metrics = {'test_loss' : ret_metrics['loss']}
    #     # ret_metrics['target']=ret_metrics['target'].cpu()
    #     # ret_metrics['predict']=ret_metrics['predict'].cpu()
    #     print(start)
    #     import h5py

    #     output_path =f'{self.args.run_save_path}/output.h5'

    #     if os.path.exists(output_path):
    #         os.remove(output_path)

    #     with h5py.File(output_path, 'w') as f:
    #         f.create_dataset(data=ret_metrics['target'].detach().cpu().numpy(), name='target')
    #         f.create_dataset(data=ret_metrics['predict'].detach().cpu().numpy(), name='predict')
    #         f.create_dataset(data=start.detach().cpu().numpy(), name='start')
    #         f.create_dataset(data=end.detach().cpu().numpy(), name='end')
    #         f.create_dataset(data=chr_name.detach().cpu().numpy(), name='chr_name')
    #         f.create_dataset(data=attention.detach().cpu().numpy(), name='attention')
    #     print(f'Output saved to {output_path}')
        
    #     # return ret_metrics
        
    #     # metrics = {'test_loss' : ret_metrics['loss'],'test_pearsonr':ret_metrics['pearson_r']}
    #     # self.log_dict(metrics, prog_bar=True, on_epoch=True)

    # def _shared_epoch_end(self, step_outputs):
    #     # print(self.args.run_save_path)
    #     loss = torch.tensor(step_outputs['loss']).mean()  
    #     #去除0
    #     # pearsonr=torch.tensor(step_outputs['pearson_r'])
    #     target=torch.cat(step_outputs['target'],dim=0)
    #     # print(target.shape)
    #     predict=torch.cat(step_outputs['predict'],dim=0)
    #     # nonzero_indices = torch.nonzero(pearsonr).squeeze()
    #     # pearsonr= pearsonr[nonzero_indices].mean()
    #     # pearsonr = torch.tensor(step_outputs['pearson_r'])
    #     # pearsonr=pearsonr.detach()
    #     # print({'loss' : loss,'target':target,'predict':predict})
    #     return {'loss' : loss,'target':target,'predict':predict}
>>>>>>> backup/old-main

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
