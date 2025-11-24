import sys 
import os
import random
import pickle
import pandas as pd
import numpy as np

from skimage.transform import resize
from torch.utils.data import Dataset

import data_feature as data_feature

class ChromosomeDataset(Dataset):
    '''
    Dataloader that provide sequence, features, and HiC data. Assume input
    folder strcuture.

    Args:
        data_root (str): Directory including sequence features, and HiC matrix 
            as subdirectories.
        chr_name (str): Name of the represented chromosome (e.g. chr1)
            as ``root/DNA/chr1/DNA`` for DNA as an example.
        omit_regions (list of tuples): start and end of excluded regions
    '''
    def __init__(self, celltype_root, chr_name, omit_regions, feature_list, use_aug = True, mode = 'train', finetune_regions = None, resolution = 4096, cool_res = 5000, target_treatment='clip',use_weight=False, clip_pc=95):

        sample_length = 2097152 # 2Mbp
        self.use_aug = use_aug
        self.res = cool_res # 5000 or 10000
        self.bins = sample_length / cool_res # 209.7152 bins 2097152 bp
        self.image_scale = sample_length//resolution # IMPORTANT, scale 210 to 256
        self.treatment = target_treatment
        self.use_weight = use_weight
        self.clip_pc = clip_pc
        if mode == 'finetune':
            self.sample_bins = self.image_scale + 20
            self.stride = self.image_scale // 20
        else:
            self.sample_bins = self.image_scale*2
            self.stride = self.image_scale // 5
        self.chr_name = chr_name

        print(f'Loading chromosome {chr_name}...')

        self.seq = data_feature.SequenceFeature(path = f'{celltype_root}/../dna_sequence/{chr_name}.fa.gz')
        self.genomic_features = feature_list
        if mode == 'finetune':
            self.mat = data_feature.HiCFeature(path = f'{celltype_root}/finetune_hic_matrix/{chr_name}.npz')
        else:
            if cool_res == 3000:
                self.mat = data_feature.HiCFeature(path = f'{celltype_root}/hic_matrix_3000_1024/{chr_name}.npz')
            elif cool_res == 10000:
                self.mat = data_feature.HiCFeature(path = f'{celltype_root}/hic_matrix/{chr_name}.npz')
            elif cool_res == 5000:
                self.mat = data_feature.HiCFeature(path = f'{celltype_root}/hic_matrix_5000_512/{chr_name}.npz')
            else:
                raise Exception('cool_res should be 3000, 5000 or 10000')

        self.omit_regions = omit_regions
        self.check_length() # Check data length
        self.all_intervals = self.get_active_intervals(mode, finetune_regions)
        if mode == 'test':
            self.intervals = self.all_intervals
        else:
            self.intervals = self.filter(self.all_intervals, omit_regions)

    def __getitem__(self, idx):
        start, end = self.intervals[idx]
        target_size = int(self.bins * self.res)

        # Shift Augmentations
        if self.use_aug: 
            start, end = self.shift_aug(target_size, start, end)
        else:
            start, end = self.shift_fix(target_size, start, end)
        seq, features, mat, weight = self.get_data_at_interval(start, end)

        if self.use_aug:
            # Extra on sequence
            seq = self.gaussian_noise(seq, 0.1)
            # Genomic features
            features = [self.gaussian_noise(item, 0.1) for item in features]
            # Reverse complement all data
            seq, features, mat = self.reverse(seq, features, mat)

        return seq, features, mat, start, end, weight

    def __len__(self):
        return len(self.intervals)

    def gaussian_noise(self, inputs, std = 1):
        noise = np.random.randn(*inputs.shape) * std
        outputs = inputs + noise
        return outputs

    def reverse(self, seq, features, mat, chance = 0.5):
        '''
        Reverse sequence and matrix
        '''
        r_bool = np.random.rand(1)
        if r_bool < chance:
            seq_r = np.flip(seq, 0).copy() # n x 5 shape
            features_r = [np.flip(item, 0).copy() for item in features] # n
            mat_r = np.flip(mat, [0, 1]).copy() # n x n

            # Complementary sequence
            seq_r = self.complement(seq_r)
        else:
            seq_r = seq
            features_r = features
            mat_r = mat
        return seq_r, features_r, mat_r

    def complement(self, seq, chance = 0.5):
        '''
        Complimentary sequence
        '''
        r_bool = np.random.rand(1)
        if r_bool < chance:
            seq_comp = np.concatenate([seq[:, 1:2],
                                       seq[:, 0:1],
                                       seq[:, 3:4],
                                       seq[:, 2:3],
                                       seq[:, 4:5]], axis = 1)
        else:
            seq_comp = seq
        return seq_comp

    def get_data_at_interval(self, start, end,cool_res = 5000):
        '''
        Slice data from arrays with transformations
        '''
        # Sequence processing
        seq = self.seq.get(start, end)
        # Features processing
        features = [item.get(self.chr_name, start, end) for item in self.genomic_features]
        # Hi-C matrix processing
        mat = self.mat.get(start, res = cool_res)

        mat = resize(mat, (self.image_scale, self.image_scale), anti_aliasing=True)
    
        #设置截断值
        #取0.85分位数
        if self.treatment == 'linear':
            mat = mat*10
            mat = np.log(mat + 10)

        elif self.treatment == 'clip':
            mat = np.log(mat + 1)
            percentile = np.percentile(mat, self.clip_pc)
            mat = np.clip(mat, 0, percentile)



        if self.use_weight:
            weight = self.get_weight(mat)
        else:
            weight = np.ones(mat.shape)
        return seq, features, mat, weight

    def get_weight(self, mat):
        # 计算分位数阈值
        q70 = np.nanpercentile(mat, 70)
        q80 = np.nanpercentile(mat, 80)

        # 初始化全1权重矩阵
        weight = np.ones_like(mat, dtype=float)

        # 构造mask：处于70~80分位之间的元素
        mask = (mat >= q70) & (mat <= q80)

        # 提高这部分区域的权重
        weight[mask] = 5.0

        return weight
    
    def get_active_intervals(self, mode, finetune_regions=None):
        '''
        Get intervals for sample data: [[start, end]]
        '''
        if mode == 'finetune' and finetune_regions is not None:
            fr = np.array(finetune_regions, dtype=np.int64).reshape(-1, 2)
            print(fr)
            all_intervals = []
            for r_start, r_end in fr:
                region_bins = (r_end - r_start) // self.res
                print(region_bins)
                if region_bins < self.sample_bins:
                    continue
                data_size = (region_bins - self.sample_bins) // self.stride + 1
                if data_size <= 0:
                    continue
                starts = np.arange(0, data_size).reshape(-1, 1) * self.stride
                intervals_bin = np.append(starts, starts + self.sample_bins, axis=1)
                intervals_bp  = intervals_bin * self.res + r_start  # 迁移回基因组坐标
                all_intervals.append(intervals_bp)
            if len(all_intervals) == 0:
                return np.empty((0, 2), dtype=int)
            intervals = np.vstack(all_intervals)
        else:
            chr_bins = len(self.seq) / self.res
            data_size = (chr_bins - self.sample_bins) / self.stride
            starts = np.arange(0, data_size).reshape(-1, 1) * self.stride
            intervals_bin = np.append(starts, starts + self.sample_bins, axis=1)
            intervals = intervals_bin * self.res
        return intervals.astype(int)

    def filter(self, intervals, omit_regions):
        valid_intervals = []
        for start, end in intervals: 
            # Way smaller than omit or way larger than omit
            start_cond = start <= omit_regions[:, 1]
            end_cond = omit_regions[:, 0] <= end
            #import pdb; pdb.set_trace()
            if sum(start_cond * end_cond) == 0:
                valid_intervals.append([start, end])
        return valid_intervals
    
    
    def encode_seq(self, seq):
        ''' 
        encode dna to onehot (n x 5)
        '''
        seq_emb = np.zeros((len(seq), 5))
        seq_emb[np.arange(len(seq)), seq] = 1
        return seq_emb

    def shift_aug(self, target_size, start, end):
        '''
        All unit are in basepairs
        '''
        offset = random.choice(range(end - start - target_size))
        return start + offset , start + offset + target_size

    def shift_fix(self, target_size, start, end):
        offset = 0
        return start + offset , start + offset + target_size

    def check_length(self):
        assert len(self.seq.seq) == self.genomic_features[0].length(self.chr_name), f'Sequence {len(self.seq)} and First feature {self.genomic_features[0].length(self.chr_name)} have different length.' 
        assert abs(len(self.seq) / self.res -  len(self.mat)) < 2, f'Sequence {len(self.seq) / self.res} and Hi-C {len(self.mat)} have different length.' 

def get_feature_list(root_dir, feat_dicts):
    '''
    Args:
        features: a list of dicts with 
            1. file name
            2. norm status
    Returns:
        feature_list: a list of genomic features (bigwig files)
    '''
    feat_list = []
    for feat_item in feat_dicts:
        file_name = feat_item['file_name']
        file_path = f'{root_dir}/{file_name}'
        norm = feat_item['norm']
        feat_list.append(data_feature.GenomicFeature(file_path, norm))
    return feat_list

def proc_centrotelo(bed_dir):
    ''' Take a bed file indicating location, output a dictionary of items 
    by chromosome which contains a list of 2 value lists (range of loc)
    '''
    df = pd.read_csv(bed_dir , sep = '\t', names = ['chr', 'start', 'end'])
    chrs = df['chr'].unique()
    centrotelo_dict = {}
    for chr_name in chrs:
        sub_df = df[df['chr'] == chr_name]
        regions = sub_df.drop('chr', axis = 1).to_numpy()
        centrotelo_dict[chr_name] = regions
    return centrotelo_dict

if __name__ == '__main__':
    main()
