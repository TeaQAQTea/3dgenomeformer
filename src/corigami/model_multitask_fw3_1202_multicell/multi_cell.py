import os
import pandas as pd
from torch.utils.data import Dataset
from genome_dataset import GenomeDataset  # 引用您的 GenomeDataset 类

class CellLineDataset(Dataset):
    '''
    Load data from multiple cell lines provided as a list.
    '''
    def __init__(self, cell_lines,  # Cell line names as a list
                 cell_lines_root, 
                 genome_assembly, 
                 feat_dicts,
                 mode='train', 
                 include_sequence=True, 
                 include_genomic_features=True, 
                 use_aug=True):
        self.cell_lines = cell_lines  # List of cell lines
        self.cell_lines_root = cell_lines_root
        self.include_sequence = include_sequence
        self.include_genomic_features = include_genomic_features
        self.use_aug = use_aug if mode == 'train' else False
        self.genome_assembly = genome_assembly
        self.feat_dicts = feat_dicts

        print(f"Using cell lines: {self.cell_lines}")

        # Create datasets for each cell line
        self.cell_line_datasets = self.load_cell_line_datasets(mode)
        self.lengths = [len(dataset) for dataset in self.cell_line_datasets]

        # Build a lookup table for dataset and index mapping
        self.ranges = self.get_ranges(self.lengths)

    def __getitem__(self, idx):
        """
        获取指定索引的细胞系数据，同时包括染色体和位置相关信息。
        返回:
        - outputs: 数据输出，包含序列、特征、矩阵等。
        - cell_line_name: 数据对应的细胞系名称。
        """
        # 确定索引所属的数据集和内部索引
        dataset_idx, dataset_item_idx = self.get_dataset_idx(idx)
        cell_line_name = self.cell_lines[dataset_idx]

        # 从对应的细胞系数据集中加载染色体数据
        chr_name, chr_idx = self.cell_line_datasets[dataset_idx].get_chr_idx(dataset_item_idx)
        seq, features, mat, di, start, end = self.cell_line_datasets[dataset_idx].chr_data[chr_name][chr_idx]

        # 处理染色体名称
        if chr_name[:3] == 'chr':
            chr_name = chr_name[3:]  # 移除 'chr' 前缀
        chr_name = int(chr_name)  # 转换为整数

        # 根据是否包含序列和/或基因组特征构造输出
        if self.include_sequence:
            if self.include_genomic_features:  # 同时包含序列和基因组特征
                outputs = seq, features, mat, di, start, end, chr_name, chr_idx
            else:  # 仅包含序列
                outputs = seq, mat, di, start, end, chr_name, chr_idx
        else:
            if self.include_genomic_features:  # 仅包含基因组特征
                outputs = features, mat, di, start, end, chr_name, chr_idx
            else:
                raise Exception('必须包含至少一个序列或基因组特征！')

        # 返回数据和对应的细胞系名称
        return outputs

    def __len__(self):
        return sum(self.lengths)

    def load_cell_line_datasets(self, mode):
        ''' Load genome datasets for each cell line in the list. '''
        cell_line_datasets = []
        for cell_line in self.cell_lines:
            cell_line_path = os.path.join(self.cell_lines_root, cell_line)
            if not os.path.exists(cell_line_path):
                raise FileNotFoundError(f"Cell line directory not found: {cell_line_path}")
            print(f"Loading data for cell line: {cell_line}")
            dataset = GenomeDataset(celltype_root=cell_line_path, 
                                    genome_assembly=self.genome_assembly, 
                                    feat_dicts=self.feat_dicts, 
                                    mode=mode, 
                                    include_sequence=self.include_sequence, 
                                    include_genomic_features=self.include_genomic_features, 
                                    use_aug=self.use_aug)
            cell_line_datasets.append(dataset)
        return cell_line_datasets

    def get_ranges(self, lengths):
        ''' Create a range mapping for each dataset. '''
        current_start = 0
        ranges = []
        for length in lengths:
            ranges.append([current_start, current_start + length - 1])
            current_start += length
        return ranges

    def get_dataset_idx(self, idx):
        ''' Find which dataset the idx belongs to. '''
        for i, (start, end) in enumerate(self.ranges):
            if start <= idx <= end:
                return i, idx - start
        raise IndexError(f"Index {idx} is out of range.")