import torch
import torch.nn as nn

import blocks
from mamba import MambaBlock
from mamba import MambaConfig

class ConvModel(nn.Module):
    def __init__(self, num_genomic_features, mid_hidden = 256):
        super(ConvModel, self).__init__()
        print('Initializing ConvModel')
        self.encoder = blocks.EncoderSplit(num_genomic_features, output_size = mid_hidden, num_blocks = 12)
        self.decoder = blocks.Decoder(mid_hidden * 2)


    def forward(self, x):
        '''
        Input feature:
        batch_size, length * res, feature_dim
        '''
        x = self.move_feature_forward(x).float()
        x = self.encoder(x)
        x = self.diagonalize(x)
        x = self.decoder(x).squeeze(1)
        return x

    def move_feature_forward(self, x):
        '''
        input dim:
        bs, img_len, feat
        to: 
        bs, feat, img_len
        '''
        return x.transpose(1, 2).contiguous()

    def diagonalize(self, x):
        x_i = x.unsqueeze(2).repeat(1, 1, 256, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, 256)
        input_map = torch.cat([x_i, x_j], dim = 1)
        return input_map


class ConvTransModel(ConvModel):
    
    def __init__(self, num_genomic_features, mid_hidden = 256, record_attn = False,record_feature = False):
        super(ConvTransModel, self).__init__(num_genomic_features)
        print('Initializing ConvTransModel')
        self.encoder = blocks.EncoderSplit(num_genomic_features, output_size = mid_hidden, num_blocks = 12)
        config = MambaConfig(
            d_model=mid_hidden,        # 输入特征维度
            n_layers=3,                # 单层 MambaBlock
            d_state=16,                # 状态空间维度
            expand_factor=2,           # 扩展因子
            dt_init="random",          # 时间步初始化
            pscan=True,                # 使用并行扫描
            use_cuda=True           # 是否使用 CUDA
        )
        self.attn = MambaBlock(config)  # 用 MambaBlock 替换 AttnModule
        self.diDecoder = blocks.diDecoder(mid_hidden,save_features=record_feature)
        self.decoder = blocks.Decoder(mid_hidden * 2)
        self.record_attn = record_attn
        self.record_feature = record_feature
    
    def forward(self, x):
        '''
        Input feature:
        batch_size, length * res, feature_dim
        '''
        x = self.move_feature_forward(x).float()
        x = self.encoder(x)
        x = self.move_feature_forward(x)
        if self.record_attn:
            x, attn_weights = self.attn(x)
        else:
            x = self.attn(x)
        feature = self.move_feature_forward(x)
        if self.record_feature:  
            di,out_feature = self.diDecoder(feature)
        else:
            di = self.diDecoder(feature)

        x = self.diagonalize(feature)
        x = self.decoder(x).squeeze(1)
        # print(x.shape)
        if self.record_attn:
            if self.record_feature:
                return x,di, attn_weights,out_feature
            else:
                return x,di, attn_weights
        else:
            if self.record_feature:
                return x,di,out_feature
            else:
                return x,di

if __name__ == '__main__':
    main()
