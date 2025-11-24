import torch
import torch.nn as nn

import blocks as blocks
from mamba import MambaBlock
from mamba import MambaConfig 
import numpy as np
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

    def diagonalize(self, x ,length = 512):
        x_i = x.unsqueeze(2).repeat(1, 1, length, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, length)
        input_map = torch.cat([x_i, x_j], dim = 1)
        return input_map
 


class ConvTransModel(ConvModel):
    
    def __init__(self, num_genomic_features,sample_length, mid_hidden = 512, record_attn = False, backbone = 'Mamba', resolution = 4096):
        super(ConvTransModel, self).__init__(num_genomic_features)
        print('Initializing ConvTransModel')
        self.sample_length = sample_length
        self.resolution = resolution
        self.encoder = blocks.EncoderSplit(num_genomic_features, output_size = mid_hidden, num_blocks =np.log2(self.resolution).astype(int)-1) 
        # self.attn = blocks.AttnModule(hidden = mid_hidden, record_attn = record_attn)
        if backbone == 'Mamba':
            config = MambaConfig(
                d_model=mid_hidden,        
                n_layers=12,                
                d_state=64,                
                expand_factor=6,           
                dt_init="random",          
                pscan=True,                
                use_cuda=True           
            )
            self.attn = MambaBlock(config)
            record_attn = False
        else:
            self.attn = blocks.AttnModule(hidden = mid_hidden, record_attn = record_attn)
        self.decoder = blocks.Decoder(mid_hidden * 2)
        self.record_attn = record_attn
    
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
        x = self.move_feature_forward(x)
        x = self.diagonalize(x, length = self.sample_length//self.resolution)
        x = self.decoder(x).squeeze(1)
        # print(x.shape)
        if self.record_attn:
            return x, attn_weights
        else:
            return x

if __name__ == '__main__':
    main()
