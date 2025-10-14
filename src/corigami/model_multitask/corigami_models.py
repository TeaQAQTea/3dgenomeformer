import torch
import torch.nn as nn

import blocks

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
    
<<<<<<< HEAD
    def __init__(self, num_genomic_features, mid_hidden = 256, record_attn = False,record_feature = False):
=======
    def __init__(self, num_genomic_features, mid_hidden = 256, record_attn = False ,record_feature = False):
>>>>>>> backup/old-main
        super(ConvTransModel, self).__init__(num_genomic_features)
        print('Initializing ConvTransModel')
        self.encoder = blocks.EncoderSplit(num_genomic_features, output_size = mid_hidden, num_blocks = 12)
        self.attn = blocks.AttnModule(hidden = mid_hidden, record_attn = record_attn)
<<<<<<< HEAD
        self.diDecoder = blocks.diDecoder(mid_hidden,save_features=record_feature)
=======
        self.diDecoder = blocks.diDecoder(mid_hidden ,save_features=record_feature)
>>>>>>> backup/old-main
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
<<<<<<< HEAD
        if self.record_feature:  
            di,out_feature = self.diDecoder(feature)
=======

        if self.record_feature:
            di, out_features = self.diDecoder(feature)
>>>>>>> backup/old-main
        else:
            di = self.diDecoder(feature)

        x = self.diagonalize(feature)
        x = self.decoder(x).squeeze(1)
        # print(x.shape)
        if self.record_attn:
            if self.record_feature:
<<<<<<< HEAD
                return x,di, attn_weights,out_feature
=======
                return x,di, attn_weights, out_features
>>>>>>> backup/old-main
            else:
                return x,di, attn_weights
        else:
            if self.record_feature:
<<<<<<< HEAD
                return x,di,out_feature
=======
                return x,di, out_features
>>>>>>> backup/old-main
            else:
                return x,di

if __name__ == '__main__':
    main()
