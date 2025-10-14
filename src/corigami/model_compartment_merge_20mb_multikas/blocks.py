import torch
import torch.nn as nn
import numpy as np
import copy

<<<<<<< HEAD
<<<<<<< HEAD
=======

    

>>>>>>> backup/old-main
=======
>>>>>>> main-clean
class ConvBlock(nn.Module):
    def __init__(self, size, stride = 2, hidden_in = 64, hidden = 64):
        super(ConvBlock, self).__init__()
        pad_len = int(size / 2)
        self.scale = nn.Sequential(
                        nn.Conv1d(hidden_in, hidden, size, stride, pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        )
        self.res = nn.Sequential(
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled = self.scale(x)
        identity = scaled
        res_out = self.res(scaled)
        out = self.relu(res_out + identity)
        return out

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> main-clean
class convsplit(nn.Module):
    def __init__(self, in_channel, output_size = 256, filter_size = 5, num_blocks =5):
        super(convsplit, self).__init__()
        self.filter_size = filter_size
<<<<<<< HEAD
=======
class CrossAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
    
    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        query = query + attn_output
        return query

class convsplit(nn.Module):
    def __init__(self, in_channel, output_size=256, filter_size=5, num_blocks=5, num_heads=4, num_cross_attentions=3):
        super(convsplit, self).__init__()
        self.filter_size = filter_size
        self.num_heads = num_heads
        self.num_cross_attentions = num_cross_attentions  # Cross-Attention 块的数量
        
>>>>>>> backup/old-main
=======
>>>>>>> main-clean
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(5, 32, 3, 1, 1),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(),
                                    )
        self.conv_start_epi = nn.Sequential(
                                    nn.Conv1d(2, 32, 3, 1, 1),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(),
                                    )
        self.conv1_seq = nn.Sequential(
                                    nn.Conv1d(32, 64, 3, 1, 1),
                                    nn.BatchNorm1d(64),
                                    nn.ReLU(),
                                    )
        self.conv1_epi = nn.Sequential(
                                    nn.Conv1d(32, 64, 3, 1, 1),
                                    nn.BatchNorm1d(64),
                                    nn.ReLU(),
                                    )
        self.conv2_seq = nn.Sequential(
                                    nn.Conv1d(64, 128, 3, 1, 1),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    )
        self.conv2_epi = nn.Sequential(
                                    nn.Conv1d(64, 128, 3, 1, 1),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    )
<<<<<<< HEAD
<<<<<<< HEAD
        self.conv_seq = self.get_res_blocks(num_blocks,128)
        self.conv_epi = self.get_res_blocks(num_blocks,128)
=======
        self.conv_seq = self.get_res_blocks(num_blocks, 128)
        self.conv_epi = self.get_res_blocks(num_blocks, 128)

        # 多个 Cross-Attention 模块
        self.cross_attentions = nn.ModuleList([
            CrossAttention(embed_size=128, heads=self.num_heads) for _ in range(self.num_cross_attentions)
        ])

        # 最后的卷积层
>>>>>>> backup/old-main
=======
        self.conv_seq = self.get_res_blocks(num_blocks,128)
        self.conv_epi = self.get_res_blocks(num_blocks,128)
>>>>>>> main-clean
        self.conv_end = nn.Conv1d(256, 256, 1)
    
    def get_res_blocks(self, n, hidden):
        blocks = []
        for i in range(n):
            dilation = 2 ** (i + 1)
<<<<<<< HEAD
<<<<<<< HEAD
=======
            blocks.append(ResBlockDilated1D(3, hidden=hidden, dilation=dilation))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
    
    def forward(self, x):
        seq = x[:, :5, :]
        epi = x[:, 5:, :]
        
        # 初始卷积
        seq = self.conv_start_seq(seq)
        epi = self.conv_start_epi(epi)

        seq = self.conv1_seq(seq)
        epi = self.conv1_epi(epi)

        seq = self.conv2_seq(seq)
        epi = self.conv2_epi(epi)

        seq = self.conv_seq(seq)
        epi = self.conv_epi(epi)

        # 转换维度以适应 MultiheadAttention
        seq = seq.permute(0, 2, 1)  # (batch, seq_len, channels)
        epi = epi.permute(0, 2, 1)

        # 多次 Cross-Attention
        for i in range(self.num_cross_attentions):
            seq_attn = self.cross_attentions[i](query=seq, key=epi, value=epi)
            epi_attn = self.cross_attentions[i](query=epi, key=seq, value=seq)

            seq = seq_attn  # 更新 seq
            epi = epi_attn  # 更新 epi

        # 恢复原来的维度顺序
        seq_attn = seq.permute(0, 2, 1)
        epi_attn = epi.permute(0, 2, 1)

        # 拼接并通过最后的卷积层
        x = torch.cat([seq_attn, epi_attn], dim=1)
        out = self.conv_end(x)

        return out
    
    def get_res_blocks(self, n, hidden):
        blocks = []
        for i in range(n):
            dilation = 2 ** (i + 1)
>>>>>>> backup/old-main
=======
>>>>>>> main-clean
            blocks.append(ResBlockDilated1D(3, hidden = hidden, dilation = dilation))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
    def forward(self, x):
        seq = x[:, :5, :]
        epi = x[:, 5:, :]
        seq = self.conv_start_seq(seq)
        print(seq.shape)
        epi = self.conv_start_epi(epi)
        seq = self.conv1_seq(seq)
        epi = self.conv1_epi(epi)
        seq = self.conv2_seq(seq)
        epi = self.conv2_epi(epi)
        seq = self.conv_seq(seq)
        epi = self.conv_epi(epi)
        print(seq.shape, epi.shape)
        x = torch.cat([seq, epi], dim = 1)
        out = self.conv_end(x)
        return out
       
class ResBlockDilated1D(nn.Module):
    def __init__(self, size, hidden, stride=1, dilation=2):
        super(ResBlockDilated1D, self).__init__()
        pad_len = dilation * (size - 1) // 2  # Adjust padding to maintain sequence length
        self.res = nn.Sequential(
            nn.Conv1d(hidden, hidden, size, stride=stride, padding=pad_len, dilation=dilation),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, size, stride=stride, padding=pad_len, dilation=dilation),
            nn.BatchNorm1d(hidden),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        res_out = self.res(x)
        out = self.relu(res_out + identity)
        return out
    
class diDecoder(nn.Module):
    def __init__(self,hidden_dim,):
        super().__init__()
        self.cnn=nn.Sequential(
               nn.Conv1d(hidden_dim,hidden_dim,3,1,1),
               nn.BatchNorm1d(hidden_dim))
        num_blocks=5
        self.res_blocks = self.get_res_blocks(num_blocks, 256)
        self.conv1=nn.Sequential(
                nn.Conv1d(256,128,3,1,1),
                nn.BatchNorm1d(128),
                nn.ReLU())
        self.conv2=nn.Sequential(
                nn.Conv1d(128,64,3,1,1),
                nn.BatchNorm1d(64),
                nn.ReLU())
        self.conv3=nn.Sequential(
                nn.Conv1d(64,32,3,1,1),
                nn.BatchNorm1d(32),
                nn.ReLU())  
        self.fc3=nn.Linear(32,1)
        

    def forward(self, x):
        x=self.cnn(x)
        x=self.res_blocks(x)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=x.permute(0,2,1).contiguous()
        x=self.fc3(x).squeeze(2)
        # print(x.shape)
        return x
    
    def get_res_blocks(self, n, hidden):
        blocks = []
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResBlockDilated1D(3, hidden = hidden, dilation = dilation))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
    
class Encoder(nn.Module):
    def __init__(self, in_channel, output_size = 256, filter_size = 5, num_blocks = 12):
        super(Encoder, self).__init__()
        self.filter_size = filter_size
        self.conv_start = nn.Sequential(
                                    nn.Conv1d(in_channel, 32, 3, 2, 1),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(),
                                    )
        hiddens =        [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
        self.res_blocks = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        self.conv_end = nn.Conv1d(256, output_size, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        return out

    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks

class EncoderSplit(Encoder):
    def __init__(self, num_epi, output_size = 256, filter_size = 5, num_blocks = 12):
        super(Encoder, self).__init__()
        self.filter_size = filter_size
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(5, 16, 3, 2, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        self.conv_start_epi = nn.Sequential(
                                    nn.Conv1d(num_epi, 16, 3, 2, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        hiddens =        [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
        hiddens_half = (np.array(hiddens) / 2).astype(int)
        hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.conv_end = nn.Conv1d(256, output_size, 1)

    def forward(self, x):

        seq = x[:, :5, :]
        epi = x[:, 5:, :]
        seq = self.res_blocks_seq(self.conv_start_seq(seq))
        # print('seq', seq.shape)


        epi = self.res_blocks_epi(self.conv_start_epi(epi))
        # print('epi', epi.shape)

        x = torch.cat([seq, epi], dim = 1)
        out = self.conv_end(x)
        return out

class ResBlockDilated(nn.Module):
    def __init__(self, size, hidden = 64, stride = 1, dil = 2):
        super(ResBlockDilated, self).__init__()
        pad_len = dil 
        self.res = nn.Sequential(
                        nn.Conv2d(hidden, hidden, size, padding = pad_len, 
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        nn.ReLU(),
                        nn.Conv2d(hidden, hidden, size, padding = pad_len,
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x 
        res_out = self.res(x)
        out = self.relu(res_out + identity)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channel, hidden = 256, filter_size = 3, num_blocks = 5):
        super(Decoder, self).__init__()
        self.filter_size = filter_size

        self.conv_start = nn.Sequential(
                                    nn.Conv2d(in_channel, hidden, 3, 1, 1),
                                    nn.BatchNorm2d(hidden),
                                    nn.ReLU(),
                                    )
        self.res_blocks = self.get_res_blocks(num_blocks, hidden)
        self.conv_end = nn.Conv2d(hidden, 1, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        return out

    def get_res_blocks(self, n, hidden):
        blocks = []
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResBlockDilated(self.filter_size, hidden = hidden, dil = dilation))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks

class TransformerLayer(torch.nn.TransformerEncoderLayer):
    # Pre-LN structure
    
    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        # MHA section
        src_norm = self.norm1(src)
        src_side, attn_weights = self.self_attn(src_norm, src_norm, src_norm, 
                                    attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src_side)

        # MLP section
        src_norm = self.norm2(src)
        src_side = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(src_side)
        return src, attn_weights

class TransformerEncoder(torch.nn.TransformerEncoder):

    def __init__(self, encoder_layer, num_layers, norm=None, record_attn = False):
        super(TransformerEncoder, self).__init__(encoder_layer, num_layers)
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.record_attn = record_attn

    def forward(self, src, mask = None, src_key_padding_mask = None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        attn_weight_list = []

        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_weight_list.append(attn_weights.unsqueeze(0).detach())
        if self.norm is not None:
            output = self.norm(output)

        if self.record_attn:
            return output, torch.cat(attn_weight_list)
        else:
            return output

    def _get_clones(self, module, N):
        return torch.nn.modules.ModuleList([copy.deepcopy(module) for i in range(N)])

class PositionalEncoding(nn.Module):

    def __init__(self, hidden, dropout = 0.1, max_len = 256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden, 2) * (-np.log(10000.0) / hidden))
        pe = torch.zeros(max_len, 1, hidden)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AttnModule(nn.Module):
    def __init__(self, hidden = 128, layers = 8, record_attn = False, inpu_dim = 256):
        super(AttnModule, self).__init__()

        self.record_attn = record_attn
        self.pos_encoder = PositionalEncoding(hidden, dropout = 0.1)
        encoder_layers = TransformerLayer(hidden, 
                                          nhead = 8,
                                          dropout = 0.1,
                                          dim_feedforward = 512,
                                          batch_first = True)
        self.module = TransformerEncoder(encoder_layers, 
                                         layers, 
                                         record_attn = record_attn)

    def forward(self, x):
        x = self.pos_encoder(x)
        output = self.module(x)
        return output

    def inference(self, x):
        return self.module(x)

if __name__ == '__main__':
    main()
