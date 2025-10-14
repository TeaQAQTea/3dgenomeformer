import torch
import torch.nn as nn
import numpy as np
import copy

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

class StackedCrossAttentionBlocks(nn.Module):
    def __init__(self, num_blocks = 4):
        super(StackedCrossAttentionBlocks, self).__init__()
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(128,128,128)
            for _ in range(num_blocks)
        ])

    def forward(self, query, key, value, mask=None):
        for block in self.blocks:
            query,key,value= block(query, key, value)
            # print('query', query.shape, 'key', key.shape, 'value', value.shape)
        return value

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
        self.cross_attn_epi = StackedCrossAttentionBlocks()
        self.cross_attn_seq = StackedCrossAttentionBlocks() 
        self.conv_end = nn.Conv1d(256, output_size, 1)

    def forward(self, x):

        seq = x[:, :5, :]
        epi = x[:, 5:, :]
        seq = self.res_blocks_seq(self.conv_start_seq(seq))
        # print('seq', seq.shape)


        epi = self.res_blocks_epi(self.conv_start_epi(epi))
        
        # print('epi', epi.shape)
        # print('seq', seq.shape)
        epi = torch.transpose(epi, 1, 2)
        seq = torch.transpose(seq, 1, 2)
        epi_cross = self.cross_attn_epi(seq, epi, epi)
        seq_cross = self.cross_attn_seq(epi, seq, seq)

        seq_cross = torch.transpose(seq_cross, 1, 2)
        epi_cross = torch.transpose(epi_cross, 1, 2)
        
        
        x = torch.cat([seq_cross, epi_cross], dim = 1)
        out = self.conv_end(x)
        return out

    def get_res_cross_blocks(self, n, key_dim, query_dim, value_dim):
        blocks = []
        for i in range(n):
            blocks.append(CrossAttentionBlock(key_dim, query_dim, value_dim))    
        res_blocks = nn.Sequential(*blocks)
        return res_blocks

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


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim,  value_dim):
        super(CrossAttention, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # 定义注意力权重参数
        self.query = nn.Linear(query_dim, query_dim)
        self.key = nn.Linear(key_dim, query_dim)
        self.value = nn.Linear(value_dim, value_dim)
        
    def forward(self, query_input, key_input, value_input):
        batch_size, query_len, _ = query_input.size()
        batch_size, key_len, _ = key_input.size()
        batch_size, value_len, _ = value_input.size()
        # print(query_input.shape, key_input.shape, value_input.shape)
        # 计算注意力的query、key和value
        query = self.query(query_input)
        key = self.key(key_input)
        value = self.value(value_input)
        
        # 计算注意力分数
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.query_dim ** 0.5
        
        # 应用softmax归一化得到注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 将注意力权重应用到value上
        attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous().view(batch_size, query_len, self.value_dim)
        
        return attn_output

class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim,layers = 8, record_attn = False):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attn = CrossAttention(query_dim, key_dim, value_dim)
        self.norm = nn.LayerNorm(value_dim)
    def forward(self, query_input, key_input, value_input):
        cross_out = self.cross_attn(query_input, key_input, value_input)
        # print('cross_out', cross_out.shape)
        value_input = self.norm(value_input+cross_out)
        key_input = value_input
        return query_input, key_input, value_input

if __name__ == '__main__':
    main()
