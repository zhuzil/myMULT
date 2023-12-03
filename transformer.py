# -*-codeing = utf-8 -*-
# @Time :2023/4/13 21:52
# @Author: zhu
# @Site:
# @File:transformer.py
# @Software:PyCharm

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

class TransformerEncoder(nn.Module):
    '''
    input:
        embed_dim: 词向量维度
        num_heads: 多头注意力中的头数
        num_layers: 编码器层数
        d_k: 注意力矩阵的维度
        d_v: 注意力矩阵的维度
    '''
    def __init__(self,embed_dim, num_heads, num_layers,  attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super(TransformerEncoder,self).__init__()
        self.embed_dropout = embed_dropout
        self.attn_mask = attn_mask

        self.PE = SinusoidalPositionalEmbedding()
        self.embed_scale = math.sqrt(embed_dim)
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            layer = TransformerEncoderLayer(embed_dim, num_heads, attn_dropout, relu_dropout, attn_mask)
            self.layers.append(layer)

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self,alpha,beta):
        '''
        beta -> alpha 交叉注意力
        如果alpha == alpha则变成自注意力
        '''
        alpha = self.embed_scale * alpha
        beta = self.embed_scale * beta
        alpha = self.PE(alpha)
        beta = self.PE(beta)
        alpha = F.dropout(alpha, p=self.embed_dropout, training=self.training)
        beta = F.dropout(beta, p=self.embed_dropout, training=self.training)
        for layer in self.layers:
            alpha = layer(alpha,beta)
            alpha = alpha.transpose(0,1)
        alpha = self.layer_norm(alpha)

        return alpha

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(1)
    if tensor2 is not None:
        dim2 = tensor2.size(1)
    # q:future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1)) 解释一下这句话
    # abs(dim2-dim1) 为了保证两个矩阵的维度一致，如果dim2>dim1,则dim2-dim1>0,则abs(dim2-dim1)为dim2-dim1,如果dim2<dim1,则dim2-dim1<0,则abs(dim2-dim1)为-dim2+dim1
    # 1+abs(dim2-dim1) 为了保证对角线上的元素为0，因为对角线上的元素不需要mask

    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]

class TransformerEncoderLayer(nn.Module):
    '''
    input:
        embed_dim
        num_heads
    '''
    def __init__(self,embed_dim, num_heads,attn_dropout=0.0, relu_dropout=0.0, attn_mask=False):
        super().__init__()
        self.attn = MultiheadAttention(embed_dim, num_heads, attn_dropout)
        self.ffn = PositionwiseFeedForward(embed_dim, relu_dropout)
        self.attn_mask = attn_mask

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self,alpha,beta):
        alpha = alpha.transpose(0,1)
        beta = beta.transpose(0,1)
        # 好像缺了层归一化
        alpha = self.norm1(alpha)
        beta = self.norm1(beta)

        mask = buffered_future_mask(alpha, beta) if self.attn_mask else None

        out,_ = self.attn(alpha,beta,beta,mask)
        out = self.ffn(out)
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self,embed_dim, relu_dropout=0.0):
        super().__init__()
        ffn_dim = 4*embed_dim # 从Mult代码中看到的
        self.w_1 = nn.Linear(embed_dim, ffn_dim)
        self.w_2 = nn.Linear(ffn_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.relu_dropout = nn.Dropout(relu_dropout)

    def forward(self,X):
        # x = self.relu_dropout(F.relu(self.w_1(X)))+X
        x = F.relu(self.w_1(X))
        x = self.relu_dropout(x)
        x = self.w_2(x)+X
        out = self.layer_norm(x)
        return out

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self,):
        super(SinusoidalPositionalEmbedding,self).__init__()
        # 将tensor注册成buffer, optim.step()的时候不会更新
        self.register_buffer('pe',self._get_sinusoid_encoding_table(1000,512))
    def _get_sinusoid_encoding_table(self,n_position, d_hid):
        '''
        Sinusoid position encoding table
        i表示embedding中dim维度，pos代表seq_len维度
        '''
        # 论文中的公式
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        # 转cuda

        return torch.FloatTensor(sinusoid_table).cuda()

    def forward(self,x):
        '''
        X : L * B * D
        '''
        # requires_grad为false 不需要训练的参数
        # b,l = x.size()
        ##############
        # 感觉这里有问题，到时候Debug再看看
        ##############
        # tmp : L * D
        l,b,d = x.size()
        # out : B * L + n_position * d_hid
        tmp = self.pe[:l, :d].clone().detach()
        # tmp 增加一维 L * 1 * D
        tmp.unsqueeze_(1)
        out = x + tmp
        # out = x + self.pe[:, :x.size(1)].clone().detach()
        return out
class MultiheadAttention(nn.Module):
    '''
    [batch_size, seq_len, embed_dim]
    '''
    def __init__(self,embed_dim, num_heads, attn_dropout=0.0, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        # self.head_dim = embed_dim // num_heads
        # assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.embed_dim ** -0.5

        # 三个全连接层
        self.Wq = nn.Linear(embed_dim, num_heads*embed_dim, bias=bias)
        self.Wk = nn.Linear(embed_dim, num_heads*embed_dim, bias=bias)
        self.Wv = nn.Linear(embed_dim, num_heads*embed_dim, bias=bias)

        self.attention = ScaledDotProductAttention(self.scaling)

        self.dropout = nn.Dropout(attn_dropout)
        self.fc = nn.Linear(num_heads * embed_dim, embed_dim, bias=bias)

        self.layerNorm = nn.LayerNorm(embed_dim)

    def forward(self,q,k,v,mask = None):
        residual = q

        batch_size = q.size(0)

        # [batch_size, seq_len, embed_dim] -> [batch_size, num_heads, seq_len, d_k/d_v]  # 错误 [batch_size, seq_len, num_heads, d_k/d_v]
        q = self.Wq(q).view(batch_size,-1,self.num_heads,self.embed_dim).transpose(1,2)
        k = self.Wk(k).view(batch_size,-1,self.num_heads,self.embed_dim).transpose(1,2)
        v = self.Wv(v).view(batch_size,-1,self.num_heads,self.embed_dim).transpose(1,2)

        # [batch_size, num_heads, seq_len, d_k/d_v]
        # q = q.transpose(2,3)
        # k = k.transpose(2,3)
        # v = v.transpose(2,3)

        # [batch_size, num_heads, seq_len, d_v]
        context, attn_weights = self.attention(q,k,v,mask)

        # [batch_size, seq_len, num_heads, d_v]
        context = context.transpose(1,2).contiguous().view(batch_size,-1,self.num_heads*self.embed_dim)

        output = self.dropout(self.fc(context))
        output = self.layerNorm(residual + output)

        # [batch_size, seq_len, d]
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return output, attn_weights

class ScaledDotProductAttention(nn.Module):
    def __init__(self,scaling,dropout=0.0):
        super(ScaledDotProductAttention,self).__init__()
        self.scaling = scaling
        self.dropout = nn.Dropout(dropout)
        # self.softmax = nn.Softmax(dim=2)

    def forward(self,q,k,v,mask = None):
        # [batch_size, num_heads, seq_len, d_k/d_v] --> [batch_size, num_heads, seq_len, seq_len]
        attn = torch.matmul(q,k.transpose(2,3))

        attn = attn * self.scaling
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn,dim=-1))
        # [batch_size, num_heads, seq_len, seq_len] --> [batch_size, num_heads, seq_len, d_v]
        output = torch.matmul(attn,v)

        return output, attn

if __name__ == '__main__':
    # 生成测试数据
    # x = torch.randn(2, 100, 512)
    # mask = torch.ones(2, 100, 100)
    # mask = torch.tril(mask)
    # mask = mask.unsqueeze(1).repeat(1, 8, 1, 1)
    encoder = TransformerEncoder(30, 5, 5)
    alpha = torch.rand(50, 8, 30).clone().detach()
    beta = torch.rand(50, 8, 30).clone().detach()
    print(encoder(alpha,beta).shape)

