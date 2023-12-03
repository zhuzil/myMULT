# -*-codeing = utf-8 -*-
# @Time :2023/4/13 21:59
# @Author: zhu
# @Site:
# @File:Mult.py
# @Software:PyCharm
import torch
from torch import nn
import torch.nn.functional as F
from transformer import TransformerEncoder

class Mult(nn.Module):
    def __init__(self,orig_d_l, orig_d_a, orig_d_v):
        super().__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = orig_d_l, orig_d_a, orig_d_v
        self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.attn_dropout = 0.1
        self.attn_dropout_a = 0
        self.attn_dropout_v = 0
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0
        self.embed_dropout = 0.25
        self.output_dim = 1
        self.attn_mask = True
        # 1D Conv
        self.Conv1D_L = nn.Conv1d(self.orig_d_l,self.d_l,kernel_size=1,padding=0, bias=False)
        self.Conv1D_V = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.Conv1D_A = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # Positional Embedding 作为Transformer中的一部分

        # transformer
        self.Transformer_A_L = TransformerEncoder(self.d_l,5,5,self.attn_dropout,self.relu_dropout,self.res_dropout,self.embed_dropout,self.attn_mask)
        self.Transformer_V_L = TransformerEncoder(self.d_l,5,5,self.attn_dropout,self.relu_dropout,self.res_dropout,self.embed_dropout,self.attn_mask)

        self.Transformer_A_V = TransformerEncoder(self.d_v,5,5,self.attn_dropout,self.relu_dropout,self.res_dropout,self.embed_dropout,self.attn_mask)
        self.Transformer_L_V = TransformerEncoder(self.d_v,5,5,self.attn_dropout,self.relu_dropout,self.res_dropout,self.embed_dropout,self.attn_mask)

        self.Transformer_V_A = TransformerEncoder(self.d_a,5,5,self.attn_dropout,self.relu_dropout,self.res_dropout,self.embed_dropout,self.attn_mask)
        self.Transformer_L_A = TransformerEncoder(self.d_a,5,5,self.attn_dropout,self.relu_dropout,self.res_dropout,self.embed_dropout,self.attn_mask)

        # Concatenation Transformer
        self.Transformer_L = TransformerEncoder(2*self.d_l,5,5,self.attn_dropout,self.relu_dropout,self.res_dropout,self.embed_dropout,self.attn_mask)
        self.Transformer_V = TransformerEncoder(2*self.d_v,5,5,self.attn_dropout,self.relu_dropout,self.res_dropout,self.embed_dropout,self.attn_mask)
        self.Transformer_A = TransformerEncoder(2*self.d_a,5,5,self.attn_dropout,self.relu_dropout,self.res_dropout,self.embed_dropout,self.attn_mask)

        # full connection
        dims = 2 * (self.d_l + self.d_a + self.d_v)
        self.full_connection = nn.Sequential(
            nn.Linear(dims,dims),
            nn.ReLU(),
            nn.Dropout(self.out_dropout),
            nn.Linear(dims,dims),
        )

        # output
        self.outLayer = nn.Linear(dims,self.output_dim)

    def forward(self, x_l, x_a, x_v):
        '''
        [batch_size, seq_len, n_features]
        '''
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training) # B, D, L
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        x_l = self.Conv1D_L(x_l) # B, D, L
        x_a = self.Conv1D_A(x_a) # B, D, A
        x_v = self.Conv1D_V(x_v) # B, D, V

        x_l = x_l.permute(2, 0, 1) # L, B, D
        x_a = x_a.permute(2, 0, 1) # A, B, D
        x_v = x_v.permute(2, 0, 1) # V, B, D

        # crossmodal transformer
        x_a_l = self.Transformer_A_L(x_l, x_a) # L, B, D
        x_v_l = self.Transformer_V_L(x_l, x_v) # L, B, D

        x_a_v = self.Transformer_A_V(x_v, x_a) # V, B, D
        x_l_v = self.Transformer_L_V(x_v, x_l) # V, B, D

        x_v_a = self.Transformer_V_A(x_a, x_v) # A, B, D
        x_l_a = self.Transformer_L_A(x_a, x_l) # A, B, D

        # concatenation transformer
        X_L = torch.cat((x_a_l, x_v_l), dim=2) # L, B, 2D
        X_V = torch.cat((x_a_v, x_l_v), dim=2) # V, B, 2D
        X_A = torch.cat((x_v_a, x_l_a), dim=2) # A, B, 2D

        X_L = self.Transformer_L(X_L,X_L) # L, B, 2D
        X_V = self.Transformer_V(X_V,X_V) # V, B, 2D
        X_A = self.Transformer_A(X_A,X_A) # A, B, 2D

        # full connection
        out = torch.cat((X_L[-1], X_V[-1], X_A[-1]), dim=1) # B, 6D

        # 残差
        output = self.full_connection(out) + out # B, 6D

        # 输出
        output = self.outLayer(output)  # B, 1

        return output,out


