import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

class GroupLinearLayer(nn.Module):
    """Modularized Linear Layer"""
    def __init__(self, num_blocks, din, dout, bias=True):
        super(GroupLinearLayer, self).__init__()

        self.bias=bias
        self.num_blocks = num_blocks
        self.din = din
        self.dout = dout
        self.w = nn.Parameter(torch.Tensor(num_blocks, din, dout))
        self.b = nn.Parameter(torch.Tensor(1, num_blocks, dout))

        self.reset_parameters()

    def reset_parameters(self):
        bound = math.sqrt(1.0 / self.din)
        nn.init.uniform_(self.w, -bound, bound)
        if self.bias:
            nn.init.uniform_(self.b, -bound, bound)

    def extra_repr(self):
        return 'groups={}, in_features={}, out_features={}, bias={}'.format(
            self.num_blocks, self.din, self.dout, self.bias is not None
        )

    def forward(self,x):
        # x - (bsz, num_blocks, din)
        x = x.permute(1,0,2)
        x = torch.bmm(x, self.w)
        x = x.permute(1,0,2)

        if self.bias:
            x = x + self.b

        return x

class Modular_Transformer(nn.Module):
    def __init__(self, dim, att_dim, num_heads, num_rules, op, bias):
        super(Modular_Transformer, self).__init__()

        self.dim = dim
        self.att_dim = att_dim
        self.head_dim = att_dim // (num_heads * num_rules)
        self.num_heads = num_heads
        self.num_rules = num_rules
        self.scaling = self.head_dim ** -0.5
        self.op = op

        self.query_net = nn.Linear(dim, self.num_heads * self.num_rules * self.head_dim, bias=bias)
        self.key_net = nn.Linear(dim, self.num_heads * self.num_rules * self.head_dim, bias=bias)
        self.value_net = nn.Linear(dim, self.num_heads * self.num_rules * self.head_dim, bias=bias)

        self.in_ = nn.Linear(dim, self.head_dim * self.num_rules, bias=bias)

        num = (2 * self.dim * (self.num_rules * self.num_heads * (self.att_dim + self.dim) + self.att_dim))
        denom = (self.num_rules * (self.att_dim * (self.num_heads + 1) + self.num_rules * self.num_heads * self.dim))

        intermediate =  int(num / denom)

        if self.op:
            self.scorer = nn.Sequential(
                nn.Linear(num_rules, self.head_dim),
                nn.ReLU(),
                nn.Linear(self.head_dim, num_rules)
            )
            self.out_proj = nn.Sequential(
                GroupLinearLayer(num_rules, self.head_dim * (self.num_heads + 1), intermediate, bias=bias),
                nn.ReLU(),
                GroupLinearLayer(num_rules, intermediate, dim)
            )
        else:
            self.out_proj = nn.Sequential(
                GroupLinearLayer(num_rules, self.head_dim * (self.num_heads + 1), intermediate, bias=bias),
                nn.ReLU(),
                GroupLinearLayer(num_rules, intermediate, dim + 1)
            )

    def forward(self, x, r_scores):
        bsz, n, _ = x.shape

        q = self.query_net(x).view(bsz, n, self.num_rules * self.num_heads, self.head_dim) * self.scaling
        k = self.key_net(x).view(bsz, n, self.num_rules * self.num_heads, self.head_dim)
        v = self.value_net(x).view(bsz, n, self.num_rules * self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        score = torch.matmul(q, k) # (bsz, num_rules * num_heads, n, n)
        mask = torch.zeros_like(score[0,0]).fill_diagonal_(1).unsqueeze(0).unsqueeze(0)
        mask = mask.repeat(bsz, self.num_heads * self.num_rules, 1, 1).bool()
        score.masked_fill_(mask, float('-inf'))
        score = F.softmax(score, dim=-1)

        out = torch.matmul(score, v).transpose(2,1).reshape(bsz * n, self.num_rules, self.head_dim * self.num_heads)
        in_ = self.in_(x).reshape(bsz * n, self.num_rules, self.head_dim)
        out = torch.cat([out, in_], dim=-1)
        out = self.out_proj(out)

        if self.op:
            r_values = out.reshape(bsz, n, self.num_rules, self.dim)
            r_scores = F.softmax(self.scorer(r_scores), dim=-1).unsqueeze(-1)
        else:
            out = out.reshape(bsz, n, self.num_rules, self.dim + 1)
            r_values = out[:,:,:,:-1]
            r_scores = F.softmax(out[:,:,:,-1:], dim=2)

        out = (r_values * r_scores).sum(dim=2)

        return out, r_scores

class GT_Modular_Transformer(nn.Module):
    def __init__(self, dim, att_dim, num_heads, num_rules, bias):
        super(GT_Modular_Transformer, self).__init__()

        self.dim = dim
        self.att_dim = att_dim
        self.head_dim = att_dim // (num_heads * num_rules)
        self.num_heads = num_heads
        self.num_rules = num_rules
        self.scaling = self.head_dim ** -0.5

        num = (2 * self.dim * (self.num_rules * self.num_heads * (self.att_dim + self.dim) + self.att_dim))
        denom = (self.num_rules * (self.att_dim * (self.num_heads + 1) + self.num_rules * self.num_heads * self.dim))

        intermediate =  int(num / denom)

        self.query_net = nn.Linear(dim, self.num_heads * self.num_rules * self.head_dim, bias=bias)
        self.key_net = nn.Linear(dim, self.num_heads * self.num_rules * self.head_dim, bias=bias)
        self.value_net = nn.Linear(dim, self.num_heads * self.num_rules * self.head_dim, bias=bias)

        self.in_ = nn.Linear(dim, self.head_dim * self.num_rules, bias=bias)

        self.out_proj = nn.Sequential(
            GroupLinearLayer(num_rules, self.head_dim * (self.num_heads + 1), intermediate, bias=bias),
            nn.ReLU(),
            GroupLinearLayer(num_rules, intermediate, dim)
        )

    def forward(self, x, r_scores):
        bsz, n, _ = x.shape

        q = self.query_net(x).view(bsz, n, self.num_rules * self.num_heads, self.head_dim) * self.scaling
        k = self.key_net(x).view(bsz, n, self.num_rules * self.num_heads, self.head_dim)
        v = self.value_net(x).view(bsz, n, self.num_rules * self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        score = torch.matmul(q, k) # (bsz, num_rules * num_heads, n, n)
        mask = torch.zeros_like(score[0,0]).fill_diagonal_(1).unsqueeze(0).unsqueeze(0)
        mask = mask.repeat(bsz, self.num_heads * self.num_rules, 1, 1).bool()
        score.masked_fill_(mask, float('-inf'))
        score = F.softmax(score, dim=-1)

        out = torch.matmul(score, v).transpose(2,1).reshape(bsz * n, self.num_rules, self.head_dim * self.num_heads)

        in_ = self.in_(x).reshape(bsz * n, self.num_rules, self.head_dim)
        out = torch.cat([out, in_], dim=-1)

        r_values = self.out_proj(out).reshape(bsz, n, self.num_rules, self.dim)
        out = (r_values * r_scores.unsqueeze(-1)).sum(dim=2)

        return out, r_scores

class Monolithic_Transformer(nn.Module):
    def __init__(self, dim, att_dim, num_heads, bias=True):
        super(Monolithic_Transformer, self).__init__()

        self.dim = dim
        self.att_dim = att_dim
        self.head_dim = att_dim // num_heads
        self.num_heads = num_heads
        self.scaling = self.head_dim ** -0.5

        self.query_net = nn.Linear(dim, self.head_dim * self.num_heads, bias=bias)
        self.key_net = nn.Linear(dim, self.head_dim * self.num_heads, bias=bias)
        self.value_net = nn.Linear(dim, self.head_dim * self.num_heads, bias=bias)

        self.in_ = nn.Linear(dim, self.head_dim, bias=bias)

        self.out_proj = nn.Sequential(
            nn.Linear(self.head_dim * (self.num_heads + 1), dim * 2, bias=bias),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, r_score=None):
        bsz, n, _ = x.shape

        q = self.query_net(x).view(bsz, n, self.num_heads, self.head_dim) * self.scaling
        k = self.key_net(x).view(bsz, n, self.num_heads, self.head_dim)
        v = self.value_net(x).view(bsz, n, self.num_heads, self.head_dim)

        q = q.transpose(2,1).contiguous()
        k = k.permute(0, 2, 3, 1).contiguous()
        v = v.transpose(2,1).contiguous()

        score = torch.matmul(q, k) # (bsz, num_heads, n, n)
        mask = torch.zeros_like(score[0,0]).fill_diagonal_(1).unsqueeze(0).unsqueeze(0)
        mask = mask.repeat(bsz, self.num_heads, 1, 1).bool()
        score.masked_fill_(mask, float('-inf'))
        score = F.softmax(score, dim=-1)

        out = torch.matmul(score, v).transpose(2, 1).reshape(bsz, n, self.head_dim * self.num_heads)
        in_ = self.in_(x).view(bsz, n, self.head_dim)
        out = torch.cat([out, in_], dim=-1)
        out = self.out_proj(out)

        return out, None

class Model(nn.Module):
    def __init__(self, dim=64, att_dim=128, num_heads=4, in_dim=10,
                 model=None, num_rules=1, op=False, bias=True):
        super(Model, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

        if model == 'Modular':
            self.model = Modular_Transformer(dim, att_dim, num_heads, num_rules, op, bias)
        elif model == 'GT_Modular':
            self.model = GT_Modular_Transformer(dim, att_dim, num_heads, num_rules, bias)
        elif model == 'Monolithic':
            self.model = Monolithic_Transformer(dim, att_dim, num_heads, bias)
        else:
            print("No Algorithm")
            exit()

    def forward(self, x, op):
        x = self.encoder(x)
        x, f_score = self.model(x, op)
        x = self.decoder(x)

        return x.squeeze(), f_score