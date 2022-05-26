# Models

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class Monolithic(nn.Module):
    def __init__(self, op_dim, encoder_dim, dim):
        super(Monolithic, self).__init__()

        self.encoder_digit = nn.Sequential(
            nn.Linear(1, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim)
        )

        self.encoder_operation = nn.Sequential(
            nn.Linear(op_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim)
        )

        self.MLP = nn.Sequential(
            nn.Linear(encoder_dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, 1)
        )

    def forward(self, x):
        dig1, dig2, op = x[:, 0:1] , x[:, 1:2] , x[:, 2:]

        dig1 = self.encoder_digit(dig1)
        dig2 = self.encoder_digit(dig2)
        op = self.encoder_operation(op)

        sample = torch.cat((dig1, dig2, op), dim=-1)
        sample = self.MLP(sample)

        return self.decoder(sample).squeeze(), None

class Modular(nn.Module):
    def __init__(self, op_dim, encoder_dim, dim, num_rules, op=False, joint=False):
        super(Modular, self).__init__()

        self.dim = dim
        self.encoder_dim = encoder_dim
        self.op_dim = op_dim
        self.num_rules = num_rules
        self.joint = joint
        self.op = op

        self.encoder_digit = nn.Sequential(
            nn.Linear(1, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim)
        )

        self.encoder_operation = nn.Sequential(
            nn.Linear(op_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim)
        )

        if joint:
            self.MLP = nn.Sequential(
                GroupLinearLayer(num_rules, encoder_dim * 3, dim // num_rules),
                nn.ReLU(),
                GroupLinearLayer(num_rules, dim // num_rules, dim + 1)
            )
        else:
            self.MLP = nn.Sequential(
                GroupLinearLayer(num_rules, encoder_dim * 3, dim // num_rules),
                nn.ReLU(),
                GroupLinearLayer(num_rules, dim // num_rules, dim)
            )
            if op:
                self.scorer = nn.Linear(encoder_dim, num_rules)
            else:
                self.scorer = nn.Sequential(
                    nn.Linear(dim + op_dim + 2, encoder_dim),
                    nn.ReLU(),
                    nn.Linear(encoder_dim, 1)
                )

        self.decoder = nn.Sequential(
            nn.Linear(dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, 1)
        )

    def forward(self, x):
        dig1, dig2, op = x[:, 0:1] , x[:, 1:2] , x[:, 2:]

        dig1 = self.encoder_digit(dig1)
        dig2 = self.encoder_digit(dig2)
        op = self.encoder_operation(op)

        sample = torch.cat((dig1, dig2, op), dim=-1)
        sample = sample.unsqueeze(1).repeat(1, self.num_rules, 1)

        if self.joint:
            out = self.MLP(sample)
            score = F.softmax(out[:,:,-1:], dim=1)
            out = out[:,:,:-1]
        else:
            out = self.MLP(sample)
            if self.op:
                score = F.softmax(self.scorer(op), dim=-1)
                score = score.unsqueeze(-1)
            else:
                score = torch.cat([out, x.unsqueeze(1).repeat(1, self.num_rules, 1)], dim=-1)
                score = F.softmax(self.scorer(score), dim=1)

        out = (out * score).sum(dim=1)

        return self.decoder(out).squeeze(), score.squeeze()

class GT_Modular(nn.Module):
    def __init__(self, op_dim, encoder_dim, dim, num_rules):
        super(GT_Modular, self).__init__()

        self.num_rules = num_rules

        self.encoder_digit = nn.Sequential(
            nn.Linear(1, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim)
        )

        self.encoder_operation = nn.Sequential(
            nn.Linear(op_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim)
        )

        self.MLP = nn.Sequential(
            GroupLinearLayer(num_rules, encoder_dim * 3, dim // num_rules),
            nn.ReLU(),
            GroupLinearLayer(num_rules, dim // num_rules, dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, 1)
        )

    def forward(self, x):
        dig1, dig2, op = x[:, 0:1] , x[:, 1:2] , x[:, 2:]

        dig1 = self.encoder_digit(dig1)
        dig2 = self.encoder_digit(dig2)
        op = self.encoder_operation(op)

        sample = torch.cat((dig1, dig2, op), dim=-1)
        sample = sample.unsqueeze(1).repeat(1, self.num_rules, 1)

        out = self.MLP(sample)
        score = x[:, 2:].unsqueeze(-1)

        out = (out * score).sum(dim=1)

        return self.decoder(out).squeeze(), score.squeeze()