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

class GroupLSTMCell(nn.Module):
    """
    GroupLSTMCell can compute the operation of N LSTM Cells at once.
    """

    def __init__(self, inp_size, hidden_size, num_lstms, gt=False, op=False, bias=True):
        super(GroupLSTMCell, self).__init__()

        self.num_lstms = num_lstms
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.op = op
        self.gt = gt

        self.i2h = nn.Linear(inp_size, 4 * num_lstms * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * num_lstms * hidden_size, bias=bias)

        if self.gt:
            pass
        elif self.op:
            self.scorer = nn.Sequential(
                nn.Linear(num_lstms, hidden_size, bias=bias),
                nn.ReLU(),
                nn.Linear(hidden_size, num_lstms, bias=bias)
            )
        else:
            self.scorer = nn.Sequential(
                nn.ReLU(),
                GroupLinearLayer(num_lstms, 8 * hidden_size, 1, bias=bias)
            )

    def forward(self, x, hid_state, op=None):
        """
        input: x (batch_size, input_size)
               hid_state (tuple of length 2 with each element of size (batch_size, num_lstms, hidden_state))
        output: h (batch_size, hidden_state)
                c ((batch_size, hidden_state))
        """
        h, c = hid_state
        bsz = h.shape[0]

        i_h = self.i2h(x).reshape(bsz, self.num_lstms, 4 * self.hidden_size)
        h_h = self.h2h(h).reshape(bsz, self.num_lstms, 4 * self.hidden_size)

        if self.gt:
            score = op.unsqueeze(-1)
        elif self.op:
            score = F.softmax(self.scorer(op), dim=1).unsqueeze(-1)
        else:
            score = F.softmax(self.scorer(torch.cat((i_h, h_h), dim=-1)), dim=1)

        preact = i_h + h_h

        gates = preact[:, :, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, :, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :, :self.hidden_size]
        f_t = gates[:, :, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, :, -self.hidden_size:]

        c_t = torch.mul(c.unsqueeze(1), f_t) + torch.mul(i_t, g_t)
        h_t = torch.mul(o_t, c_t.tanh())

        h_t = (h_t * score).sum(dim=1)
        c_t = (c_t * score).sum(dim=1)

        return h_t, c_t, score

class Model(nn.Module):
    def __init__(self, in_dim, enc_dim, hid_dim, out_dim, num_rules, model=None, op=False, bias=True):
        super(Model, self).__init__()

        self.in_dim = in_dim
        self.enc_dim = enc_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_rules = num_rules
        self.model = model
        self.op = op
        self.bias = bias

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, enc_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(enc_dim, enc_dim, bias=bias)
        )

        hid_new = (4 * self.num_rules * self.enc_dim + 1) ** 2
        hid_new += 4 * (4 * self.num_rules + 1) * (5 * hid_dim * hid_dim + hid_dim * (4 * self.enc_dim + 1))
        hid_new = math.sqrt(hid_new) - (4 * self.num_rules * self.enc_dim + 1)
        hid_new /= 2 * (4 * self.num_rules + 1)

        if self.model == 'Monolithic':
            self.rnn = nn.LSTMCell(self.enc_dim, self.hid_dim, bias=bias)
        elif self.model == 'Modular':
            self.hid_dim = int(hid_new)
            self.rnn = GroupLSTMCell(self.enc_dim, self.hid_dim, self.num_rules, False, self.op, bias=False)
        elif self.model == 'GT_Modular':
            self.hid_dim = int(hid_new)
            self.rnn = GroupLSTMCell(self.enc_dim, self.hid_dim, self.num_rules, True, False, bias=False)
        else:
            print("No Algorithm")
            exit()

        self.decoder = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.out_dim, bias=bias)
        )

    def forward(self, x, op):
        # x - (bsz, time, dim)
        # op - (bsz, time, rules)

        x = self.encoder(x)

        h = torch.zeros([x.shape[0], self.hid_dim]).cuda()
        c = torch.zeros_like(h)

        out = []
        scores = []

        for i in range(x.shape[1]):
            if self.model == 'Monolithic':
                h, c = self.rnn(x[:, i, :], (h, c))
            else:
                h, c, score = self.rnn(x[:, i, :], (h, c), op[:, i, :])
                scores.append(score)

            out.append(h)

        out = torch.stack(out)
        if self.model == 'Monolithic':
            scores = None
        else:
            scores = torch.stack(scores).transpose(1,0).contiguous()

        out = self.decoder(out).transpose(1,0).contiguous()

        return out, scores