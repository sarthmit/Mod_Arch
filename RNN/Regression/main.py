# Trainer file

import argparse
import torch
import torch.nn as nn
import random
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data import rules
from model import Model

parser = argparse.ArgumentParser(description='Rule MLP')
parser.add_argument('--gt-rules', type=int, default=2)
parser.add_argument('--order', type=int, default=1)
parser.add_argument('--seq-len', type=int, default=10)
parser.add_argument('--d-dim', type=int, default=1)
parser.add_argument('--data-seed', type=int, default=0)

parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--iterations', type=int, default=500000)

parser.add_argument('--enc-dim', type=int, default=128)
parser.add_argument('--h-dim', type=int, default=128)
parser.add_argument('--model', type=str, default='Monolithic', choices=('Monolithic', 'Modular', 'GT_Modular'))
parser.add_argument('--num-rules', type=int, default=2)
parser.add_argument('--op', action='store_true', default=False)

parser.add_argument('--scheduler', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

args.seed += int(os.environ['SLURM_PROCID'])
args.data_seed = args.seed % 5
args.seed = args.seed // 5

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(args.seed)

if args.seq_len == 10:
    test_lens = [3, 5, 10, 20, 30]
else:
    test_lens = [10, 20, 30, 40, 50]

config = vars(args)
device = torch.device('cuda')

if args.op:
    extras = f'_operation-only_'
else:
    extras = '_'

if args.scheduler:
    ext='_scheduler'
else:
    ext=''

name = f'Sequence_{args.seq_len}_Order_{args.order}_Dim_{args.d_dim}{ext}/Data-Seed_{args.data_seed}/GT_Rules_{args.gt_rules}/{args.model}{extras}{args.enc_dim}_{args.h_dim}_{args.num_rules}_{args.seed}'

if not os.path.exists(name):
    os.makedirs(name)
else:
    print(name)
    print("Folder Already Exists")
    if os.path.exists(os.path.join(name, 'loss.png')):
        print("Model Already Exists")
        exit()

in_dim = args.d_dim + args.gt_rules

gt_ticks = [f'Ground Truth Rule {i}' for i in range(1, args.gt_rules + 1)]

model = Model(
    in_dim = in_dim,
    enc_dim = args.enc_dim,
    hid_dim = args.h_dim,
    out_dim = args.d_dim,
    num_rules = args.num_rules,
    model = args.model,
    op = args.op
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"Number of Parameters: {num_params}")
with open(os.path.join(name, 'log.txt'), 'w') as f:
    f.write(f"Number of Parameters: {num_params}\n")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.L1Loss()

df = pd.DataFrame(columns=["Iterations", "Loss"])

def eval_step(eval_len=args.seq_len, ood=False, n_evals=10):
    model.eval()
    total_loss = 0.

    for _ in range(n_evals):
        data, label, op = rules(args.batch_size, eval_len, args.gt_rules, args.order, \
                            args.d_dim, args.data_seed, ood)

        data = torch.Tensor(data).to(device)
        label = torch.Tensor(label).to(device)
        op = torch.Tensor(op).to(device)

        out, score = model(data, op)

        loss = criterion(out, label)
        total_loss += loss.item()

    return total_loss / n_evals

def train_step():
    model.train()
    model.zero_grad()

    data, label, op = rules(args.batch_size, args.seq_len, args.gt_rules, args.order, \
                            args.d_dim, args.data_seed)

    data = torch.Tensor(data).to(device)
    label = torch.Tensor(label).to(device)
    op = torch.Tensor(op).to(device)

    out, score = model(data, op)
    loss = criterion(out, label)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss

eval_log = f'Iteration: 0 | '
train_log = f'Iteration: 0 | '
eval_ood_log = f'Iteration: 0 | '

for seq_len in test_lens:
    eval_loss = eval_step(seq_len)
    eval_ood_loss = eval_step(seq_len, True)

    if seq_len == args.seq_len:
        df.loc[-1] = [0, eval_loss]
        df.index = df.index + 1

    eval_log += f'Seq. Len: {seq_len} - Eval Loss: {eval_loss} | '
    train_log += f'Seq. Len: {seq_len} - Train Loss: {eval_loss} | '
    eval_ood_log += f'Seq. Len: {seq_len} - Eval OoD Loss: {eval_ood_loss} | '

log = train_log + '\n' + eval_log + '\n' + eval_ood_log + '\n'
print(log)

with open(os.path.join(name, 'log.txt'), 'a') as f:
    f.write(log)

best_val = float('inf')
for i in range(1, args.iterations+1):
    if i % 5000 == 0:
        eval_loss = eval_step()
        val_loss = eval_step()

        df.loc[-1] = [i, eval_loss]
        df.index = df.index + 1

    train_loss = train_step()

    if i % 5000 == 0:
        if args.scheduler:
            scheduler.step()

        log = f'Iteration: {i} | Train Loss: {train_loss}\n' \
              f'Iteration: {i} | Eval Loss: {eval_loss}\n'
        print(log)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(name, 'model_best.pt'))

        torch.save(model.state_dict(), os.path.join(name, 'model_last.pt'))

        with open(os.path.join(name, 'log.txt'), 'a') as f:
            f.write(log)

sns.lineplot(data=df, x="Iterations", y="Loss")
plt.savefig(os.path.join(name, 'loss.png'), bbox_inches='tight')
plt.close()