# Trainer file

import argparse
import torch
import torch.nn as nn
import random
import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from data import data_v1, data_v2
from model import Monolithic, Modular, GT_Modular

parser = argparse.ArgumentParser(description='Rule MLP')
parser.add_argument('--gt-rules', type=int, default=2)
parser.add_argument('--data-seed', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--iterations', type=int, default=100000)

parser.add_argument('--encoder-dim', type=int, default=32)
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--model', type=str, default='Monolithic', choices=('Monolithic', 'Modular', 'GT_Modular'))
parser.add_argument('--num-rules', type=int, default=2)
parser.add_argument('--joint', action='store_true', default=False)
parser.add_argument('--op', action='store_true', default=False)

parser.add_argument('--scheduler', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

sns.color_palette("dark", as_cmap=True)
sns.set(style="darkgrid")

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

config = vars(args)
device = torch.device('cuda')
op_dim = args.gt_rules

if args.op:
    extras = f'_operation-only_'
elif args.joint:
    extras = f'_joint_'
else:
    extras = '_'

if args.scheduler:
    ext='_scheduler'
else:
    ext=''

name = f'Logs{ext}/Data-Seed_{args.data_seed}/GT_Rules_{args.gt_rules}/{args.model}{extras}{args.encoder_dim}_{args.dim}_{args.num_rules}_{args.seed}'

if not os.path.exists(name):
    os.makedirs(name)
else:
    if os.path.exists(f'{name}/loss.png'):
        print('Model Exists')
        exit()

if args.model == 'Monolithic':
    model = Monolithic(op_dim, args.encoder_dim, args.dim).to(device)
elif args.model == 'Modular':
    model = Modular(op_dim, args.encoder_dim, args.dim, args.num_rules, args.op, args.joint).to(device)
elif args.model == 'GT_Modular':
    model = GT_Modular(op_dim, args.encoder_dim, args.dim, args.gt_rules).to(device)
else:
    print("Model Not Implemented")
    exit()

gt_ticks = [f'Ground Truth Rule {i}' for i in range(1, args.gt_rules+1)]
data_call = data_v2

num_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"Number of Parameters: {num_params}")
with open(os.path.join(name, 'log.txt'), 'w') as f:
    f.write(f"Number of Parameters: {num_params}\n")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15], gamma=0.5)
criterion = nn.L1Loss()

df = pd.DataFrame(columns=["Iterations", "Loss"])

def eval_step(ood=False, n_evals=10):
    model.eval()
    total_loss = 0.

    with torch.no_grad():
        for _ in range(n_evals):
            data, label = data_call(args.batch_size, args.gt_rules, args.data_seed, ood)

            data = torch.Tensor(data).to(device)
            label = torch.Tensor(label).to(device)

            out, score = model(data)

            loss = criterion(out, label)
            total_loss += loss.item()

    return total_loss / float(n_evals)

def train_step():
    model.train()
    model.zero_grad()

    data, label = data_call(args.batch_size, args.gt_rules, args.data_seed)

    data = torch.Tensor(data).to(device)
    label = torch.Tensor(label).to(device)

    out, score = model(data)

    loss = criterion(out, label)

    loss.backward()
    optimizer.step()

    return loss

eval_loss = eval_step()
eval_ood_loss = eval_step(True)

log = f'Iteration: 0 | Eval OoD Loss: {eval_ood_loss}\n' \
      f'Iteration: 0 | Train Loss: {eval_loss}\n' \
      f'Iteration: 0 | Eval Loss: {eval_loss}\n'

print(log)

with open(os.path.join(name, 'log.txt'), 'a') as f:
    f.write(log)

best_val = float('inf')
df.loc[-1] = [0, eval_loss]
df.index = df.index + 1
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