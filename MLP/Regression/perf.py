# Eval file

import argparse
import torch
import torch.nn as nn
import random
import numpy as np
import os

from data import data_v1, data_v2
from model import Monolithic, Modular, GT_Modular

parser = argparse.ArgumentParser(description='Rule MLP')
parser.add_argument('--gt-rules', type=int, default=2)
parser.add_argument('--data-seed', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--iterations', type=int, default=100000)

parser.add_argument('--encoder-dim', type=int, default=32)
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--model', type=str, default='Monolithic', choices=('Monolithic', 'Modular', 'GT_Modular'))
parser.add_argument('--num-rules', type=int, default=2)
parser.add_argument('--joint', action='store_true', default=False)
parser.add_argument('--op', action='store_true', default=False)

parser.add_argument('--scheduler', action='store_true', default=False)
parser.add_argument('--best', action='store_true', default=False)
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

def slow_exit():
    import time
    time.sleep(1)
    exit()

set_seed(args.seed)

def eval_step(ood=False, n_evals=100):
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

if args.best:
    ckpt = '_best'
else:
    ckpt = '_last'

if not os.path.exists(name):
    print('Model not found')
    slow_exit()
else:
    if not os.path.exists(f'{name}/loss.png'):
        print('Incomplete Model Training')
        slow_exit()

    if os.path.exists(f'{name}/perf{ckpt}.txt'):
        print('Performance Computed Already')
        slow_exit()

if args.model == 'Monolithic':
    model = Monolithic(op_dim, args.encoder_dim, args.dim).to(device)
elif args.model == 'Modular':
    model = Modular(op_dim, args.encoder_dim, args.dim, args.num_rules, args.op, args.joint).to(device)
elif args.model == 'GT_Modular':
    model = GT_Modular(op_dim, args.encoder_dim, args.dim, args.gt_rules).to(device)
else:
    print("Model Not Implemented")
    slow_exit()

gt_ticks = [f'Ground Truth Rule {i}' for i in range(1, args.gt_rules+1)]
data_call = data_v2

num_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"Number of Parameters: {num_params}")

criterion = nn.L1Loss()

print('Loading Model')
model.load_state_dict(torch.load(f'{name}/model{ckpt}.pt'))

eval_loss = eval_step()
eval_ood_loss = eval_step(True)

log = f'Final Eval Loss: {eval_loss}\n' \
      f'Final Eval OoD Loss: {eval_ood_loss}\n'

print(log)

with open(os.path.join(name, f'perf{ckpt}.txt'), 'w') as f:
    f.write(log)