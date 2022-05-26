# Eval file

import argparse
import torch
import torch.nn as nn
import random
import numpy as np
import os

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
parser.add_argument('--iterations', type=int, default=200000)

parser.add_argument('--enc-dim', type=int, default=128)
parser.add_argument('--h-dim', type=int, default=128)
parser.add_argument('--model', type=str, default='Monolithic', choices=('Monolithic', 'Modular', 'GT_Modular'))
parser.add_argument('--num-rules', type=int, default=2)
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

def eval_step(eval_len=args.seq_len, ood=False, n_evals=100):
    model.eval()
    total_loss = 0.

    with torch.no_grad():
        for _ in range(n_evals):
            data, label, op = rules(args.batch_size, eval_len, args.gt_rules, args.order, \
                                    args.d_dim, args.data_seed, ood)

            data = torch.Tensor(data).to(device)
            label = torch.Tensor(label).to(device)
            op = torch.Tensor(op).to(device)

            out, score = model(data, op)

            loss = criterion(out, label)
            total_loss += loss.item()

            del data, label, op

    return total_loss / float(n_evals)

if args.seq_len == 10:
    test_lens = [3, 5, 10, 20, 30]
else:
    test_lens = [10, 20, 30, 40, 50]

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

if args.best:
    ckpt = '_best'
else:
    ckpt = '_last'

if not os.path.exists(name):
    print(name)
    print('Model not found')
    slow_exit()
else:
    if not os.path.exists(f'{name}/loss.png'):
        print('Incomplete Model Training')
        slow_exit()

    if os.path.exists(f'{name}/perf{ckpt}.txt'):
        print('Performance Computed Already')
        slow_exit()

in_dim = args.d_dim + args.gt_rules

model = Model(
    in_dim = in_dim,
    enc_dim = args.enc_dim,
    hid_dim = args.h_dim,
    out_dim = args.d_dim,
    num_rules = args.num_rules,
    model = args.model,
    op = args.op
).to(device)

print('Loading Model')
model.load_state_dict(torch.load(f'{name}/model{ckpt}.pt'))
criterion = nn.L1Loss()

gt_ticks = [f'Ground Truth Rule {i}' for i in range(1, args.gt_rules+1)]

num_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"Number of Parameters: {num_params}")

# Compute Performance

eval_log = f'Iteration: 0 | '
eval_ood_log = f'Iteration: 0 | '

for seq_len in test_lens:
    eval_loss = eval_step(seq_len)
    eval_ood_loss = eval_step(seq_len, True)

    eval_log += f'Seq. Len: {seq_len} - Final Eval Loss: {eval_loss} | '
    eval_ood_log += f'Seq. Len: {seq_len} - Final Eval OoD Loss: {eval_ood_loss} | '

log = eval_log + '\n' + eval_ood_log + '\n'

print(log)
with open(os.path.join(name, f'perf{ckpt}.txt'), 'w') as f:
    f.write(log)