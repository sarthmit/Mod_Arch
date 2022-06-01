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
parser.add_argument('--best', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

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

def specialization_score(true_p, empirical_p):
    true_p.sort()
    empirical_p.sort()

    return np.sum(np.abs(true_p - empirical_p)) / 2.

def specialization_metric():
    spec = 0.

    with torch.no_grad():
        for eval_seed in range(100):
            scores = np.zeros(args.gt_rules)
            rng = np.random.RandomState(eval_seed)
            p = rng.dirichlet(alpha = np.ones(args.gt_rules))
            for _ in range(10):
                data, label = data_call(1000, args.gt_rules, args.data_seed, prob=p)
                data = torch.Tensor(data).to(device)
                label = torch.Tensor(label).to(device)

                out, score = model(data)
                score = score.mean(dim=0).detach().cpu().numpy()
                scores += score

            spec += specialization_score(p, scores / 10.)

        return spec / 100.

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

    if os.path.exists(f'{name}/specialization{ckpt}.txt'):
        print('Specialization Computed Already')
        slow_exit()

if args.model == 'Monolithic':
    slow_exit()
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

print('Loading Model')
model.load_state_dict(torch.load(f'{name}/model{ckpt}.pt'))

spec = specialization_metric()
print(f'Specialization Metric: {spec}')

with open(os.path.join(name, f'specialization{ckpt}.txt'), 'w') as f:
    f.write(f'Specialization Metric: {spec}')