# Eval file

import argparse
import numpy as np
import os
from scipy.optimize import linear_sum_assignment

parser = argparse.ArgumentParser(description='Rule MLP')
args = parser.parse_args()

EPS = 1e-8

def get_name(rules, model, enc_dim, dim, seed):
    d = seed % 5
    s = seed // 5

    return f'Sequence_10_Order_1_Dim_32/Data-Seed_{d}/GT_Rules_{rules}/{model}_{enc_dim}_{dim}_{rules}_{s}'

def metrics(rules, model, enc_dim, dim, seed, ckpt='_last'):
    name = get_name(rules, model, enc_dim, dim, seed)
    print(name)

    if 'Monolithic' in name:
        return

    if not os.path.exists(name):
        print(f'Model not found')
        return
    else:
        if not os.path.exists(f'{name}/loss.png'):
            print(f'Incomplete Model Training')
            return

        if not os.path.exists(f'{name}/prob{ckpt}.npy'):
            print(f'Probability Not Found')
            return

    with open(os.path.join(name, f'prob{ckpt}.npy'), 'rb') as f:
        prob = np.load(f, allow_pickle=True)
        prob = prob / np.sum(prob)

    prob = prob * rules
    cost = 1 - prob
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.zeros((rules, rules))
    perm[row_ind, col_ind] = 1.
    hung_score = np.sum(np.abs(perm - prob)) / (2 * rules)

    print(f'Hungarian Score: {hung_score}')
    with open(os.path.join(name, f'hung{ckpt}.txt'), 'w') as f:
        f.write(f'Hungarian Score: {hung_score}')

models = ['Monolithic', 'GT_Modular', 'Modular', 'Modular_operation-only']
rules = [2, 4, 8, 16, 32]
encs = [32, 64, 128, 256, 512]
dims = [128, 256, 512, 1024, 2048]

for r in rules:
    for enc, dim in zip(encs, dims):
        for model in models:
            for ckpt in ['_last', '_best']:
                for seed in range(25):
                    metrics(r, model, enc, dim, seed, ckpt)
