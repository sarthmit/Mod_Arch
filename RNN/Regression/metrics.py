# Eval file

import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description='Rule MLP')
args = parser.parse_args()

EPS = 1e-8

def collapse_metric_worse(prob, rules):
    p = np.min(np.sum(prob, axis=0))
    cmw = 1 - rules * p
    return cmw

def collapse_metric(prob, rules):
    p = np.sum(prob, axis=0)
    cm = rules * np.sum(np.maximum(np.ones_like(p) / rules - p, 0)) / (rules - 1)
    return cm

def mutual_info(prob):
    m1 = np.sum(prob, axis=0, keepdims=True)
    m2 = np.sum(prob, axis=1, keepdims=True)
    m = m1 * m2
    return np.sum(prob * np.log(prob / (m + EPS) + EPS))

def emp_prob(prob):
    p = np.sum(prob, axis=0)
    p.sort()
    return p

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

    # Compute Collapse Metric
    collapse = collapse_metric(prob, rules)
    print(f'Collapse Metric: {collapse}')
    with open(os.path.join(name, f'collapse{ckpt}.txt'), 'w') as f:
        f.write(f'Collapse Metric: {collapse}')

    # Compute Collapse Worst Metric
    collapse_worst = collapse_metric_worse(prob, rules)
    print(f'Collapse Worst Metric: {collapse_worst}')
    with open(os.path.join(name, f'collapse_worst{ckpt}.txt'), 'w') as f:
        f.write(f'Collapse Worst Metric: {collapse_worst}')

    # Compute Mutual Information
    mi = mutual_info(prob)
    print(f'Mutual Information: {mi}')
    with open(os.path.join(name, f'mi{ckpt}.txt'), 'w') as f:
        f.write(f'Mutual Information: {mi}')

    # Activation Probability
    emp = emp_prob(prob)
    emp = ' | '.join([str(x) for x in emp])
    print(f'Empirical Probability: {emp}')
    with open(os.path.join(name, f'emp{ckpt}.txt'), 'w') as f:
        f.write(emp)

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
