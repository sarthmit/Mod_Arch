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

# def entropy(p):
#     return - np.sum(p * np.log(p))
#
# def conditional_entropy(prob, axis):
#     marginal = np.sum(prob, axis=axis, keepdims=True)
#     return -np.sum(prob * np.log(prob / (marginal + EPS) + EPS))
#
# def mutual_info(prob):
#     return entropy(np.sum(prob, axis=1)) - conditional_entropy(prob, 1)

def emp_prob(prob):
    p = np.sum(prob, axis=0)
    p.sort()
    return p

def get_name(sv, rules, model, enc_dim, dim, seed):
    d = seed % 5
    s = seed // 5

    if 'Monolithic' in model:
        heads = 2 * rules
    else:
        heads = 2

    return f'Sequence_10/Search-Version_{sv}/Data-Seed_{d}/GT_Rules_{rules}/{model}_{enc_dim}_{dim}_{heads}_{rules}_{s}'

def metrics(sv, rules, model, enc_dim, dim, seed, ckpt='_last'):
    name = get_name(sv, rules, model, enc_dim, dim, seed)
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

for sv in range(1,3):
    for r in rules:
        for enc, dim in zip(encs, dims):
            for model in models:
                for ckpt in ['_last', '_best']:
                    for seed in range(25):
                        metrics(sv, r, model, enc, dim, seed, ckpt)