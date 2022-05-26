# Script to generate data

import torch
import numpy as np

def sum(a, b):
    return a + b

def subtract(a, b):
    return a - b

def product(a, b):
    return a * b

def min(a, b):
    return np.minimum(a, b)

def max(a, b):
    return np.maximum(a, b)

def onehot(task, num_rules):
    task_onehot = np.zeros((task.size, num_rules))
    task_onehot[np.arange(task.size), task] = 1.
    return task_onehot

def data_v1(num_examples, num_rules=2, data_seed=None, ood=False, prob=None):
    a = np.random.randn(num_examples, 1)
    b = np.random.randn(num_examples, 1)

    if ood:
        a = a * 2
        b = b * 2

    rules = [sum, subtract, product, min, max]

    if prob is not None:
        task = np.random.choice(num_rules, size=num_examples, p=prob)
    else:
        task = np.random.choice(num_rules, size=num_examples)
    task = onehot(task, num_rules)

    result = np.zeros([num_examples, num_rules])

    for r in range(num_rules):
        result[:, r] = rules[r](a, b)[:,0] >= 0.

    result = np.sum(result * task, axis=-1)
    sample = np.concatenate((a, b, task), axis=-1)

    return sample, result

def data_v2(num_examples, num_rules, data_seed, ood=False, prob=None):
    rng = np.random.RandomState(data_seed)
    coeff1 = rng.randn(num_rules)
    coeff2 = rng.randn(num_rules)

    a = np.random.randn(num_examples, 1)
    b = np.random.randn(num_examples, 1)

    if ood:
        a = a * 2
        b = b * 2

    if prob is not None:
        task = np.random.choice(num_rules, size=num_examples, p=prob)
    else:
        task = np.random.choice(num_rules, size=num_examples)
    task = onehot(task, num_rules)

    result = np.zeros([num_examples, num_rules])

    for r in range(num_rules):
        result[:, r] = (coeff1[r] * a + coeff2[r] * b)[:,0] >= 0.

    result = np.sum(result * task, axis=-1)
    sample = np.concatenate((a, b, task), axis=-1)

    return sample, result

if __name__ == '__main__':
    sample, result = data_v1(10, 3)
    print(sample)
    print(result)

    sample, result = data_v2(10, 10, 0)
    print(sample)
    print(result)