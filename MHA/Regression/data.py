import numpy as np

def search_v1(x):
    deltas = np.abs(np.expand_dims(x, axis=1) - np.expand_dims(x, axis=2)) # (num_points x length x length)
    l = np.arange(deltas.shape[1])
    deltas[:, l, l] = float('inf')

    indices = np.argmin(deltas, 2)

    return indices

def search_v2(x):
    deltas = np.expand_dims(x, axis=1) * np.expand_dims(x, axis=2) # (num_points x length x length)
    deltas = np.sum(deltas, axis=-1)
    l = np.arange(deltas.shape[1])
    deltas[:, l, l] = float('-inf')

    indices = np.argmax(deltas, 2)

    return indices

def retrieve(data, winners, index):
    indexing = np.expand_dims(np.arange(winners.shape[0]), axis=-1)
    indexing = np.repeat(indexing, repeats=winners.shape[1], axis=1)

    return data[indexing, winners, index]

def sum(a, b):
    return a + b

def subtract(a, b):
    return a - b

def max(a, b):
    return np.maximum(a, b)

def min(a, b):
    return np.minimum(a, b)

def product(a, b):
    return a * b

def coeff_sum(a, b, c1, c2):
    return c1 * a + c2 * b

def rule_v1(data, search1, search2, retrieve1, retrieve2, func, search=search_v1):
    s1 = search(data[:,:,search1])
    out1 = retrieve(data, s1, retrieve1)

    s2 = search(data[:,:,search2])
    out2 = retrieve(data, s2, retrieve2)

    return func(out1, out2)

def rule_v2(data, search1, search2, retrieve1, retrieve2, c1=None, c2=None, search=search_v1):
    s1 = search(data[:, :, search1])
    out1 = retrieve(data, s1, retrieve1)

    s2 = search(data[:, :, search2])
    out2 = retrieve(data, s2, retrieve2)

    return coeff_sum(out1, out2, c1, c2)

def onehot(task, num_rules):
    task_onehot = np.zeros((task.size, num_rules))
    task_onehot[np.arange(task.size), task] = 1.
    return task_onehot

def rules(num_points, length, num_rules, version=2, search_version=1, data_seed=0, ood=False, prob=None):
    rng = np.random.RandomState(data_seed)
    coeff1 = rng.randn(num_rules)
    coeff2 = rng.randn(num_rules)

    if version == 1:
        rule_functions = [sum, subtract, product, min, max]
    elif version == 2:
        pass
    else:
        print("Wrong Data Version")
        exit()

    if search_version == 1:
        data = np.random.randn(num_points, length, 4 * num_rules)
    elif search_version == 2:
        search_data = np.random.randn(num_points, length, num_rules, 2, 2)
        data = np.random.randn(num_points, length, num_rules, 2)
        search_data /= np.expand_dims(np.linalg.norm(search_data, axis=-1), axis=-1)
        search_data = np.reshape(search_data, (num_points, length, num_rules, 4))
        data = np.reshape(np.concatenate((search_data, data), axis=3), (num_points, length, 6 * num_rules))
    else:
        print("Wrong Search Version")
        exit()

    if ood:
        data = data * 2

    if prob is not None:
        task = np.random.choice(num_rules, num_points * length, p=prob)
    else:
        task = np.random.choice(num_rules, num_points * length)
    task = np.reshape(onehot(task, num_rules), (num_points, length, num_rules))
    hyp = np.zeros([num_points, length, num_rules])

    for r in range(num_rules):
        if version == 1 and search_version == 1:
            hyp[:, :, r] = rule_v1(data, 4 * r, 4 * r + 1, 4 * r + 2, 4 * r + 3, rule_functions[r], search_v1)
        elif version == 1 and search_version == 2:
            hyp[:, :, r] = rule_v1(data, [6 * r, 6 * r + 1], [6 * r + 2, 6 * r + 3], 6 * r + 4, 6 * r + 5, rule_functions[r], search_v2)
        elif version == 2 and search_version == 1:
            hyp[:, :, r] = rule_v2(data, 4 * r, 4 * r + 1, 4 * r + 2, 4 * r + 3, coeff1[r], coeff2[r], search_v1)
        elif version == 2 and search_version == 2:
            hyp[:, :, r] = rule_v2(data, [6 * r, 6 * r + 1], [6 * r + 2, 6 * r + 3], 6 * r + 4, 6 * r + 5, coeff1[r], coeff2[r], search_v2)
        else:
            print("Wrong data parameters")
            exit()

    labels = np.sum(hyp * task, axis=-1)
    samples = np.concatenate((data, task), axis=-1)
    return samples, labels, task