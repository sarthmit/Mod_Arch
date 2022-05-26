import numpy as np

def onehot(task, num_rules):
    task_onehot = np.zeros((task.size, num_rules))
    task_onehot[np.arange(task.size), task] = 1.
    return task_onehot

def rules(num_points, length, num_rules, order, dim, data_seed=0, ood=False, prob=None):
    rng = np.random.RandomState(data_seed)
    coeff1 = rng.randn(num_rules, order, dim, dim) / np.sqrt(order * dim)
    coeff2 = rng.randn(num_rules, dim, dim) / np.sqrt(dim)

    if prob is not None:
        task = np.random.choice(num_rules, num_points * length, p=prob)
    else:
        task = np.random.choice(num_rules, num_points * length)

    task = np.reshape(onehot(task, num_rules), (num_points, length, num_rules))
    data = np.random.randn(num_points, length, dim)
    states = np.zeros((num_points, length, dim))

    if ood:
        data = data * 2

    for l in range(length):
        d = np.zeros((num_points, dim, num_rules))

        for r in range(num_rules):
            d[:, :, r] += np.matmul(data[:, l, :], coeff2[r, :, :])

        for o in range(1, order + 1):
            if (l - o) < 0:
                break

            for r in range(num_rules):
                d[:, :, r] += np.matmul(states[:, l-o, :], coeff1[r, o-1, :, :])

        states[:, l, :] = np.sum(d * task[:, l:l+1, :], axis=-1)

    inp = np.concatenate([data, task], axis=-1)

    return inp, states, task

if __name__ == '__main__':
    dim = 4

    for r in [2, 4, 8, 16, 32]:
        for d in range(5):
            d, l, t = rules(1, 10, r, 1, dim, d)
            print(l[-1, -1, :])