import os

rules = [2, 4, 8, 16, 32]
ds = [0, 1, 2, 3, 4]
encs = [32, 64, 128, 256, 512]
dims = [128, 256, 512, 1024, 2048]
models = ['GT_Modular', 'Modular', 'Modular_operation-only', 'Modular_joint', 'Monolithic']
direcs = ['Classification/', 'Regression/']

files = ['perf_last.txt', 'perf_best.txt', 'loss.png', 'specialization_last.txt', 'specialization_best.txt', 'prob_last.npy',
         'prob_best.npy']

def check_files(name):
    if 'Monolithic' in name:
        trial_files = files[:3]
    else:
        trial_files = files
    for file in trial_files:
        if not os.path.exists(f'{name}/{file}'):
            print(file)
            print(f'Missing File: {name}')
            break

def check_log(name):
    with open(f'{name}/log.txt', 'r') as f:
        data = f.read()
        if 'Iteration: 100000' not in data:
            print(f'Log Incomplete: {name}')

        if len(data.split('\n')[3:-1:2]) != 21:
            print(f'Weird Log: {name}')

for direc in direcs:
    for r in rules:
        print(f'Rules: {r}')
        for model in models:
            for enc, dim in zip(encs, dims):
                for seed in range(25):
                    d = seed % 5
                    s = seed // 5
                    name = f'{direc}Logs/Data-Seed_{d}/GT_Rules_{r}/{model}_{enc}_{dim}_{r}_{s}'
                    check_files(name)
                    check_log(name)