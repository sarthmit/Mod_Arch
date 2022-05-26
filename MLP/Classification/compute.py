import numpy as np
import os
import pandas as pd
import pickle

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

rules = [2, 4, 8, 16, 32]
ds = [0, 1, 2, 3, 4]
encs = [32, 64, 128, 256, 512]
dims = [128, 256, 512, 1024, 2048]
models = ['GT_Modular', 'Modular_operation-only', 'Monolithic', 'Modular_joint']
modes = ['last', 'best']

df_final = pd.DataFrame(columns=['Mode', 'Model', 'Rules', 'Encoder Dimension',
                           'Dimension', 'Data Seed', 'Seed', 'Number of Parameters',
                           'Perf', 'Perf-OoD', 'Collapse', 'Collapse Worst',
                           'Specialization', 'Mutual Information', 'Hungarian Score', 'Eval Mode'])

df_training = pd.DataFrame(columns=['Mode', 'Model', 'Rule', 'Encoder Dimension',
                           'Dimension', 'Data Seed', 'Seed', 'Number of Parameters',
                           'Iteration', 'Perf'])

def get_ranking_init():
    rank = dict()
    rank['perf_last'] = np.zeros(4)
    rank['perf_best'] = np.zeros(4)
    rank['perf_ood_last'] = np.zeros(4)
    rank['perf_ood_best'] = np.zeros(4)

    rank['perf_sub_last'] = np.zeros(2)
    rank['perf_ood_sub_last'] = np.zeros(2)
    rank['perf_sub_best'] = np.zeros(2)
    rank['perf_ood_sub_best'] = np.zeros(2)

    return rank

rank = get_ranking_init()

mapping = {
    'GT_Modular': 0,
    'Modular_joint': 1,
    'Modular_operation-only': 2,
    'Monolithic': 3
}

mapping_sub = {
    'Modular_joint': 0,
    'Monolithic': 1
}

def get_min(avg_perf, model, min_perf, min_name):
    if avg_perf > min_perf:
        min_perf = avg_perf
        min_name = model

    return min_perf, min_name

for mode in modes:
    print(f'Mode: {mode}')
    for r in rules:
        print(f'Rule: {r}')
        for enc, dim in zip(encs, dims):
            print(f'Enc: {enc} | Dim: {dim}')
            for ds in range(5):
                min_perf, min_perf_ood, min_perf_sub, min_perf_ood_sub = float('-inf'), float('-inf'), float('-inf'), float('-inf')
                min_name ,min_name_ood, min_name_sub, min_name_ood_sub = None, None, None, None

                for model in models:
                    avg_perf = 0.
                    avg_perf_ood = 0.

                    for seed in range(5):
                        name = f'Logs/Data-Seed_{ds}/GT_Rules_{r}/{model}_{enc}_{dim}_{r}_{seed}'

                        # Get number of parameters
                        with open(f'{name}/log.txt', 'r') as f:
                            data = f.read().split('\n')
                            params = float(data[0].split(':')[-1])
                            for d in data[3:-1:2]:
                                iter = float(d.split('|')[0].split(':')[-1])
                                perf = float(d.split(':')[-1].split('|')[0])

                                df_training.loc[-1] = ['MLP-Classification-WithDecoder', model, r, enc, dim,
                                                    ds, seed, params, iter, perf]
                                df_training.index = df_training.index + 1

                        # Get perf and OoD perf
                        with open(f'{name}/perf_{mode}.txt', 'r') as f:
                            data = f.read().split('\n')[:-1]
                            perf = float(data[0].split(':')[-1].split('|')[0])
                            perf_ood = float(data[1].split(':')[-1].split('|')[0])

                            avg_perf_ood += perf_ood / 5.
                            avg_perf += perf / 5.

                        if 'Monolithic' in model:
                            collapse, collapse_worst, spec, mi, hung = np.nan, np.nan, np.nan, np.nan, np.nan
                        else:
                            # Get Specialization Metric
                            with open(f'{name}/specialization_{mode}.txt', 'r') as f:
                                spec = float(f.read().split('\n')[0].split(':')[-1])

                            # Get Collapse Metric
                            with open(f'{name}/collapse_{mode}.txt', 'r') as f:
                                collapse = float(f.read().split('\n')[0].split(':')[-1])

                            # Get Collapse Worst Metric
                            with open(f'{name}/collapse_worst_{mode}.txt', 'r') as f:
                                collapse_worst = float(f.read().split('\n')[0].split(':')[-1])

                            # Get MI Metric
                            with open(f'{name}/mi_{mode}.txt', 'r') as f:
                                mi = float(f.read().split('\n')[0].split(':')[-1])

                            # Get Hungarian Metric
                            with open(f'{name}/hung_{mode}.txt', 'r') as f:
                                hung = float(f.read().split('\n')[0].split(':')[-1])

                        df_final.loc[-1] = ['MLP-Classification-WithDecoder', model, r, enc, dim,
                                             ds, seed, params, perf, perf_ood, collapse,
                                             collapse_worst, spec, mi, hung, mode]
                        df_final.index = df_final.index + 1

                    min_perf, min_name = get_min(avg_perf, model, min_perf, min_name)
                    min_perf_ood, min_name_ood = get_min(avg_perf_ood, model, min_perf_ood, min_name_ood)

                    if not ('GT' in model or 'operation' in model):
                        min_perf_sub, min_name_sub = get_min(avg_perf, model, min_perf_sub, min_name_sub)
                        min_perf_ood_sub, min_name_ood_sub = get_min(avg_perf_ood, model, min_perf_ood_sub, min_name_ood_sub)

                rank[f'perf_{mode}'][mapping[min_name]] += 1
                rank[f'perf_ood_{mode}'][mapping[min_name_ood]] += 1
                rank[f'perf_sub_{mode}'][mapping_sub[min_name_sub]] += 1
                rank[f'perf_ood_sub_{mode}'][mapping_sub[min_name_ood_sub]] += 1

df_final["Model"] = df_final["Model"].replace({'Modular_joint': 'Modular', 'Modular_operation-only': 'Modular-op', 'GT_Modular': 'GT-Modular'})
df_training["Model"] = df_training["Model"].replace({'Modular_joint': 'Modular', 'Modular_operation-only': 'Modular-op', 'GT_Modular': 'GT-Modular'})
df_training = df_training.drop_duplicates()

df_final.to_pickle('Logistics/df_final.pt')
df_training.to_pickle('Logistics/df_training.pt')
with open(f'Logistics/ranking.pickle', 'wb') as handle:
    pickle.dump(rank, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Logistics Computed')