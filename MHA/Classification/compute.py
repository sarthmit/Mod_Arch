import numpy as np
import os
import pandas as pd
import pickle

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import argparse
parser = argparse.ArgumentParser(description='Rule MLP')
parser.add_argument('--sver', type=int, default=1, choices=(1,2))
parser.add_argument('--rule', type=int, default=2, choices=(2,4,8,16,32))
parser.add_argument('--mode', type=str, default='last', choices=('last','best'))
args = parser.parse_args()

sver = [1, 2]
rules = [2, 4, 8, 16, 32]
ds = [0, 1, 2, 3, 4]
encs = [32, 64, 128, 256, 512]
dims = [128, 256, 512, 1024, 2048]
models = ['GT_Modular', 'Modular_operation-only', 'Monolithic', 'Modular']
modes = ['last', 'best']

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

mapping = {
    'GT_Modular': 0,
    'Modular': 1,
    'Modular_operation-only': 2,
    'Monolithic': 3
}

mapping_sub = {
    'Modular': 0,
    'Monolithic': 1
}

def get_min(avg_perf, model, min_perf, min_name):
    if avg_perf > min_perf:
        min_perf = avg_perf
        min_name = model

    return min_perf, min_name


df_final = pd.DataFrame(columns=['Mode', 'Model', 'Rules', 'Search-Version', 'Encoder Dimension',
                                 'Dimension', 'Data Seed', 'Seed', 'Number of Parameters',
                                 'Perf-3', 'Perf-5', 'Perf-10', 'Perf-20', 'Perf-30',
                                 'Perf-OoD-3', 'Perf-OoD-5', 'Perf-OoD-10', 'Perf-OoD-20', 'Perf-OoD-30',
                                 'Collapse', 'Collapse Worst', 'Specialization', 'Mutual Information',
                                 'Hungarian Score', 'Eval Mode'])

df_training = pd.DataFrame(columns=['Mode', 'Model', 'Rule', 'Search-Version', 'Encoder Dimension',
                                    'Dimension', 'Data Seed', 'Seed', 'Number of Parameters',
                                    'Iteration', 'Perf'])

for sv in sver:
    print(f'Search-Version: {sv}')
    rank = get_ranking_init()
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
                            if 'Monolithic' in model:
                                name = f'Sequence_10/Search-Version_{sv}/Data-Seed_{ds}/GT_Rules_{r}/{model}_{enc}_{dim}_{r*2}_{r}_{seed}'
                            else:
                                name = f'Sequence_10/Search-Version_{sv}/Data-Seed_{ds}/GT_Rules_{r}/{model}_{enc}_{dim}_2_{r}_{seed}'

                            # Get number of parameters
                            with open(f'{name}/log.txt', 'r') as f:
                                data = f.read().split('\n')
                                params = float(data[0].split(':')[-1])
                                for d in data[3:-1:2]:
                                    iter = float(d.split('|')[0].split(':')[-1])
                                    perf = float(d.split(':')[-1].split('|')[0])

                                    df_training.loc[-1] = ['MHA-Classification', model, r, sv, enc, dim,
                                                        ds, seed, params, iter, perf]
                                    df_training.index = df_training.index + 1

                            # Get perf and OoD perf
                            with open(f'{name}/perf_{mode}.txt', 'r') as f:
                                data = f.read().split('\n')[:-1]
                                id = data[0].split('|')[1:-1]
                                id = [float(x.split(':')[-1]) for x in id]
                                ood = data[1].split('|')[1:-1]
                                ood = [float(x.split(':')[-1]) for x in ood]

                                perf = float(data[0].split(':')[-1].split('|')[0])
                                perf_ood = float(data[1].split(':')[-1].split('|')[0])

                                avg_perf_ood += ood[-1] / 5.
                                avg_perf += id[2] / 5.

                            if 'Monolithic' in model:
                                collapse, collapse_worst, spec, mi = np.nan, np.nan, np.nan, np.nan
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

                            df_final.loc[-1] = ['MHA-Classification', model, r, sv, enc, dim,
                                                 ds, seed, params, id[0], id[1], id[2], id[3], id[4],
                                                 ood[0], ood[1], ood[2], ood[3], ood[4], collapse,
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

    with open(f'Logistics/ranking_{sv}.pickle', 'wb') as handle:
        pickle.dump(rank, handle, protocol=pickle.HIGHEST_PROTOCOL)

df_final["Model"] = df_final["Model"].replace({'Modular_operation-only': 'Modular-op', 'GT_Modular': 'GT-Modular'})
df_training["Model"] = df_training["Model"].replace({'Modular_operation-only': 'Modular-op', 'GT_Modular': 'GT-Modular'})
df_training = df_training.drop_duplicates()

df_final.to_pickle(f'Logistics/df_final.pt')
df_training.to_pickle(f'Logistics/df_training.pt')

print('Logistics Computed')