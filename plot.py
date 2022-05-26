# Compute Plots for different purposes

import seaborn as sns
import matplotlib as mpl
from matplotlib import font_manager
from scipy.optimize import linear_sum_assignment
font_manager._rebuild()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

EPS = 1e-8
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Set Plotting Variables

sns.color_palette("dark", as_cmap=True)
sns.set(style="darkgrid", font_scale=1.2)
font = {'family' : 'Open Sans',
        'size'   : 12}

mpl.rc('font', **font)
colors = ['#31356e', '#704e85', '#a66c98', '#d690aa']
explode=[0.05]*4

model_names = ['GT-Modular', 'Modular-op', 'Modular', 'Monolithic']

def add_random(df, mode, arch='MLP'):
    rules = [2, 4, 8, 16, 32]
    encs = [32, 64, 128, 256, 512]
    dims = [128, 256, 512, 1024, 2048]
    modes = ['last', 'best']

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

    def specialization_score(true_p, empirical_p):
        true_p.sort()
        empirical_p.sort()

        return np.sum(np.abs(true_p - empirical_p)) / 2.

    def specialization_metric(prob, rules):
        spec = 0.
        p_ = np.sum(prob, axis=0)

        for eval_seed in range(100):
            rng = np.random.RandomState(eval_seed)
            p = rng.dirichlet(alpha = np.ones(rules))
            spec += specialization_score(p, p_)

        return spec / 100.

    def hungarian_metric(prob, rules):
        prob = prob * rules
        cost = 1 - prob
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = np.zeros((rules, rules))
        perm[row_ind, col_ind] = 1.
        hung_score = np.sum(np.abs(perm - prob)) / (2 * rules)
        return hung_score

    for rule in [2, 4, 8, 16, 32]:
        prob = np.ones((rule, rule)) / (rule * rule)
        cmw = collapse_metric_worse(prob, rule)
        cm = collapse_metric(prob, rule)
        mi = mutual_info(prob)
        spec = specialization_metric(prob, rule)
        hung = hungarian_metric(prob, rule)

        for enc, dim in zip(encs, dims):
            for m in modes:
                if arch == 'MLP':
                    df.loc[-1] = [mode, 'Random', rule, enc, dim, -1, -1, -1, -1, -1,
                                  cm, cmw, spec, mi, hung, m]
                elif arch == 'MHA':
                    df.loc[-1] = [mode, 'Random', rule, 1, enc, dim, -1, -1, -1, -1,
                                  -1, -1, -1, -1, -1, -1, -1, -1, -1, cm, cmw, spec, mi, hung, m]
                    df.index += 1
                    df.loc[-1] = [mode, 'Random', rule, 2, enc, dim, -1, -1, -1, -1,
                                  -1, -1, -1, -1, -1, -1, -1, -1, -1, cm, cmw, spec, mi, hung, m]
                elif arch == 'RNN':
                    df.loc[-1] = [mode, 'Random', rule, enc, dim, -1, -1, -1, -1,
                                  -1, -1, -1, -1, -1, -1, -1, -1, -1, cm, cmw, spec, mi, hung, m]
                df.index += 1

def plot_line(df, x, y, col, row, hue, name, ci=None, hue_order = model_names, title=None, xlabel = None, ylabel = None, marker=None):
    g = sns.relplot(
        data=df, x=x, y=y, col=col, row=row,
        hue=hue, kind="line", facet_kws={'sharey': False},
        hue_order=hue_order, ci=ci, marker=marker
    )

    if row != None:
        sns.move_legend(
            g,
            loc=(0.3, -0.05),
            ncol=4,
            title=None,
            frameon=False,
            fontsize=18,
            borderaxespad=3,
            borderpad=3
        )

        (g.set_titles(title, size=20)
         .set_ylabels(ylabel, clear_inner=False, size=20)
         .set_xlabels(xlabel, clear_inner=False, size=20)
         .despine(left=True))

        for i, ax in enumerate(g.axes[0]):
            plt.setp(ax.get_xticklabels(), fontsize=16)
            plt.setp(ax.get_yticklabels(), fontsize=16)
            ax.set_xlabel('', fontsize=24)

        for i, ax in enumerate(g.axes[1]):
            plt.setp(ax.get_xticklabels(), fontsize=16)
            plt.setp(ax.get_yticklabels(), fontsize=16)

            if i != 2:
                ax.set_xlabel('', fontsize=24)

    elif col == None:
        sns.move_legend(
            g,
            loc=(0.01, -0.12),
            ncol=2,
            title=None,
            frameon=False,
            fontsize=17,
            borderaxespad=3,
            borderpad=3
        )

        (g.set_titles(title, size=17)
         .set_ylabels(ylabel, clear_inner=False, size=17)
         .set_xlabels(xlabel, clear_inner=False, size=17)
         .despine(left=True))
        plt.xticks(size=14)
        plt.yticks(size=14)
    else:
        (g.set_titles(title, size=20)
         .set_ylabels(ylabel, clear_inner=False, size=20)
         .set_xlabels(xlabel, clear_inner=False, size=20)
         .despine(left=True))

        sns.move_legend(
            g,
            loc=(0.3, -0.07),
            ncol=4,
            title=None,
            frameon=False,
            fontsize=18,
            borderaxespad=2,
            borderpad=2
        )

        for i, ax in enumerate(g.axes[0]):
            plt.setp(ax.get_xticklabels(), fontsize=16)
            plt.setp(ax.get_yticklabels(), fontsize=16)

            if i != 2:
                ax.set_xlabel('', fontsize=24)

    for ax in g.axes[0]:
        for index, label in enumerate(ax.get_yticklabels()):
            if index % 2 == 0:
                label.set_visible(False)
            else:
                label.set_visible(True)

    plt.savefig(name, bbox_inches='tight')
    plt.close()

def plot_log_line(df, x, y, col, row, hue, name, ci = None, hue_order = model_names, title = None, xlabel = None, ylabel = None):
    g = sns.relplot(
        data=df, x=x, y=y, col=col, row=row,
        hue=hue, kind="line", facet_kws={'sharey': False},
        hue_order=hue_order, ci=ci,
        marker='o'
    )

    if xlabel is None:
        xlabel = x
    if ylabel is None:
        ylabel = y

    g.set(xscale="log")

    if row != None:
        sns.move_legend(
            g,
            loc=(0.3, -0.05),
            ncol=4,
            title=None,
            frameon=False,
            fontsize=17,
            borderaxespad=3,
            borderpad=3
        )

        (g.set_titles(title, size=20)
         .set_ylabels(ylabel, clear_inner=False, size=20)
         .set_xlabels(xlabel, clear_inner=False, size=20)
         .despine(left=True))

        for i, ax in enumerate(g.axes[0]):
            plt.setp(ax.get_xticklabels(), fontsize=16)
            plt.setp(ax.get_yticklabels(), fontsize=16)
            ax.set_xlabel('', fontsize=24)

        for i, ax in enumerate(g.axes[1]):
            plt.setp(ax.get_xticklabels(), fontsize=16)
            plt.setp(ax.get_yticklabels(), fontsize=16)

            if i != 2:
                ax.set_xlabel('', fontsize=24)

    elif col == None:
        sns.move_legend(
            g,
            loc=(0.01, -0.12),
            ncol=2,
            title=None,
            frameon=False,
            fontsize=17,
            borderaxespad=3,
            borderpad=3
        )

        (g.set_titles(title, size=17)
         .set_ylabels(ylabel, clear_inner=False, size=17)
         .set_xlabels(xlabel, clear_inner=False, size=17)
         .despine(left=True))
        plt.xticks(size=14)
        plt.yticks(size=14)
    else:
        (g.set_titles(title, size=20)
         .set_ylabels(ylabel, clear_inner=False, size=20)
         .set_xlabels(xlabel, clear_inner=False, size=20)
         .despine(left=True))

        sns.move_legend(
            g,
            loc=(0.3, -0.07),
            ncol=4,
            title=None,
            frameon=False,
            fontsize=18,
            borderaxespad=2,
            borderpad=2
        )

        for i, ax in enumerate(g.axes[0]):
            plt.setp(ax.get_xticklabels(), fontsize=16)
            plt.setp(ax.get_yticklabels(), fontsize=16)

            if i != 2:
                ax.set_xlabel('', fontsize=24)

    for ax in g.axes[0]:
        for index, label in enumerate(ax.get_yticklabels()):
            if index % 2 == 0:
                label.set_visible(False)
            else:
                label.set_visible(True)

    plt.savefig(name, bbox_inches='tight')
    plt.close()

def plot_metrics(df, x, y, hue, name, hue_order = model_names + ['Random'], dodge=True):
    d = df[df['Model'] != 'Monolithic']
    g = sns.barplot(data=d, x=x, y=y, hue=hue,
                    hue_order=hue_order,
                    alpha=0.9, saturation=0.8, dodge=dodge)

    if x is None or x == hue:
        g.set(xticklabels=[])
        g.set(xlabel=None)
        g.tick_params(bottom=False, size=16)
        g.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.3), facecolor="white", edgecolor='white',
                 fontsize=17)
    else:
        g.set_xlabel(x, size=20)
        plt.xticks(fontsize=16)
        g.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.45), facecolor="white", edgecolor='white', fontsize=18)

    g.set_ylabel(y, size=17)
    plt.yticks(fontsize=16)

    for index, label in enumerate(g.get_yticklabels()):
        if index % 2 == 0:
            label.set_visible(False)
        else:
            label.set_visible(True)

    plt.savefig(name, bbox_inches='tight')
    plt.close()

def plot_metrics_full(df, x, y, col, row, hue, name, hue_order = model_names):
    d = df[df['Model'] != 'Monolithic']
    g = sns.catplot(data=d, x=x, y=y, col=col, row=row, hue=hue,
                kind='bar', order=['Small', 'Medium', 'Large'],
                hue_order=hue_order, alpha=0.9, saturation=0.8)

    g.set_xlabels(x, size=24)
    g.set_ylabels(y, size=24)
    g.set_titles("Rules = {col_name}", size=24)

    g.set_xticklabels(size=20)
    g.set_yticklabels(size=20)

    for i, ax in enumerate(g.axes[0]):
        if i != 2:
            ax.set_xlabel('', fontsize=24)

    for ax in g.axes[0]:
        for index, label in enumerate(ax.get_yticklabels()):
            if index % 2 == 0:
                label.set_visible(False)
            else:
                label.set_visible(True)

    if row != None:
        for i, ax in enumerate(g.axes[1]):
            if i != 2:
                ax.set_xlabel('', fontsize=24)

        sns.move_legend(
            g,
            loc=(0.23, -0.075),
            ncol=4,
            title=None,
            frameon=False,
            fontsize=24,
            borderpad=3
        )
    else:
        sns.move_legend(
            g,
            loc=(0.23, -0.13),
            ncol=4,
            title=None,
            frameon=False,
            fontsize=24,
            borderpad=3
        )

    plt.savefig(name, bbox_inches='tight')
    plt.close()

def plot_pie(arr, labels, colors, name, circle=True):
    def filter(arr, lab, color):
        a = []
        l = []
        c = []
        for i in range(len(arr)):
            if arr[i] == 0:
                continue
            a.append(arr[i])
            l.append(lab[i])
            c.append(color[i])

        return a, l, c

    arr, lab, col = filter(arr, labels, colors)
    _, texts, autotexts = plt.pie(arr,
                                  colors=col,
                                  labels=lab,
                                  autopct='%.1f%%',
                                  startangle=90,
                                  pctdistance=0.4
                                  )

    for text in texts:
        text.set_color('#404040')
        if not circle:
            text.set_color('#000000')
        text.set_fontsize(22)
    for autotext in autotexts:
        autotext.set_color('#404040')
        if not circle:
            autotext.set_color('#000000')
        autotext.set_fontsize(22)

    # draw circle
    if circle:
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.gca().axis('equal')
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def plot_full(df_training, df_final, rank, name):
    if 'Classification' in name:
        ylabel = 'Error'
    else:
        ylabel = 'Loss'

    df_final = df_final.rename(columns={
        'Collapse': 'Collapse-Avg',
        'Collapse Worst': 'Collapse-Worst',
        'Specialization': 'Adaptation',
        'Mutual Information': 'Inverse Mutual Information',
        'Hungarian Score': 'Alignment'
    })
    print(list(df_final))

    df_training_mini = df_training[df_training['Iteration'] > 0]
    df_final = df_final[df_final['Eval Mode'] == 'last']

    hue_perf = ['GT-Modular', 'Modular-op', 'Modular', 'Monolithic']
    hue_metrics = ['GT-Modular', 'Modular-op', 'Modular', 'Random']
    hue_pie = ['GT-Modular', 'Modular', 'Modular-op', 'Monolithic']
    hue_pie_sub = ['Modular', 'Monolithic']

    if 'MHA' in name:
        plot_line(df_training, 'Iteration', 'Perf', 'Rule', 'Search-Version', 'Model', f'{name}/training_rs.pdf',
                  title='Search-Version: {row_name} | Rules: {col_name}', xlabel='Iterations', ylabel=ylabel, hue_order=hue_perf)
    plot_line(df_training, 'Iteration', 'Perf', 'Rule', None, 'Model', f'{name}/training_r.pdf',
              title='Rules: {col_name}', xlabel='Iterations', ylabel=ylabel, hue_order=hue_perf)
    plot_line(df_training, 'Iteration', 'Perf', None, None, 'Model', f'{name}/training.pdf',
              title=None, xlabel='Iterations', ylabel=ylabel, hue_order=hue_perf)

    if 'MHA' in name:
        plot_line(df_training_mini, 'Iteration', 'Perf', 'Rule', 'Search-Version', 'Model', f'{name}/training_mrs.pdf',
                  title='Search-Version: {row_name} | Rules: {col_name}', xlabel='Iterations', ylabel=ylabel, hue_order=hue_perf)
    plot_line(df_training_mini, 'Iteration', 'Perf', 'Rule', None, 'Model', f'{name}/training_mr.pdf',
              title='Rules: {col_name}', xlabel='Iterations', ylabel=ylabel, hue_order=hue_perf)
    plot_line(df_training_mini, 'Iteration', 'Perf', None, None, 'Model', f'{name}/training_m.pdf',
              title=None, xlabel='Iterations', ylabel=ylabel, hue_order=hue_perf)

    plot_line(df_final, 'Rules', 'Perf', None, None, 'Model', f'{name}/perf_r.pdf',
              title=None, xlabel='Rules', ylabel=ylabel, hue_order=hue_perf, marker='o')
    plot_line(df_final, 'Rules', 'Perf-OoD', None, None, 'Model', f'{name}/perf_ood_r.pdf',
              title=None, xlabel='Rules', ylabel=ylabel, hue_order=hue_perf, marker='o')

    if 'MHA' in name:
        for i in [3, 5, 10, 20, 30]:
            plot_log_line(df_final, 'Number of Parameters', f'Perf-{i}', 'Rules', 'Search-Version', 'Model',
                          f'{name}/perf_{i}_rs.pdf', title='Search-Version: {row_name} | Rules: {col_name}', xlabel=None,
                          ylabel=ylabel, hue_order=hue_perf)
            plot_log_line(df_final, 'Number of Parameters', f'Perf-OoD-{i}', 'Rules', 'Search-Version', 'Model',
                          f'{name}/perf_{i}_ood_rs.pdf', title='Search-Version: {row_name} | Rules: {col_name}',
                          xlabel=None, ylabel=ylabel, hue_order=hue_perf)
            plot_log_line(df_final, 'Dimension', f'Perf-{i}', 'Rules', 'Search-Version', 'Model',
                          f'{name}/perf_{i}_drs.pdf', title='Search-Version: {row_name} | Rules: {col_name}', xlabel=None,
                          ylabel=ylabel, hue_order=hue_perf)
            plot_log_line(df_final, 'Dimension', f'Perf-OoD-{i}', 'Rules', 'Search-Version', 'Model',
                          f'{name}/perf_{i}_ood_drs.pdf', title='Search-Version: {row_name} | Rules: {col_name}',
                          xlabel=None, ylabel=ylabel, hue_order=hue_perf)
    elif 'RNN' in name:
        for i in [3, 5, 10, 20, 30]:
            plot_log_line(df_final, 'Number of Parameters', f'Perf-{i}', 'Rules', None, 'Model',
                          f'{name}/perf_{i}_r.pdf', title='Rules: {col_name}', xlabel=None,
                          ylabel=ylabel, hue_order=hue_perf)
            plot_log_line(df_final, 'Number of Parameters', f'Perf-OoD-{i}', 'Rules', None, 'Model',
                          f'{name}/perf_{i}_ood_r.pdf', title='Rules: {col_name}',
                          xlabel=None, ylabel=ylabel, hue_order=hue_perf)
            plot_log_line(df_final, 'Dimension', f'Perf-{i}', 'Rules', None, 'Model',
                          f'{name}/perf_{i}_dr.pdf', title='Rules: {col_name}', xlabel=None,
                          ylabel=ylabel, hue_order=hue_perf)
            plot_log_line(df_final, 'Dimension', f'Perf-OoD-{i}', 'Rules', None, 'Model',
                          f'{name}/perf_{i}_ood_dr.pdf', title='Rules: {col_name}',
                          xlabel=None, ylabel=ylabel, hue_order=hue_perf)

    if 'MHA' in name:
        plot_log_line(df_final, 'Number of Parameters', 'Perf', 'Rules', 'Search-Version', 'Model',
                      f'{name}/perf_rs.pdf', title='Search-Version: {row_name} | Rules: {col_name}', xlabel=None, ylabel=ylabel, hue_order=hue_perf)
        plot_log_line(df_final, 'Number of Parameters', 'Perf-OoD', 'Rules', 'Search-Version', 'Model',
                  f'{name}/perf_ood_rs.pdf', title='Search-Version: {row_name} | Rules: {col_name}', xlabel=None, ylabel=ylabel, hue_order=hue_perf)
        plot_log_line(df_final, 'Dimension', 'Perf', 'Rules', 'Search-Version', 'Model',
                      f'{name}/perf_drs.pdf', title='Search-Version: {row_name} | Rules: {col_name}', xlabel=None, ylabel=ylabel, hue_order=hue_perf)
        plot_log_line(df_final, 'Dimension', 'Perf-OoD', 'Rules', 'Search-Version', 'Model',
                  f'{name}/perf_ood_drs.pdf', title='Search-Version: {row_name} | Rules: {col_name}', xlabel=None, ylabel=ylabel, hue_order=hue_perf)

    plot_log_line(df_final, 'Number of Parameters', 'Perf', 'Rules', None, 'Model',
                  f'{name}/perf_pr.pdf', title='Rules: {col_name}', xlabel=None, ylabel=ylabel, hue_order=hue_perf)
    plot_log_line(df_final, 'Number of Parameters', 'Perf-OoD', 'Rules', None, 'Model',
                  f'{name}/perf_ood_pr.pdf', title='Rules: {col_name}', xlabel=None, ylabel=ylabel, hue_order=hue_perf)
    plot_log_line(df_final, 'Number of Parameters', 'Perf', None, None, 'Model',
                  f'{name}/perf_p.pdf', title=None, xlabel=None, ylabel=ylabel, hue_order=hue_perf)
    plot_log_line(df_final, 'Number of Parameters', 'Perf-OoD', None, None, 'Model',
                  f'{name}/perf_ood_p.pdf', title=None, xlabel=None, ylabel=ylabel, hue_order=hue_perf)

    plot_log_line(df_final, 'Dimension', 'Perf', 'Rules', None, 'Model',
                  f'{name}/perf_dr.pdf', title='Rules: {col_name}', xlabel=None, ylabel=ylabel, hue_order=hue_perf)
    plot_log_line(df_final, 'Dimension', 'Perf-OoD', 'Rules', None, 'Model',
                  f'{name}/perf_ood_dr.pdf', title='Rules: {col_name}', xlabel=None, ylabel=ylabel, hue_order=hue_perf)
    plot_log_line(df_final, 'Dimension', 'Perf', None, None, 'Model',
                  f'{name}/perf_d.pdf', title=None, xlabel=None, ylabel=ylabel, hue_order=hue_perf)
    plot_log_line(df_final, 'Dimension', 'Perf-OoD', None, None, 'Model',
                  f'{name}/perf_ood_d.pdf', title=None, xlabel=None, ylabel=ylabel, hue_order=hue_perf)

    plot_metrics(df_final, 'Rules', 'Collapse-Worst', 'Model', f'{name}/cw_r.pdf', hue_order=hue_metrics)
    plot_metrics(df_final, 'Rules', 'Collapse-Avg', 'Model', f'{name}/ca_r.pdf', hue_order=hue_metrics)
    plot_metrics(df_final, 'Rules', 'Adaptation', 'Model', f'{name}/adap_r.pdf', hue_order=hue_metrics)
    plot_metrics(df_final, 'Rules', 'Inverse Mutual Information', 'Model', f'{name}/imi_r.pdf', hue_order=hue_metrics)
    plot_metrics(df_final, 'Rules', 'Alignment', 'Model', f'{name}/h_r.pdf', hue_order=hue_metrics)

    plot_metrics(df_final, 'Model', 'Collapse-Worst', 'Model', f'{name}/cw.pdf', dodge=False, hue_order=hue_metrics)
    plot_metrics(df_final, 'Model', 'Collapse-Avg', 'Model', f'{name}/ca.pdf', dodge=False, hue_order=hue_metrics)
    plot_metrics(df_final, 'Model', 'Adaptation', 'Model', f'{name}/adap.pdf', dodge=False, hue_order=hue_metrics)
    plot_metrics(df_final, 'Model', 'Inverse Mutual Information', 'Model', f'{name}/imi.pdf', dodge=False, hue_order=hue_metrics)
    plot_metrics(df_final, 'Model', 'Alignment', 'Model', f'{name}/h.pdf', dodge=False, hue_order=hue_metrics)

    df = df_final[df_final["Encoder Dimension"].isin([32, 128, 512])].copy()
    df["Capacity"] = df["Encoder Dimension"]
    df["Capacity"] = df["Capacity"].replace({
        32: 'Small',
        128: 'Medium',
        512: 'Large'
    })

    if 'MHA' in name:
        plot_metrics_full(df, 'Capacity', 'Collapse-Worst', 'Rules', 'Search-Version', 'Model', f'{name}/cw_full_s.pdf',
                          hue_order=hue_metrics)
        plot_metrics_full(df, 'Capacity', 'Collapse-Avg', 'Rules', 'Search-Version', 'Model', f'{name}/ca_full_s.pdf',
                          hue_order=hue_metrics)
        plot_metrics_full(df, 'Capacity', 'Adaptation', 'Rules', 'Search-Version', 'Model', f'{name}/adap_full_s.pdf',
                          hue_order=hue_metrics)
        plot_metrics_full(df, 'Capacity', 'Inverse Mutual Information', 'Rules', 'Search-Version', 'Model', f'{name}/imi_full_s.pdf',
                          hue_order=hue_metrics)
        plot_metrics_full(df, 'Capacity', 'Alignment', 'Rules', 'Search-Version', 'Model', f'{name}/h_full_s.pdf',
                          hue_order=hue_metrics)

    plot_metrics_full(df, 'Capacity', 'Collapse-Worst', 'Rules', None, 'Model', f'{name}/cw_full.pdf', hue_order=hue_metrics)
    plot_metrics_full(df, 'Capacity', 'Collapse-Avg', 'Rules', None, 'Model', f'{name}/ca_full.pdf', hue_order=hue_metrics)
    plot_metrics_full(df, 'Capacity', 'Adaptation', 'Rules', None, 'Model', f'{name}/adap_full.pdf', hue_order=hue_metrics)
    plot_metrics_full(df, 'Capacity', 'Inverse Mutual Information', 'Rules', None, 'Model', f'{name}/imi_full.pdf', hue_order=hue_metrics)
    plot_metrics_full(df, 'Capacity', 'Alignment', 'Rules', None, 'Model', f'{name}/h_full.pdf', hue_order=hue_metrics)

    if 'MHA' in name:
        plot_pie(rank[0]['perf_last'], hue_pie, colors, f'{name}/rank_1.pdf')
        plot_pie(rank[0]['perf_ood_last'], hue_pie, colors, f'{name}/rank_ood_1.pdf')
        plot_pie(rank[0]['perf_sub_last'], hue_pie_sub, [colors[1], colors[-1]], f'{name}/rank_sub_1.pdf', circle=False)
        plot_pie(rank[0]['perf_ood_sub_last'], hue_pie_sub, [colors[1], colors[-1]], f'{name}/rank_ood_sub_1.pdf',
                 circle=False)

        plot_pie(rank[1]['perf_last'], hue_pie, colors, f'{name}/rank_2.pdf')
        plot_pie(rank[1]['perf_ood_last'], hue_pie, colors, f'{name}/rank_ood_2.pdf')
        plot_pie(rank[1]['perf_sub_last'], hue_pie_sub, [colors[1], colors[-1]], f'{name}/rank_sub_2.pdf', circle=False)
        plot_pie(rank[1]['perf_ood_sub_last'], hue_pie_sub, [colors[1], colors[-1]], f'{name}/rank_ood_sub_2.pdf', circle=False)

        r = dict()
        for key in rank[0].keys():
            r[key] = rank[0][key] + rank[1][key]
        rank = r

    plot_pie(rank['perf_last'], hue_pie, colors, f'{name}/rank.pdf')
    plot_pie(rank['perf_ood_last'], hue_pie, colors, f'{name}/rank_ood.pdf')
    plot_pie(rank['perf_sub_last'], hue_pie_sub, [colors[1], colors[-1]], f'{name}/rank_sub.pdf', circle=False)
    plot_pie(rank['perf_ood_sub_last'], hue_pie_sub, [colors[1], colors[-1]], f'{name}/rank_ood_sub.pdf', circle=False)

    df = df_final[df_final[["Rules", "Encoder Dimension"]].apply(tuple, 1).isin([(2, 32), (4, 64), (8, 128), (16, 256), (32, 512)])].copy()

    plot_line(df, 'Rules', 'Perf', None, None, 'Model', f'{name}/perf_mr.pdf',
              title=None, xlabel='Rules', ylabel=ylabel, hue_order=hue_perf, marker='o')
    plot_line(df, 'Rules', 'Perf-OoD', None, None, 'Model', f'{name}/perf_ood_mr.pdf',
              title=None, xlabel='Rules', ylabel=ylabel, hue_order=hue_perf, marker='o')

# Get MLP Details

print('MLP')

mlp_versions = ['MLP-Classification-WithDecoder', 'MLP-Classification-WithoutDecoder', 'MLP-Regression-WithDecoder', 'MLP-Regression-WithoutDecoder']
names = [f'MLP/Classification/With_Decoder', f'MLP/Classification/Without_Decoder', f'MLP/Regression/With_Decoder', f'MLP/Regression/Without_Decoder']

mlp_clf_wd_final = pd.read_pickle(f'MLP/ToyTask/Classification/With_Decoder/Logistics/df_final.pt')
mlp_clf_wd_final['Perf'] = 100. - mlp_clf_wd_final['Perf']
mlp_clf_wd_final['Perf-OoD'] = 100. - mlp_clf_wd_final['Perf-OoD']
add_random(mlp_clf_wd_final, mlp_versions[0])
mlp_clf_wd_final['Mutual Information'] = (np.log(mlp_clf_wd_final['Rules'].astype(np.float)) - mlp_clf_wd_final['Mutual Information']) / np.log(mlp_clf_wd_final['Rules'].astype(np.float))
mlp_clf_wd_training = pd.read_pickle(f'MLP/ToyTask/Classification/With_Decoder/Logistics/df_training.pt')
mlp_clf_wd_training['Perf'] = 100. - mlp_clf_wd_training['Perf']

# for model in ['Modular', 'GT-Modular', 'Modular-op', 'Monolithic']:
#     for rule in [2, 4, 8, 16, 32]:
#         for enc in [32, 64, 128, 256, 512]:
#             for ds in range(5):
#                 for seed in range(5):
#                     x = mlp_clf_wd_training.loc[(mlp_clf_wd_training['Mode'] == 'MLP-Classification-WithDecoder') &
#                                                 (mlp_clf_wd_training['Model'] == model) &
#                                                 (mlp_clf_wd_training['Rule'] == rule) &
#                                                 (mlp_clf_wd_training['Encoder Dimension'] == enc) &
#                                                 (mlp_clf_wd_training['Dimension'] == 4 * enc) &
#                                                 (mlp_clf_wd_training['Data Seed'] == ds) &
#                                                 (mlp_clf_wd_training['Seed'] == seed) &
#                                                 (mlp_clf_wd_training['Iteration'] == 50000.0)].iloc[0]
#                     params = x['Number of Parameters']
#                     perf = x['Perf']
#
#                     temp = dict()
#                     temp['Mode'] = ['MLP-Classification-WithDecoder'] * 80
#                     temp['Model'] = [model] * 80
#                     temp['Rule'] = [rule] * 80
#                     temp['Encoder Dimension'] = [enc] * 80
#                     temp['Dimension'] = [4*enc] * 80
#                     temp['Data Seed'] = [ds] * 80
#                     temp['Seed'] = [seed] * 80
#                     temp['Iteration'] = np.array(list(range(105000, 505000, 5000))).astype(np.float)
#                     temp['Number of Parameters'] = [params] * 80
#                     temp['Perf'] = [perf] * 80
#
#                     mlp_clf_wd_training = pd.concat([mlp_clf_wd_training, pd.DataFrame.from_dict(temp)], ignore_index=True)

with open(f'MLP/ToyTask/Classification/With_Decoder/Logistics/ranking.pickle', 'rb') as handle:
    mlp_clf_wd_rank = pickle.load(handle)

# mlp_clf_wo_final = pd.read_pickle(f'MLP/ToyTask/Classification/Without_Decoder/Logistics/df_final.pt')
# mlp_clf_wo_final['Perf'] = 100. - mlp_clf_wo_final['Perf']
# mlp_clf_wo_final['Perf-OoD'] = 100. - mlp_clf_wo_final['Perf-OoD']
# add_random(mlp_clf_wo_final, mlp_versions[1])
# mlp_clf_wo_final['Mutual Information'] = (np.log(mlp_clf_wo_final['Rules'].astype(np.float)) - mlp_clf_wo_final['Mutual Information']) / np.log(mlp_clf_wo_final['Rules'].astype(np.float))
# mlp_clf_wo_training = pd.read_pickle(f'MLP/ToyTask/Classification/Without_Decoder/Logistics/df_training.pt')
# mlp_clf_wo_training['Perf'] = 100. - mlp_clf_wo_training['Perf']
# with open(f'MLP/ToyTask/Classification/Without_Decoder/Logistics/ranking.pickle', 'rb') as handle:
#     mlp_clf_wo_rank = pickle.load(handle)

mlp_reg_wd_final = pd.read_pickle(f'MLP/ToyTask/Regression/With_Decoder/Logistics/df_final.pt')
add_random(mlp_reg_wd_final, mlp_versions[2])
mlp_reg_wd_final['Mutual Information'] = (np.log(mlp_reg_wd_final['Rules'].astype(np.float)) - mlp_reg_wd_final['Mutual Information']) / np.log(mlp_reg_wd_final['Rules'].astype(np.float))
mlp_reg_wd_training = pd.read_pickle(f'MLP/ToyTask/Regression/With_Decoder/Logistics/df_training.pt')

# for model in ['Modular', 'GT-Modular', 'Modular-op', 'Monolithic']:
#     for rule in [2, 4, 8, 16, 32]:
#         for enc in [32, 64, 128, 256, 512]:
#             for ds in range(5):
#                 for seed in range(5):
#                     x = mlp_reg_wd_training.loc[(mlp_reg_wd_training['Mode'] == 'MLP-Regression-WithDecoder') &
#                                                 (mlp_reg_wd_training['Model'] == model) &
#                                                 (mlp_reg_wd_training['Rule'] == rule) &
#                                                 (mlp_reg_wd_training['Encoder Dimension'] == enc) &
#                                                 (mlp_reg_wd_training['Dimension'] == 4 * enc) &
#                                                 (mlp_reg_wd_training['Data Seed'] == ds) &
#                                                 (mlp_reg_wd_training['Seed'] == seed) &
#                                                 (mlp_reg_wd_training['Iteration'] == 50000.0)].iloc[0]
#                     params = x['Number of Parameters']
#                     perf = x['Perf']
#
#                     temp = dict()
#                     temp['Mode'] = ['MLP-Regression-WithDecoder'] * 80
#                     temp['Model'] = [model] * 80
#                     temp['Rule'] = [rule] * 80
#                     temp['Encoder Dimension'] = [enc] * 80
#                     temp['Dimension'] = [4*enc] * 80
#                     temp['Data Seed'] = [ds] * 80
#                     temp['Seed'] = [seed] * 80
#                     temp['Iteration'] = np.array(list(range(105000, 505000, 5000))).astype(np.float)
#                     temp['Number of Parameters'] = [params] * 80
#                     temp['Perf'] = [perf] * 80
#
#                     mlp_reg_wd_training = pd.concat([mlp_reg_wd_training, pd.DataFrame.from_dict(temp)], ignore_index=True)

with open(f'MLP/ToyTask/Regression/With_Decoder/Logistics/ranking.pickle', 'rb') as handle:
    mlp_reg_wd_rank = pickle.load(handle)

# mlp_reg_wo_final = pd.read_pickle(f'MLP/ToyTask/Regression/Without_Decoder/Logistics/df_final.pt')
# add_random(mlp_reg_wo_final, mlp_versions[3])
# mlp_reg_wo_final['Mutual Information'] = (np.log(mlp_reg_wo_final['Rules'].astype(np.float)) - mlp_reg_wo_final['Mutual Information']) / np.log(mlp_reg_wo_final['Rules'].astype(np.float))
# mlp_reg_wo_training = pd.read_pickle(f'MLP/ToyTask/Regression/Without_Decoder/Logistics/df_training.pt')
# with open(f'MLP/ToyTask/Regression/Without_Decoder/Logistics/ranking.pickle', 'rb') as handle:
#     mlp_reg_wo_rank = pickle.load(handle)

# mlp_classification_final = pd.concat([mlp_clf_wd_final, mlp_clf_wo_final], ignore_index=True)
# mlp_regression_final = pd.concat([mlp_reg_wd_final, mlp_reg_wo_final], ignore_index=True)
# mlp_final = pd.concat([mlp_clf_wd_final, mlp_clf_wo_final, mlp_reg_wd_final, mlp_reg_wo_final], ignore_index=True)
mlp_final = pd.concat([mlp_clf_wd_final, mlp_reg_wd_final], ignore_index=True)

# mlp_rank_classification = dict()
# for key in mlp_clf_wo_rank.keys():
#     mlp_rank_classification[key] = mlp_clf_wo_rank[key] + mlp_clf_wd_rank[key]
# mlp_rank_regression = dict()
# for key in mlp_clf_wo_rank.keys():
#     mlp_rank_regression[key] = mlp_reg_wo_rank[key] + mlp_reg_wd_rank[key]
mlp_rank = dict()
for key in mlp_clf_wd_rank.keys():
    mlp_rank[key] = mlp_clf_wd_rank[key] + mlp_reg_wd_rank[key]
    # mlp_rank[key] = mlp_clf_wo_rank[key] + mlp_clf_wd_rank[key] + \
    #                 mlp_reg_wo_rank[key] + mlp_reg_wd_rank[key]

# mlp_classification_training = pd.concat([mlp_clf_wd_training, mlp_clf_wo_training], ignore_index=True)
# mlp_regression_training = pd.concat([mlp_reg_wd_training, mlp_reg_wo_training], ignore_index=True)
# mlp_training = pd.concat([mlp_clf_wd_training, mlp_clf_wo_training, mlp_reg_wd_training, mlp_reg_wo_training], ignore_index=True)
mlp_training = pd.concat([mlp_clf_wd_training, mlp_reg_wd_training], ignore_index=True)
plot_full(mlp_clf_wd_training, mlp_clf_wd_final, mlp_clf_wd_rank, f'Plots/MLP/Classification/With_Decoder/')
# plot_full(mlp_clf_wo_training, mlp_clf_wo_final, mlp_clf_wo_rank, f'Plots/MLP/Classification/Without_Decoder/')
plot_full(mlp_reg_wd_training, mlp_reg_wd_final, mlp_reg_wd_rank, f'Plots/MLP/Regression/With_Decoder/')
# plot_full(mlp_reg_wo_training, mlp_reg_wo_final, mlp_reg_wo_rank, f'Plots/MLP/Regression/Without_Decoder/')

# plot_full(mlp_classification_training, mlp_classification_final, mlp_rank_classification, f'Plots/MLP/Classification/')
# plot_full(mlp_regression_training, mlp_regression_final, mlp_rank_regression, f'Plots/MLP/Regression/')

plot_full(mlp_training, mlp_final, mlp_rank, f'Plots/MLP/')

# Get Transformer Details

print('MHA')

tsf_versions = ['MHA-Classification', 'MHA-Regression']
names = [f'MHA/Classification', f'MHA/Regression']
lens = [3, 5, 10, 20, 30]

mha_clf_final = pd.read_pickle(f'Transformer/ToyTask/Classification/Nonlinear/Logistics/df_final.pt')
for l in lens:
    mha_clf_final[f'Perf-{l}'] = 100. - mha_clf_final[f'Perf-{l}']
    mha_clf_final[f'Perf-OoD-{l}'] = 100. - mha_clf_final[f'Perf-OoD-{l}']
add_random(mha_clf_final, tsf_versions[0], arch='MHA')
mha_clf_final['Mutual Information'] = (np.log(mha_clf_final['Rules'].astype(np.float)) - mha_clf_final['Mutual Information']) / np.log(mha_clf_final['Rules'].astype(np.float))
mha_clf_final['Perf'] = mha_clf_final['Perf-10']
mha_clf_final['Perf-OoD'] = mha_clf_final['Perf-OoD-30']
mha_clf_training = pd.read_pickle(f'Transformer/ToyTask/Classification/Nonlinear/Logistics/df_training.pt')
mha_clf_training['Perf'] = 100. - mha_clf_training['Perf']
with open(f'Transformer/ToyTask/Classification/Nonlinear/Logistics/ranking_1.pickle', 'rb') as handle:
    mha_clf_rank_1 = pickle.load(handle)
with open(f'Transformer/ToyTask/Classification/Nonlinear/Logistics/ranking_2.pickle', 'rb') as handle:
    mha_clf_rank_2 = pickle.load(handle)

mha_reg_final = pd.read_pickle(f'Transformer/ToyTask/Regression/Nonlinear/Logistics/df_final.pt')
add_random(mha_reg_final, tsf_versions[0], arch='MHA')
mha_reg_final['Mutual Information'] = (np.log(mha_reg_final['Rules'].astype(np.float)) - mha_reg_final['Mutual Information']) / np.log(mha_reg_final['Rules'].astype(np.float))
mha_reg_final['Perf'] = mha_reg_final['Perf-10']
mha_reg_final['Perf-OoD'] = mha_reg_final['Perf-OoD-30']
mha_reg_training = pd.read_pickle(f'Transformer/ToyTask/Regression/Nonlinear/Logistics/df_training.pt')
with open(f'Transformer/ToyTask/Regression/Nonlinear/Logistics/ranking_1.pickle', 'rb') as handle:
    mha_reg_rank_1 = pickle.load(handle)
with open(f'Transformer/ToyTask/Regression/Nonlinear/Logistics/ranking_2.pickle', 'rb') as handle:
    mha_reg_rank_2 = pickle.load(handle)

mha_final = pd.concat([mha_clf_final, mha_reg_final], ignore_index=True)
mha_training = pd.concat([mha_clf_training, mha_reg_training], ignore_index=True)
mha_rank_1 = dict()
mha_rank_2 = dict()
for key in mha_clf_rank_1.keys():
    mha_rank_1[key] = mha_clf_rank_1[key] + mha_reg_rank_1[key]
    mha_rank_2[key] = mha_clf_rank_2[key] + mha_reg_rank_2[key]

plot_full(mha_clf_training, mha_clf_final, (mha_clf_rank_1, mha_clf_rank_2), f'Plots/MHA/Classification/')
plot_full(mha_reg_training, mha_reg_final, (mha_reg_rank_1, mha_reg_rank_2), f'Plots/MHA/Regression/')
plot_full(mha_training, mha_final, (mha_rank_1, mha_rank_2), f'Plots/MHA/')

# Get RNN Details

print('RNN')

rnn_versions = ['RNN-Classification', 'RNN-Regression']
names = [f'RNN/SCOFF/Classification', f'RNN/SCOFF/Regression']
lens = [3, 5, 10, 20, 30]

rnn_clf_final = pd.read_pickle(f'RNN/SCOFF/Classification/Logistics/df_final.pt')
for l in lens:
    rnn_clf_final[f'Perf-{l}'] = 100. - rnn_clf_final[f'Perf-{l}']
    rnn_clf_final[f'Perf-OoD-{l}'] = 100. - rnn_clf_final[f'Perf-OoD-{l}']
add_random(rnn_clf_final, rnn_versions[0], arch='RNN')
rnn_clf_final['Mutual Information'] = (np.log(rnn_clf_final['Rules'].astype(np.float)) - rnn_clf_final['Mutual Information']) / np.log(rnn_clf_final['Rules'].astype(np.float))
rnn_clf_final['Perf'] = rnn_clf_final['Perf-10']
rnn_clf_final['Perf-OoD'] = rnn_clf_final['Perf-OoD-30']
rnn_clf_training = pd.read_pickle(f'RNN/SCOFF/Classification/Logistics/df_training.pt')
rnn_clf_training['Perf'] = 100. - rnn_clf_training['Perf']
with open(f'RNN/SCOFF/Classification/Logistics/ranking.pickle', 'rb') as handle:
    rnn_clf_rank = pickle.load(handle)

rnn_reg_final = pd.read_pickle(f'RNN/SCOFF/Regression/Logistics/df_final.pt')
add_random(rnn_reg_final, rnn_versions[0], arch='RNN')
rnn_reg_final['Mutual Information'] = (np.log(rnn_reg_final['Rules'].astype(np.float)) - rnn_reg_final['Mutual Information']) / np.log(rnn_reg_final['Rules'].astype(np.float))
rnn_reg_final['Perf'] = rnn_reg_final['Perf-10']
rnn_reg_final['Perf-OoD'] = rnn_reg_final['Perf-OoD-30']
rnn_reg_training = pd.read_pickle(f'RNN/SCOFF/Regression/Logistics/df_training.pt')
with open(f'RNN/SCOFF/Regression/Logistics/ranking.pickle', 'rb') as handle:
    rnn_reg_rank = pickle.load(handle)

rnn_final = pd.concat([rnn_clf_final, rnn_reg_final], ignore_index=True)
rnn_training = pd.concat([rnn_clf_training, rnn_reg_training], ignore_index=True)
rnn_rank = dict()
for key in rnn_clf_rank.keys():
    rnn_rank[key] = rnn_clf_rank[key] + rnn_reg_rank[key]

plot_full(rnn_clf_training, rnn_clf_final, rnn_clf_rank, f'Plots/RNN/Classification')
plot_full(rnn_reg_training, rnn_reg_final, rnn_reg_rank, f'Plots/RNN/Regression')
plot_full(rnn_training, rnn_final, rnn_rank, f'Plots/RNN')

# Get Consolidated Details

print('Consolidated')

rank = dict()
for key in mha_rank_1.keys():
    rank[key] = mha_rank_1[key] + mha_rank_2[key] + mlp_rank[key] + rnn_rank[key]

hue_pie = ['GT-Modular', 'Modular', 'Modular-op', 'Monolithic']
hue_pie_sub = ['Modular', 'Monolithic']

df_final = pd.concat([mlp_final, mha_final, rnn_final], ignore_index=True)
df_training = pd.concat([mlp_training, mha_training, rnn_training], ignore_index=True)

df_training.to_pickle('training.pkl')

plot_full(df_training, df_final, rank, f'Plots/')

plot_pie(rank['perf_last'], hue_pie, colors, f'Plots/rank.pdf')
plot_pie(rank['perf_ood_last'], hue_pie, colors, f'Plots/rank_ood.pdf')
plot_pie(rank['perf_sub_last'], hue_pie_sub, [colors[1], colors[-1]], f'Plots/rank_sub.pdf', circle=False)
plot_pie(rank['perf_ood_sub_last'], hue_pie_sub, [colors[1], colors[-1]], f'Plots/rank_ood_sub.pdf', circle=False)
