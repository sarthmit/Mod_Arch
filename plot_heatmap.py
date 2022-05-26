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

# Set Plotting Variables

sns.color_palette("dark", as_cmap=True)
sns.set(style="darkgrid")#, font_scale=1.1)
font = {'family' : 'Open Sans',
        'size'   : 32}

mpl.rc('font', **font)
size=16

activations = [
    [0.95, 0., 0., 0.05],
    [0.0, 0., 0., 1.0],
    [0.35, 0.1, 0., 0.55],
    [0., 1., 0., 0.]
]
activations = np.array(activations)
ax = sns.heatmap(activations, cmap="YlGnBu", linewidths=2.,
                 xticklabels=range(1, 5), yticklabels=range(1, 5),
                 vmin=0., vmax=1.
                 )
ax.set_xlabel('Module', fontsize=size)
ax.set_ylabel('Rule', fontsize=size)
ax.set_title('Ground Truth Rules: 4 | Modules: 4', fontsize=size)
plt.savefig('Main_Plots/collapse_eg.pdf', bbox_inches='tight')
plt.close()
