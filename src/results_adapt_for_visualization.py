import pandas as pd
import numpy as np
from NegotiationConfig import *
import seaborn as sb

# This contains data for each step performed
steps_data = pd.read_csv('results/run_2019-09-12 v-unreliable.csv', index_col=0)
steps_data['consensus_progress'] = 1. - steps_data['consensus_current']/steps_data['consensus_start']
#data = data[data['step']<=MAX_STEPS]


MAX_STEP_DF = steps_data['step'].max()
step_expanded = pd.DataFrame()
for g, group in steps_data.groupby(['sample', 'method']):
    last_step = group['step'].max()
    last_step_row = group.loc[group['step']==last_step]
    print('\r Processing sample {} of {}'.format(group['sample'].max(), steps_data['sample'].max()), end='')
    for step in range(last_step, MAX_STEP_DF+1):
        last_step_row.loc[:,'step'] = step
        last_step_row.loc[:,'status'] = 'consensus'
        group = group.append(last_step_row, ignore_index=True, sort=False)
    step_expanded = step_expanded.append(group, ignore_index=True, sort=False)
    step_expanded.to_csv('results/results_visualization.csv')