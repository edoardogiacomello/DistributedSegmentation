import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import NegotiationTools as negtools
from NegotiationConfigNeuralNet import *
import os
import glob
import imageio
import Experiments as exp
import tensorflow as tf
from itertools import groupby

PRED_PATH = 'predictions/brats2015_validation/'
RESULT_PATH = 'results/brats2015_validation/'
os.makedirs(RESULT_PATH, exist_ok=True)
os.makedirs(RESULT_PATH+'csv/', exist_ok=True)
GT_LIST = sorted(glob.glob(PRED_PATH+'GT/*')) # Here the files are forted by SAMPLE but not by SLICE

AGENTS = [
 'Segan_IO_TF2_brats_on_T1',
 'Segan_IO_TF2_brats_on_T1c',
 'Segan_IO_TF2_brats_on_T2', 
 'Segan_IO_TF2_brats_on_FLAIR',
 'Transfer_Brats_Flair_to_T1',
 'Transfer_Brats_Flair_to_T1_freeze_all',
 'Transfer_Brats_Flair_to_T1c',
 'Transfer_Brats_Flair_to_T1c_freeze_all',
 'Transfer_Brats_Flair_to_T2',
 'Transfer_Brats_Flair_to_T2_freeze_all'
]

for sample_id, gt_paths_group in groupby(GT_LIST, key=lambda x: int(os.path.basename(x).replace('.npy', '').split('_')[0])):
    gt_slice_paths = sorted(list(gt_paths_group), key=lambda x: int(os.path.basename(x).replace('.npy', '').split('_')[-1]))

    prediction_slices = list()
    gt_slices = list()

    for gt_sp in gt_slice_paths:
        prediction = list()
        for agent in AGENTS:
            pred_binary = np.load(gt_sp.replace('/GT/', '/{}/'.format(agent))) # Binary representation
            pred_softmax = np.concatenate([1-pred_binary, pred_binary], axis=-1) # One_hot (Not tumor, Tumor) representation
            prediction.append(pred_softmax)
        prediction = np.stack(prediction)
        prediction_slices.append(prediction)
        gt_binary = np.load(gt_sp)
        gt_slices.append(np.concatenate([1-gt_binary, gt_binary], axis=-1))

    results, outputs = exp.run_experiment_on_list(proposals_list=prediction_slices, gt_list=gt_slices, return_mean=False, agent_names=AGENTS, label_names=['Negative', 'Positive'], return_outputs=True, MAX_STEPS=3000)
    results['gt_id'] = sample_id

    results.to_csv(RESULT_PATH+'csv/'+ str(sample_id) + '.csv')
    for output in outputs:
        for method_name, outp in output.items():
            os.makedirs(RESULT_PATH+method_name+'/', exist_ok=True)
            if isinstance(outp, dict):
                for agr_prop in outp.keys():
                    np.save(RESULT_PATH+method_name+'/' + str(sample_id) + '_' + agr_prop + '.npy', outp[agr_prop])
            else:
                np.save(RESULT_PATH+method_name+'/' + str(sample_id)+'.npy', outp.astype(np.float32))
        
    print("Done sample {}".format(str(sample_id)))
    