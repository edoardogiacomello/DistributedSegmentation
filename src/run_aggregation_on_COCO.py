"""

RUN AGGREGATION METHODS ON PREVIOUSLY SAVED COCO PREDICTIONS AND STORE THE RESULTS
Please run "generate_coco_predictions.py" before running this script. Predictions should be stored in ./predictions/bcdh/<img_id>.npy 

"""

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

db = pd.read_csv('./datasets/coco_animals_test_balanced.csv')

RESULT_PATH = 'results/bcdh/'
os.makedirs(RESULT_PATH, exist_ok=True)
os.makedirs(RESULT_PATH+'csv/', exist_ok=True)

FILE_LIST = sorted(glob.glob('predictions/bcdh/*.npy'))

### COMMENT THIS SECTION FOR DISABLING RESUMING
print("Checking previous results status... (you can delete {} to restart from scratch)...".format(RESULT_PATH))
existing_files = [sorted(glob.glob(subfolder + '*')) for subfolder in sorted(glob.glob(RESULT_PATH + '*/'))]
file_found = {file_to_search: [any([os.path.basename(file_to_search).replace('.npy', '') in file for file in subfolder]) for subfolder in existing_files] for file_to_search in FILE_LIST}

done_samples = list()
partials = list()
todo = list()

for fname, record in file_found.items():
    if all(record):
        done_samples.append(fname)
        continue
    if any(record):
        partials.append(fname)
        continue
    else:
        todo.append(fname)

if len(todo) != len(FILE_LIST):
    choice = ''
    while choice.lower() not in ['restart', 'continue']:
        choice = input("There are {} completed, {} undone and {} partially done samples. \n Do you want to RESTART or CONTINUE?".format(len(done_samples), len(partials), len(todo)))
        if choice.lower() == 'continue':
            FILE_LIST = partials + todo
    

for i, pred_path in enumerate(FILE_LIST):
    try:
        png_basename = os.path.basename(pred_path).replace('.npy', '.png')
        seg_path = db.loc[db.png.str.contains(png_basename)]['seg'].iloc[0]
        raw_seg = imageio.imread(seg_path)[...,np.newaxis]
        seg = tf.image.resize_with_pad(raw_seg, 224, 224, method='nearest').numpy()
        seg = np.stack([np.equal(seg, lid).astype(np.float32) for lid in CHANNEL_ORDER], axis=-1)
        pred = np.load(pred_path)

        results, outputs = exp.run_experiment_on_list(proposals_list=[pred], gt_list=[seg], return_mean=False, agent_names =AGENT_NAMES, label_names=CHANNEL_NAMES, return_outputs=True)
        results['pred_path'] = pred_path
        results['gt_path'] = seg_path
        results.to_csv(RESULT_PATH+'csv/'+ png_basename.replace('.png', '.csv'))

        for method_name, outp in outputs[0].items():
            os.makedirs(RESULT_PATH+method_name+'/', exist_ok=True)
            if isinstance(outp, dict):
                for agr_prop in outp.keys():
                    np.save(RESULT_PATH+method_name+'/' + os.path.basename(pred_path).replace('.npy', '') + '_' + agr_prop + '.npy', outp[agr_prop])
            else:
                np.save(RESULT_PATH+method_name+'/' + os.path.basename(pred_path), outp.astype(np.float32))
    except Exception as e:
        print("ERROR processing file {}".format(pred_path))
        print(e)
        
    print("Done sample {} ({} of {})".format(png_basename, i, len(FILE_LIST)))