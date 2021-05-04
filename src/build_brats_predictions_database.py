import pandas as pd
import os
import glob
import numpy as np
import Experiments
from itertools import groupby

PREDICTION_FOLDER = '../../DeepMRI/src/predictions/'
DATASETS = {'testing': 'brats2019v2_testing_crop_mri',
            'validation': 'brats2019v2_validation_crop_mri'}
REGISTRY = pd.read_csv(os.path.join(PREDICTION_FOLDER, 'scores.csv'), index_col=0) # Contains the scores for the single models and the model names
# Appending the model paths to the registry
buildpath = lambda row, dataset_map=DATASETS, base_folder=PREDICTION_FOLDER: os.path.join(base_folder, dataset_map[row['dataset']], row['model'])
REGISTRY['path'] = REGISTRY.apply(buildpath, axis=1)

FILES_REGISTRY = pd.DataFrame()
extract_patient_key = lambda x: os.path.basename(x).replace('.npy', '').split('_')[0]
extract_slice_id = lambda x: int(os.path.basename(x).replace('.npy', '').split('_')[1])
extract_gt_or_pred = lambda x: os.path.dirname(x).split('/')[-1]

counter = 0
for _, row in REGISTRY.iterrows():
    file_paths = glob.glob(os.path.join(row['path'], '*', '*.npy'))
    print("Inspecting: {}".format(row.path))
    for slice_file in file_paths:
        FILES_REGISTRY = FILES_REGISTRY.append({'dataset': row.dataset,
                                                'modality':row.modality,
                                                'output_label':row.output_label,
                                                'patient':extract_patient_key(slice_file),
                                                'slice':extract_slice_id(slice_file),
                                                'gt_or_pred': extract_gt_or_pred(slice_file),
                                                'model_path':row.path,
                                                'slice_path':slice_file
                                               }, ignore_index=True)
        counter +=1
        print("\rFound {} files".format(counter), end="")
FILES_REGISTRY.to_csv(PREDICTION_FOLDER+'database.csv')
print("Done")