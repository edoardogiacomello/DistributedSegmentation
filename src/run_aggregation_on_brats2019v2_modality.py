import pandas as pd
import os
import glob
import numpy as np
import Experiments as exp
from itertools import groupby
import seaborn as sns

DB_CSV_PATH = './predictions/database_singlemodel_singlelabel_brats2019v2.csv'
DATABASE = pd.read_csv(DB_CSV_PATH, index_col=0)
OUTPUT_FOLDER = './aggregations_brats2019v2/'
DATABASE = DATABASE.sort_values(by=['dataset', 'modality', 'output_label', 'patient', 'slice'])
LABELS = ['NCR/NET', 'ED', 'ET']
MODALITIES = [ 't1', 't1ce', 't2', 'flair']

AGGREGATION_RESULTS = pd.DataFrame() 
for (test_or_val, output_label), current_dataset in DATABASE.groupby(['dataset', 'output_label']):
    temp = current_dataset.drop(columns=['dataset', 'output_label', 'model_path'])
    temp = temp.set_index(['patient', 'gt_or_pred', 'modality'])
    # Aggregations expexts Preds: [Agents, H, W, Aggregation_Labels], GT: [H, W, Aggregation_Labels]
    for patient, patient_samples in temp.groupby(['patient']):
        # Loading slices
        
        GT = list()
        PREDS = list()
        for modality in MODALITIES:
            if len(GT) == 0:
                # We take just the GT for the first modality as they should be the same for each of them
                GT += [np.load(path) for path in patient_samples.loc[patient, 'GT', modality]['slice_path']]
                
            PREDS.append([np.load(path) for path in patient_samples.loc[patient, 'predictions', modality]['slice_path']])
        
        GT = np.array(GT) # Now GT is [Slices, H, W, 1]
        PREDS = np.stack(PREDS, axis=1) # Now PREDS is [Slices, Modality, H, W, 1]
        
        # Expanding aggregation labels: Converting each image from binary to [p("negative"), p("positive")]. Making a list of slices for each image.
        GT = np.concatenate([1.0-GT, GT], axis=-1)
        PREDS = np.concatenate([1.0-PREDS, PREDS], axis=-1)
        
        results, outputs = exp.run_experiment_on_list(proposals_list=PREDS, gt_list=GT, return_mean=False, agent_names=MODALITIES, label_names=['Negative', 'Positive'], return_outputs=True, skip_negotiation=True)
        results.rename(columns = {'sample_id':'slice'}, inplace = True)
        results['dataset'] = test_or_val
        results['output_label'] = output_label
        results['patient'] = patient
        AGGREGATION_RESULTS = AGGREGATION_RESULTS.append(results, ignore_index=True)
        
        # Saving the aggregations for later inspection
        
        for slice_id, slice_aggregation in enumerate(outputs):
            if len(slice_aggregation.keys()) != 0:
                #print(slice_id)
                for aggregation_type, aggr_result in slice_aggregation.items():
                    out_folder = os.path.join(OUTPUT_FOLDER, test_or_val, output_label, aggregation_type)
                    filename = os.path.join(out_folder, '{}_{}.npy'.format(patient, slice_id))
                    os.makedirs(out_folder, exist_ok=True)
                    try:
                        np.save(filename, aggr_result)
                    except:
                        print("Error in saving result. Maybe your disk is full?")
            print(f"\rProcessed patient {patient} slice {slice_id}")
        AGGREGATION_RESULTS.to_csv(OUTPUT_FOLDER+'aggregation_results_partial.csv')
AGGREGATION_RESULTS.to_csv(OUTPUT_FOLDER+'aggregation_results.csv')
print("Done")
    
    