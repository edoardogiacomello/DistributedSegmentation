from SegmentationModel import SegmentationModel
import tensorflow as tf
import tensorflow.keras as tk
import skimage as sk
import numpy as np
from NegotiationTools import StatisticsLogger
from DCSegUtils import *
import numpy as np
import pandas as pd

for MODEL_EPOCH in [1, 10, 20, 275]:

    tools = StatisticsLogger()
    model_path = ['models/{}/model_ep{}.h5'.format(label,MODEL_EPOCH) for label in LABEL_TO_ID.keys()]

    models = [SegmentationModel(label) for label in ALL_LABELS]
    # Load the checkpoints
    for m in models:
        m.load_finetuned_network(epoch=MODEL_EPOCH)

    # Producing validation scores
    results = pd.DataFrame()
    for m in models:
        print("Processing model trained on {} ".format(m.target_label))
        for i, (x, ground_truth) in enumerate(m.load_dataset(m.ds_csv_paths['validation'][m.target_label], batch_size=1, size=m.size)):
            row = {'model': m.target_label, 'sample_n': i}
            ground_truth = ground_truth.numpy()
            pred = m.predict(x).numpy()
            row['mae'] = np.mean(np.abs(pred - ground_truth))
            pred_binary = np.equal(pred, pred.max(axis=-1)[...,np.newaxis])
            stats = tools.compute_statistics(ground_truth=ground_truth[0], predictions=pred_binary[0], prefix='baseline_', label_names=m.CHANNEL_NAMES)
            row.update(stats)
            results = results.append(row, ignore_index=True)

    results.to_csv('results/bcdh_baseline_ep{}_validation_performances_per_sample.csv'.format(MODEL_EPOCH), index_label=False, index=False)

    # Producing test scores
    results = pd.DataFrame()
    test_set_loader = dummy_model = SegmentationModel('loader')
    for m in models:
        print("Processing model trained on {} ".format(m.target_label))
        for i, (png_path, seg_path, x, ground_truth) in enumerate(test_set_loader.load_test_dataset('datasets/coco_animals_test_balanced.csv')):
            row = {'model': m.target_label, 'sample_n': i}
            ground_truth = ground_truth.numpy()
            pred = m.predict(x).numpy()
            row['mae'] = np.mean(np.abs(pred - ground_truth))
            pred_binary = np.equal(pred, pred.max(axis=-1)[...,np.newaxis])
            stats = tools.compute_statistics(ground_truth=ground_truth[0], predictions=pred_binary[0], prefix='baseline_', label_names=m.CHANNEL_NAMES)
            row.update(stats)
            results = results.append(row, ignore_index=True)
    results.to_csv('results/bcdh_baseline_ep{}_test_performances_per_sample.csv'.format(MODEL_EPOCH), index_label=False, index=False)
    # scores = pd.DataFrame()
    # for g, group in results.groupby(['model']):
    #     row = {'model_label':g, 'mae':group['mae'].mean(), 'dice_score':group['dice_score'].mean()}
    #     scores = scores.append(row, ignore_index=True)

    # z_scores = scores[scores.select_dtypes(include=[np.number]).columns].apply(zscore)
    # performances = scores.join(z_scores, rsuffix='_z')
    # performances.to_csv('results/model_ep{}_test_performances.csv'.format(MODEL_EPOCH), index_label=False, index=False)

    # Processing Performances on other datasets
    final_performances = pd.DataFrame()
    for g, gr in results.groupby('model'):
        # For each model, we select only the samples that has each label and average their performances
        for label in CHANNEL_NAMES:
            filtered = gr.loc[gr['baseline_{}_support'.format(label)]>0]['baseline_{}_f1-score'.format(label)].to_frame()
            filtered['model'] = g
            filtered['target'] = label
            final_performances = final_performances.append(filtered, ignore_index=True)
    final_performances = final_performances.groupby(['model', 'target']).mean()

    final_performances.to_csv('results/bcdh_baseline_ep{}_test_performances_model_vs_class.csv'.format(MODEL_EPOCH))