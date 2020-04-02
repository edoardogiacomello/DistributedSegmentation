import Negotiation
import SegmentationModel as sm
import NegotiationTools as nt
import numpy as np
import os
import pandas

ALL_LABELS = ['bird', 'cat', 'dog', 'horse']
model_path = ['models/{}/model_ep{}.h5'.format(label,1) for label in ALL_LABELS]

models = [sm.SegmentationModel(label) for label in ALL_LABELS]
# Load the checkpoints
for m in models:
    m.load_finetuned_network(epoch=1)

DATASET = 'datasets/coco_animals_test_balanced.csv'
current_path = 'predictions/bcdh/'
for sample_id, (png_path, seg_path, input_sample, ground_truth) in enumerate(sm.SegmentationModel('dummy').load_test_dataset(DATASET, batch_size=1)):
    pred = list()
    for modlabel, mod in zip(ALL_LABELS, models):
        os.makedirs(current_path, exist_ok=True)
        pred.append(mod.predict(input_sample).numpy())
    pred = np.concatenate(pred)
    filename = current_path + png_path.numpy()[0].decode('utf-8').split('/')[-1].replace('.png', '')
    np.save(filename + '.npy', pred)
    print("\rSaved {} predictions".format(sample_id), end='')