

import numpy as np

ID_TO_LABEL = {16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse'}
LABEL_TO_ID = {'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19}
CHANNEL_ORDER = [0, 16, 17, 18, 19] # Order of channels in output segmentation and corresponding dataset labels
CHANNEL_NAMES = [ID_TO_LABEL[i] if i!=0 else 'other' for i in CHANNEL_ORDER]
ALL_LABELS = [ID_TO_LABEL[i]  for i in CHANNEL_ORDER if i!=0]
AGENT_NAMES = ['ag_{}'.format(l) for l in ALL_LABELS]
ds_csv_paths = {dset: {label: 'datasets/coco_animals_{}_{}.csv'.format(dset, label) for label in ALL_LABELS} for dset in ['train', 'validation', 'test']}