import os, urllib
from PIL import Image
from io import BytesIO
import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt
from matplotlib import gridspec
from ipywidgets import FloatSlider, interact, fixed, HBox, VBox, Label, Button, Output, IntProgress, FloatProgress
import pandas as pd
import IPython
from skimage.measure import label

from DCSegUtils import *
from SegmentationModel import SegmentationModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from NegotiationTools import NegotiationTools

import seaborn as sb
import pandas as pd
import datetime
from multiprocessing import Process, Queue

MODEL_OUT_FOLDER = 'models/'
ID_TO_LABEL = {16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse'}
LABEL_TO_ID = {'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19}
CHANNEL_ORDER = [0, 16, 17, 18, 19] # Order of channels in output segmentation and corresponding dataset labels
CHANNEL_NAMES = [ID_TO_LABEL[i] if i!=0 else 'other' for i in CHANNEL_ORDER]
ALL_LABELS = [ID_TO_LABEL[i]  for i in CHANNEL_ORDER if i!=0]
ds_csv_paths = {dset: {label: 'datasets/coco_animals_{}_{}.csv'.format(dset, label) for label in ALL_LABELS} for dset in ['train', 'validation', 'test']}

class Agent():
    def __init__(self, agentname, model, alpha_fun):
        self.agentname=agentname
        self.model=model
        self.task = None
        self.initial_proposal = None
        self.alpha = alpha_fun
        
    def new_task(self, image):
        self.task = image
        self.initial_proposal = self.model.predict(image).numpy()[0]
        #logits=logits[:self.task.size[1], :self.task.size[0], ...] # Otherwise logits are a square matrix [when using DeepLab]
        #self.initial_proposal = softmax(logits, axis=-1)
        self.last_proposal = self.initial_proposal
        return self.initial_proposal
    
    def utility(self, proposal):
        'Returns a utility of shape (labels) between a proposal and self.optimal'
        return np.array([np.linalg.norm(self.optimal[...,l]-proposal[...,l]) for l in range(self.optimal.shape[-1])])
    
    def propose(self, agreement):
        self.last_agreement = agreement
        self.last_proposal = self.last_proposal + self.alpha(self.last_proposal)*(agreement - self.last_proposal)
        
        return self.last_proposal

    
class Mediator():
    def __init__(self, agents):
        self.agents = agents
        self.last_step=0
        self.W = None
        self.tools = NegotiationTools()
        
    def start_new_task(self, image):
        self.last_step=0
        self.task = image
        self.initial_proposals = np.array([agent.new_task(self.task) for agent in self.agents])
        self.last_proposals = self.initial_proposals
        self.W = np.ones_like(self.initial_proposals)
        
        return self.last_proposals

        
    def negotiation(self, task, timeout = 1000):
        for i in range(self.last_step, self.last_step+timeout):
            if i==0:
                self.last_proposals = self.start_new_task(task)
                # self.agent_queues = [Queue() for a in self.agents]
                # self.agent_processes = [(self.agent_queues[a], Process(target=self.agents[a].propose, args=(self.agent_queues[a],))) for a in range(len(self.agents))]
            else:
                # Propose the new agreement to the agents
                
                self.last_proposals = np.array([agent.propose(self.last_agreement) for agent in self.agents]) # ((p0, u0), (p1, u1), ...)
                
                    
            self.last_step = i            
            self.last_agreement = np.divide(np.sum(self.last_proposals*self.W, axis=0), np.sum(self.W, axis=0))

            if self.tools.get_consensus(self.last_proposals).all():
                return 'consensus', self.last_agreement, self.last_proposals
            else:
                yield 'negotiation', self.last_agreement, self.last_proposals
        return 'timeout', self.last_agreement, self.last_proposals
    
# Creating the models
models = {AGENT_NAMES[i]: SegmentationModel(ALL_LABELS[i]) for i in range(len(AGENT_NAMES))}
# Load the checkpoints
for l, m in models.items():
    m.load_finetuned_network(epoch=275)
    
alphasliders = {m:[FloatSlider(description=l, min=0., max=1., step=0.01, value=np.array(0.5)) for l in CHANNEL_NAMES] for m in AGENT_NAMES}
sliders = [VBox([Label(model)]+alphasliders[model]) for model in alphasliders.keys()]
HBox(sliders)

def alpha_funcs_val():
    '''Definitions of the alpha_fun for each model. An alpha_fun takes the last proposal as input and outputs a value alpha for each pixel'''
    alpha_acc = {m: [1-slider.value for slider in alphasliders[m]] for m in alphasliders.keys()}
    functions = {modelname: lambda x, v=values: np.array(v) for (modelname, values) in alpha_acc.items()}
    return functions

def alpha_funcs_ep():
    '''Definitions of the alpha_fun for each model. An alpha_fun takes the last proposal as input and outputs a value alpha for each pixel'''
    def entropy_over_pixels(last_proposal):
        n_labels = last_proposal.shape[-1]
        entr = lambda x, base=n_labels, eps=10e-16: -np.sum(x*np.log(x+eps)/np.log(base),axis=-1)
        entr_over_pixels = entr(last_proposal)
        return np.expand_dims(entr_over_pixels, axis=-1)
    return {modelname: entropy_over_pixels for modelname in models.keys()}

def alpha_funcs_ei():
    '''Definitions of the alpha_fun for each model. An alpha_fun takes the last proposal as input and outputs a value alpha for each pixel'''
    from scipy.stats import entropy
    def entropy_over_image(last_proposal):
        n_labels = last_proposal.shape[-1]
        entr = lambda x, base=n_labels, eps=10e-16: -np.sum(x*np.log(x+eps)/np.log(base),axis=-1)
        entr_over_pixels = entr(last_proposal)
        return np.mean(entr_over_pixels)   
    return {model_name: entropy_over_image for model_name in models.keys()}



# Instancing agents and mediators

result_logger = NegotiationTools()

alpha_v = alpha_funcs_val()
alpha_ep = alpha_funcs_ep()
alpha_ei = alpha_funcs_ei()

METHODS = ['v', 'ep', 'ei']
STEPS = 500
DATASET = 'datasets/coco_animals_test_balanced.csv'
for i, (png_path, seg_path, input_sample, ground_truth) in enumerate(SegmentationModel('dummy').load_test_dataset(DATASET, batch_size=1)):
    
    agents_v = [Agent(modelname, models[modelname], alpha_v[modelname]) for modelname in AGENT_NAMES]
    mediator_v = Mediator(agents_v)
    agents_ep = [Agent(modelname, models[modelname], alpha_ep[modelname]) for modelname in AGENT_NAMES]
    mediator_ep = Mediator(agents_ep)
    agents_ei = [Agent(modelname, models[modelname], alpha_ei[modelname]) for modelname in AGENT_NAMES]
    mediator_ei = Mediator(agents_ei)
    
    
    ground_truth_np = ground_truth.numpy()
    png_path_str = png_path.numpy().astype(np.str).item()
    seg_path_str = seg_path.numpy().astype(np.str).item()
    # V

    next_step_v = enumerate(mediator_v.negotiation(input_sample, timeout=STEPS))
    for step, (status, curr_agreement, curr_proposals) in next_step_v:
        result_logger.log_step(sample=i, png_path=png_path_str, seg_path=seg_path_str, step=step, method='v', status=status, agreement=curr_agreement, proposals=curr_proposals, ground_truth=ground_truth_np)
        print("\rRunning: Sample: {}, Step: {}, Method: {}".format(i, step, 'v'), end='')
        
    # EP

    next_step_ep = enumerate(mediator_ep.negotiation(input_sample, timeout=STEPS))
    for step, (status, curr_agreement, curr_proposals) in next_step_ep:
        result_logger.log_step(sample=i, png_path=png_path_str, seg_path=seg_path_str, step=step, method='ep', status=status, agreement=curr_agreement, proposals=curr_proposals, ground_truth=ground_truth_np)
        print("\rRunning: Sample: {}, Step: {}, Method: {}".format(i, step, 'ep'), end='')
    # EI

    next_step_ei = enumerate(mediator_ei.negotiation(input_sample, timeout=STEPS))
    for step, (status, curr_agreement, curr_proposals) in next_step_ei:
        result_logger.log_step(sample=i, png_path=png_path_str, seg_path=seg_path_str, step=step, method='ei', status=status, agreement=curr_agreement, proposals=curr_proposals, ground_truth=ground_truth_np)
        print("\rRunning: Sample: {}, Step: {}, Method: {}".format(i, step, 'ei'), end='')
    
    if i % 10 ==1:
        result_logger.save()

result_logger.save()