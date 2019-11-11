import numpy as np
import NegotiationTools as negtools
import matplotlib.pyplot as plt
from skimage.draw import rectangle
import skimage as skim
import pandas as pd
import seaborn as sb
import Experiments as exp
import math

stats = negtools.StatisticsLogger()
nt = negtools.NegTools()

W = 25
H = 25
W2 = 5
H2 = 5
templates = dict()

templates['bin_blob'] = np.zeros((W, H), dtype=np.uint8)
templates['bin_blob'][tuple(rectangle(start=(5,5), extent=(15,15), shape=(W,H)))] = 1

templates['bin_chk'] = np.zeros((W, H), dtype=np.uint8)
is_one = True
for i in range(0,W,W2):
    for j in range(0,H,H2):
        if is_one is True:
            templates['bin_chk'][tuple(rectangle(start=(i,j), extent=(W2,H2), shape=(W,H)))] = 1
        is_one = not is_one
        
n_class = 4
blobs_w = 8
blobs_h = 8
w_off = 2
h_off = 2
blobs_val = 1

templates['blobs'] = np.zeros((W, H), dtype=np.uint8)
for i in range(0,W-w_off,blobs_w+2*w_off):
    for j in range(0,H-h_off,blobs_h+2*h_off):
        templates['blobs'][tuple(rectangle(start=(i+w_off,j+h_off), extent=(blobs_w,blobs_h), shape=(W,H)))] = blobs_val
        blobs_val = blobs_val+1

W = 25
H = 25
CHK_W = 5
CHK_H = 5
val = 0
n_class = 4
templates['chk'] = np.zeros((W, H), dtype=np.uint8)
is_one = True
for i in range(0,W,CHK_W):
    for j in range(0,H,CHK_H):
        templates['chk'][tuple(rectangle(start=(i,j), extent=(CHK_W,CHK_H), shape=(W,H)))] = val + 1
        val = (val + 1 ) % n_class
templates['chk'] = templates['chk'] - 1

for i, (name, templ) in enumerate(templates.items()):
    plt.subplot(2, math.ceil(len(templates.items())/2), i+1)
    plt.imshow(templ)
    plt.title(name)

def agent_binary_balanced(mu, std):

    mu_mat = [[mu, 1.-mu],
           [1.-mu, mu]]
    std_mat = np.ones_like(mu_mat)*std
    return (np.asarray(mu_mat),np.asarray(std_mat))

def agent_binary_unbalanced(mu_1, mu_2, std):
    
    mu_mat = [[mu_1, 1.-mu_1],
           [1.-mu_2, mu_2]]
    std_mat = np.ones_like(mu_mat)*std
    return (np.asarray(mu_mat),np.asarray(std_mat))


def agent_expert(mu_star, mu, std, c_star, n_labels):
    '''
    :param mu_star - "competence" on expertise class
    :param mu - "competence" on non expertise classes
    :param std - standard deviation
    :param c_star - index of expertise class
    :param n_labels - number of labels
    '''
    samp_mu = np.random.normal(loc=mu, scale=std)
    samp_mu_star = np.random.normal(loc=mu_star, scale=std)
    samp_c_star_others = (1. - samp_mu_star)/(n_labels - 1.)
    samp_gamma = (1. - samp_c_star_others - samp_mu) / (n_labels-2.) if n_labels > 2 else 0

    mat = list()
    for t in range(n_labels):
        row = list()
        for p in range(n_labels):
            if p == c_star and t == c_star:
                row.append(samp_mu_star)
            elif t == c_star or p == c_star:
                row.append(samp_c_star_others)
            elif p == t:
                row.append(samp_mu)
            else:
                row.append(samp_gamma)
        mat.append(row)
    return np.asarray(mat)

def generate_predictions(gt_template, mu_matrix, std_matrix):
    prediction = list()
    # Iterating through the columns (predicted classes) of agent matrix, filling the corresponding ground_truth areas with predictions
    for pred_label in range(mu_matrix.shape[-1]):
        label_image = np.zeros_like(gt_template)
        for true_label in np.unique(gt_template):
            agent_labels = np.random.normal(loc=np.ones_like(gt_template)*mu_matrix[true_label, pred_label],scale=np.ones_like(gt_template)*std_matrix[true_label,pred_label])
            label_image = np.where(gt_template == true_label, agent_labels, label_image)
        prediction.append(label_image)
    prediction = np.stack(prediction, axis=-1)
    # Normalization
    prediction = np.clip(prediction, 0.0, 1.0)
    prediction = prediction / prediction.sum(axis=-1, keepdims=True) 
    return prediction

def generate_ground_truth(gt_template, n_labels):
    gt = list()
    for l in range(n_labels):
        gt_slice = np.where(gt_template==l, 1.0, 0.0)
        gt.append(gt_slice)
    return np.stack(gt, axis=-1)

def agent_multiclass_expert(mu_star, mu, std, c_star, n_labels):
    '''
    :param mu_star - "competence" on expertise class
    :param mu - "competence" on non expertise classes
    :param std - standard deviation
    :param c_star - index of expertise class
    :param n_labels - number of labels
    '''
    gamma = 1 - mu - (1 - mu_star)/(n_labels-1)
    
    mat = list()
    for t in range(n_labels):
        row = list()
        for p in range(n_labels):
            if p == c_star and t == c_star:
                row.append(mu_star)
            elif t == c_star or p == c_star:
                row.append((1.-mu_star)/(n_labels - 1.))
            elif p == t:
                row.append(mu)
            else:
                row.append(gamma/(n_labels - 2.))
        mat.append(row)
    mu_mat = np.asarray(mat)
    std_mat = np.ones_like(mu_mat)*std
    return mu_mat, std_mat

def agent_multiclass_unbalanced(mu_star, mu, gamma, std, c_star, n_labels):
    '''
    :param mu_star - "competence" on expertise class
    :param mu - "competence" on non expertise classes
    :param std - standard deviation
    :param c_star - index of expertise class
    :param n_labels - number of labels
    '''
    
    mat = list()
    for t in range(n_labels):
        row = list()
        for p in range(n_labels):
            if p == c_star and t == c_star:
                row.append(mu_star)
            elif t == c_star:
                row.append((1.-mu_star)/(n_labels - 1.))
            elif p == c_star:
                row.append((1.-mu)*gamma)
            elif p == t:
                row.append(mu*gamma)
            else:
                row.append((1.-gamma)/(n_labels - 2.))
        mat.append(row)
    mu_mat = np.asarray(mat)
    std_mat = np.ones_like(mu_mat)*std
    return mu_mat, std_mat