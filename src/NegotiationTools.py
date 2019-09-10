import numpy as np
import pandas as pd
from DCSegUtils import *
import datetime


class NegotiationTools():

    def __init__(self):
        self.majority = None
        
        columns = [
                    'sample', # Sample number
                    'png_path', # Path of the input file 
                    'seg_path', # Path for the ground truth
                    'method', # Method name
                    'step', # Negotiation step index
                    'agent', # Agent index
                    'status', # Comment about the current run (consensus reached, timeout or still running)
                    'consensus_start', # Number of pixels to negotiate (start)
                    'consensus_current', # Number of pixels to negotiate (current)
                    'd_prop_agr_MAE', # MAE Distance between current proposal and agreement (agent convergence)
                    'd_prop_agr_DSC', # Dice Score between current proposal and agreement [Binarized] (agent convergence)
                    'd_prop_gt_MAE', # MAE Distance between current proposal and ground truth (Agent absolute performance for the current proposal)
                    'd_prop_gt_DSC', # Dice Score Distance between current proposal and ground truth [Binarized] (Agent absolute performance for the current proposal)
                    'd_agr_gt_MAE', # Distance between agreement and ground truth (corvengence quality)
                    'd_agr_gt_DSC', # Distance between agreement and ground truth [Binarized] (corvengence quality)
                    'd_mv_gt_MAE', # Distance between majority voting and ground truth [Represented as float] (baseline quality)
                    'd_mv_gt_DSC', # Distance between majority voting and ground truth (baseline quality)
                    'd_mv_agr_MAE', # Distance between majority voting and agreement [Represented as float] (baseline)
                    'd_mv_agr_DSC', # Distance between majority voting and agreement (baseline)
                    'fp_agr_gt',  # False Positives  for the current agreement wrt the ground truth
                    'fp_mv_gt', # False Positives  for the current majority voting result wrt the ground truth
                    'fp_prop_gt', # False Positives for the current proposal wrt the ground truth
                    'fn_agr_gt',  # False Negatives for the current agreement wrt the ground truth
                    'fn_mv_gt', # False Negatives for the current majority voting result wrt the ground truth
                    'fn_prop_gt', # False Negatives for the current proposal wrt the ground truth
                    'tp_agr_gt',  # True Positives for the current agreement wrt the ground truth
                    'tp_mv_gt', # True Positives for the current majority voting result wrt the ground truth
                    'tp_prop_gt', # True Positives for the current proposal wrt the ground truth
                    'tn_agr_gt', # True Negatives for the current agreement wrt the ground truth
                    'tn_mv_gt', # True Negatives for the current majority voting result wrt the ground truth
                    'tn_prop_gt' # True Negatives for the current proposal wrt the ground truth
                  ]
        
        self.pd = pd.DataFrame(columns=columns)
        
    
    
    def get_consensus(self, proposals_np):
        binary_predictions = np.equal(proposals_np, proposals_np.max(axis=3)[...,np.newaxis])
        # For each pixel check if there's any label fow which every agent proposes True
        return np.all(binary_predictions, axis=0).any(axis=-1)

    def compute_majority_voting(self, proposals_np):
        '''Calculates the Majority voting between the given agent proposals. Resolves ties by picking the highest score for the accounted labels'''
        binary_predictions = np.equal(proposals_np, proposals_np.max(axis=3)[...,np.newaxis])
        votes = np.count_nonzero(binary_predictions, axis=0)
        majority = np.equal(votes, votes.max(axis=-1, keepdims=True))

        # Get a binary mask for tie pixels (h, w)
        ties_mask = np.count_nonzero(majority, axis=-1)>1
        if ties_mask.any():
            # Replicate the mask to match prediction matrix [ag, h, w, labels]
            ties_exp  = np.tile(ties_mask[np.newaxis, ..., np.newaxis], [len(AGENT_NAMES), 1, 1, 5])
            # Use this mask to get only the predictions for which there is a tie, for each pixel (every agent that voted "True" for a label and pixel in which there's a tie)
            ties_preds = np.logical_and(binary_predictions, ties_exp)
            # Fetch prediction scores corresponding to ties
            ties_weights = np.where(ties_preds, proposals_np, 0.0).sum(axis=0)
            # where there's a tie, select the label with the maximum score, otherwise keep the last winner
            solved_ties = np.equal(ties_weights, ties_weights.max(axis=-1, keepdims=True))
            majority = np.where(ties_preds.any(axis=0), solved_ties, majority)

        return majority
    
    
    def log_step(self,sample, png_path, seg_path,  step,  method, status, agreement, proposals, ground_truth):
        
        if step==0:
            # Computation to be performed just once per sample
            self.majority = self.compute_majority_voting(proposals)
            self.majority_float = self.majority.astype(np.float32)
            self.consensus_start = np.count_nonzero(np.logical_not(self.get_consensus(proposals)))
            self.ground_truth = ground_truth
            self.ground_truth_bool = ground_truth.astype(np.bool)
            
            self.mv_stats = {
                'd_mv_gt_MAE': np.mean(np.abs(self.majority_float - self.ground_truth)), 
                'fp_mv_gt': np.count_nonzero(np.logical_and(self.majority, np.logical_not(self.ground_truth_bool))),
                'fn_mv_gt': np.count_nonzero(np.logical_and(np.logical_not(self.majority), self.ground_truth_bool)),
                'tp_mv_gt': np.count_nonzero(np.logical_and(self.majority, self.ground_truth_bool)),
                'tn_mv_gt': np.count_nonzero(np.logical_and( np.logical_not(self.majority), np.logical_not(self.ground_truth_bool))),
            }
            self.mv_stats.update({'d_mv_gt_DSC': 2.*self.mv_stats['tp_mv_gt']/(2*self.mv_stats['tp_mv_gt'] + self.mv_stats['fp_mv_gt'] + self.mv_stats['fn_mv_gt'])})
        
        # Computation to be performed once per step (as proposals and agreement are different at each call)
        binary_proposals = np.equal(proposals, proposals.max(axis=3)[...,np.newaxis])
        binary_agreement = np.equal(agreement, agreement.max(axis=-1)[...,np.newaxis])
        
        step_row = { # This data remains the same for each agent performing the same step
                'sample':sample,
                'png_path':png_path,
                'seg_path':seg_path,
                'method':method, 
                'step':step,
                'status':status,
                'consensus_start': self.consensus_start, 
                'consensus_current': np.count_nonzero(np.logical_not(self.get_consensus(proposals))), 

                'd_agr_gt_MAE': np.mean(np.abs(agreement - self.ground_truth)),
                'd_mv_agr_MAE': np.mean(np.abs(self.majority_float - agreement)),
                'd_mv_agr_DSC': self.dice_score(self.majority, binary_agreement),
                'fp_agr_gt': np.count_nonzero(np.logical_and(binary_agreement, np.logical_not(self.ground_truth_bool))),
                'fn_agr_gt': np.count_nonzero(np.logical_and(np.logical_not(binary_agreement), self.ground_truth_bool)),
                'tp_agr_gt': np.count_nonzero(np.logical_and(binary_agreement, self.ground_truth_bool)),
                'tn_agr_gt': np.count_nonzero(np.logical_and(np.logical_not(binary_agreement), np.logical_not(self.ground_truth_bool)))
            }
        step_row.update({'d_agr_gt_DSC': 2.*step_row['tp_agr_gt']/(2*step_row['tp_agr_gt'] + step_row['fp_agr_gt'] + step_row['fn_agr_gt'])})
        
        # Computation to be performed for each agent (there is one proposal per agent)
        for ag_id in range(proposals.shape[0]):
            agent_row = {
                'agent':AGENT_NAMES[ag_id],
                'd_prop_agr_MAE': np.mean(np.abs(proposals[ag_id] - agreement)),
                'd_prop_agr_DSC': self.dice_score(binary_proposals[ag_id], binary_agreement),
                'd_prop_gt_MAE': np.mean(np.abs(proposals[ag_id] - self.ground_truth)),                
                'fp_prop_gt': np.count_nonzero(np.logical_and(binary_proposals[ag_id], np.logical_not(self.ground_truth_bool))),
                'fn_prop_gt': np.count_nonzero(np.logical_and(np.logical_not(binary_proposals[ag_id]), self.ground_truth_bool)),
                'tp_prop_gt': np.count_nonzero(np.logical_and(binary_proposals[ag_id], self.ground_truth_bool)),
                'tn_prop_gt': np.count_nonzero(np.logical_and(np.logical_not(binary_proposals[ag_id]), np.logical_not(self.ground_truth_bool)))
            }
            # Derived quantities
            agent_row.update({'d_prop_gt_DSC': 2.*agent_row['tp_prop_gt']/(2*agent_row['tp_prop_gt'] + agent_row['fp_prop_gt'] + agent_row['fn_prop_gt'])})
            
            # Collecting data
            agent_row.update(step_row)
            agent_row.update(self.mv_stats)
            
            # Logging
            self.pd = self.pd.append(agent_row, ignore_index=True)
            
    def dice_score(self, pred, true):
        tp = np.count_nonzero(np.logical_and(pred, true))
        fp = np.count_nonzero(np.logical_and(np.logical_not(pred), true))
        fn = np.count_nonzero(np.logical_and(pred, np.logical_not(true)))
        return 2.*tp/(2*tp + fp + fn)
    
    
    def save(self):
        self.pd.to_csv('results/run_{}.csv'.format(str(datetime.datetime.now())))