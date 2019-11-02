import numpy as np
import NegotiationTools as negtools
import matplotlib.pyplot as plt
from NegotiationConfig import *
import pandas as pd
import seaborn as sb

stats = negtools.StatisticsLogger()
nt = negtools.NegTools()
import Negotiation as neg 


def run_simple_aggregation(proposals, gt, mask, agg_method, noise_samples, noise_std, confidence_functions=None, binary_strategy='maximum'):
    results = pd.DataFrame()
    
    # Define which function to call and the human readable names for each method
    if agg_method == 'majority voting':
        agreement_function = lambda prop, binary_str: nt.compute_majority_voting(prop, binary_strategy=binary_str)
    elif agg_method == 'mean':
        agreement_function = lambda prop, binary_str: nt.mean_proposal(prop, binary_strategy=binary_str)
    elif agg_method == 'maximum':
        agreement_function = lambda prop, binary_str: nt.max_proposal(prop, binary_strategy=binary_str)
    elif agg_method == 'weighted_mean_confidence':
        # weighted_average() takes a vector of proposals and a vector of confidences, which are calculated applying each confidence function to the corresponding slice of the proposals
        agreement_function = lambda prop, binary_str, conf_funcs=confidence_functions: nt.weighted_average(prop, np.stack([conf_funcs[p](pr) for p, pr in enumerate(prop)]))
        
    # Run the aggregation
    if noise_samples == 0:
        agr = agreement_function(proposals, binary_strategy)
        results = results.append(stats.compute_statistics(gt, agr, '', mask=mask), ignore_index=True, sort=False)
        return results
    else:
        for sample_run in range(noise_samples):
            proposals_noisy = np.stack([nt.add_noise(p, noise_std) for p in proposals])
            agr = agreement_function(proposals_noisy, binary_strategy)
            run_stats = stats.compute_statistics(gt, agr, '', mask=mask)
            results = results.append(run_stats, ignore_index=True, sort=False)
        means = results.mean(axis=0).to_frame().transpose()
        return means

def run_negotiation(proposals, gt, mask, noise_samples, noise_std, confidence_functions=None, binary_strategy='maximum', MAX_STEPS=1000):
    results = pd.DataFrame()
    
    # Run the aggregation
    if noise_samples == 0:
        last_agr, last_prop = neg.run_negotiation_on_proposasls(sample_id=0, 
                                                                        initial_proposals=proposals, 
                                                                        ground_truth=gt, 
                                                                        confidence_functions=confidence_functions, 
                                                                        method_name='negotiation', 
                                                                        log_process=False,
                                                                        max_steps=MAX_STEPS)
        results = results.append(stats.compute_statistics(gt, last_agr, '', mask=mask), ignore_index = True, sort=False)
        return results
    else:
        for sample_run in range(noise_samples):
            proposals_noisy = np.stack([nt.add_noise(p, noise_std) for p in proposals])
            last_agr, last_prop = neg.run_negotiation_on_proposasls(sample_id=sample_run, 
                                                                        initial_proposals=proposals_noisy, 
                                                                        ground_truth=gt, 
                                                                        confidence_functions=confidence_functions, 
                                                                        method_name='negotiation', 
                                                                        log_process=False,
                                                                        max_steps=MAX_STEPS)
            run_stats = stats.compute_statistics(gt, last_agr, '', mask=mask)
            results = results.append(run_stats, ignore_index=True, sort=False)
            print("\r run {}".format(sample_run), end='\r')
        means = results.mean(axis=0).to_frame().transpose()
        return means