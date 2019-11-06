import numpy as np
import NegotiationTools as negtools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb


 


def run_simple_aggregation(proposals, gt, mask, agg_method, noise_samples, noise_std, confidence_functions=None, binary_strategy='maximum', label_names=None, return_outputs=False):
    stats = negtools.StatisticsLogger()
    nt = negtools.NegTools()
    import Negotiation as neg
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
        agreement_function = lambda prop, binary_str, conf_funcs=confidence_functions: nt.weighted_average(prop, np.stack([conf_funcs[p](pr) for p, pr in enumerate(prop)]), binary_strategy=binary_str)
        
    # Run the aggregation
    if noise_samples == 0:
        agr = agreement_function(proposals, binary_strategy)
        results = results.append(stats.compute_statistics(gt, agr, '', mask=mask, label_names=label_names), ignore_index=True, sort=False)
        
        if not return_outputs:
            return results
        else:
            return results, agr
    
    else:
        for sample_run in range(noise_samples):
            proposals_noisy = np.stack([nt.add_noise(p, noise_std) for p in proposals])
            agr = agreement_function(proposals_noisy, binary_strategy)
            run_stats = stats.compute_statistics(gt, agr, '', mask=mask, label_names=label_names)
            results = results.append(run_stats, ignore_index=True, sort=False)
        means = results.mean(axis=0).to_frame().transpose()
        return means

def run_negotiation(proposals, gt, mask, noise_samples, noise_std, confidence_functions=None, binary_strategy='maximum', agent_names=None, label_names=None, MAX_STEPS=1000, return_outputs=False):
    stats = negtools.StatisticsLogger()
    nt = negtools.NegTools()
    import Negotiation as neg
    
    results = pd.DataFrame()
    
    # Run the aggregation
    if noise_samples == 0:
        last_agr, last_prop = neg.run_negotiation_on_proposasls(sample_id=0, 
                                                                        initial_proposals=proposals, 
                                                                        ground_truth=gt, 
                                                                        confidence_functions=confidence_functions, 
                                                                        method_name='negotiation', 
                                                                        log_process=False,
                                                                        agent_names=agent_names,
                                                                        max_steps=MAX_STEPS)
        results = results.append(stats.compute_statistics(gt, last_agr, '', mask=mask, label_names=label_names), ignore_index = True, sort=False)
        if not return_outputs:
            return results
        else:
            return results, last_agr, last_prop
    else:
        for sample_run in range(noise_samples):
            proposals_noisy = np.stack([nt.add_noise(p, noise_std) for p in proposals])
            last_agr, last_prop = neg.run_negotiation_on_proposasls(sample_id=sample_run, 
                                                                        initial_proposals=proposals_noisy, 
                                                                        ground_truth=gt, 
                                                                        confidence_functions=confidence_functions, 
                                                                        method_name='negotiation', 
                                                                        log_process=False,
                                                                        agent_names=agent_names,
                                                                        max_steps=MAX_STEPS)
            run_stats = stats.compute_statistics(gt, last_agr, '', mask=mask, label_names=label_names)
            results = results.append(run_stats, ignore_index=True, sort=False)
            print("\r run {}".format(sample_run), end='\r')
        means = results.mean(axis=0).to_frame().transpose()
        if not return_outputs:
            return means
        else:
            return means, last_agr, last_prop
    

def run_experiment_on_list(proposals_list, gt_list, return_mean=True, agent_names=None, label_names=None, MAX_STEPS=1000, return_outputs=False):
    ''' 
    Runs a given experiment with the given parameters and returns a DataFrame with the corresponding statistics
    
    :param proposals_list: list of initial proposals or predictions, of shape (Agents, H, W, Labels)
    :param gt_list: list of ground truths, of shape (H, W, Labels)
    :param MAX_STEPS - Negotiation steps timeout
    :param return_mean: Whether to return the mean over the provided list or the full record
    :param return_outputs: Wether to also return the output of each applied method
    :param label_names: 
    :return DataFrame containing the average metrics for the given samples if return_mean is true, the full DataFrame otherwise.
    '''
    stats = negtools.StatisticsLogger()
    nt = negtools.NegTools()
    import Negotiation as neg
    
    results = pd.DataFrame()
    outputs = list()
    for sample_id, (prop, gt) in enumerate(zip(proposals_list, gt_list)):
        sample_results = pd.DataFrame()
        sample_outputs = dict()
        
        mask = np.logical_not(nt.get_consensus(prop))
        if np.all(~mask):
            # Consensus may be enforced by construction, in these case we consider all the image
            mask = None
       
        # One shot methods
        mv_results = run_simple_aggregation(prop, gt, mask, agg_method='majority voting', noise_samples=0, noise_std=None, label_names=label_names, binary_strategy='maximum', return_outputs=return_outputs)
        if return_outputs:
            sample_outputs['Majority Voting'] = mv_results[1]
            mv_results = mv_results[0]
        mv_results['method'] = 'Majority Voting'
        
        sample_results = sample_results.append(mv_results, ignore_index=True, sort=False)

        mean_results = run_simple_aggregation(prop, gt, mask, agg_method='mean', noise_samples=0, noise_std=None, label_names=label_names, binary_strategy='maximum', return_outputs=return_outputs)
        if return_outputs:
            sample_outputs['Mean Proposal'] = mean_results[1]
            mean_results = mean_results[0]
        mean_results['method'] = 'Mean Proposal'
        
        sample_results = sample_results.append(mean_results, ignore_index=True, sort=False)

        max_results = run_simple_aggregation(prop, gt, mask, agg_method='maximum', noise_samples=0, noise_std=None, label_names=label_names, binary_strategy='maximum', return_outputs=return_outputs)
        if return_outputs:
            sample_outputs['Maximum Proposal'] = max_results[1]
            max_results = max_results[0]
        max_results['method'] = 'Maximum Proposal'
        
        sample_results = sample_results.append(max_results, ignore_index=True, sort=False)

        # Weighted Average methods based on confidence
        confidence_functions = [lambda x: nt.get_confidence(x, method='pixelwise_entropy')]*prop.shape[0]
        wa_results = run_simple_aggregation(prop, gt, mask, agg_method='weighted_mean_confidence', confidence_functions=confidence_functions, noise_samples=0, noise_std=None, label_names=label_names, binary_strategy='maximum', return_outputs=return_outputs)
        if return_outputs:
            sample_outputs['Weighted Mean - Pixelwise Entropy'] = wa_results[1]
            wa_results = wa_results[0]
        wa_results['method'] = 'Weighted Mean - Pixelwise Entropy'
        
        sample_results = sample_results.append(wa_results, ignore_index=True, sort=False)

        confidence_functions = [lambda x: nt.get_confidence(x, method='mean_entropy')]*prop.shape[0]
        wa_results = run_simple_aggregation(prop, gt, mask, agg_method='weighted_mean_confidence', confidence_functions=confidence_functions, noise_samples=0, noise_std=None, label_names=label_names, binary_strategy='maximum', return_outputs=return_outputs)
        if return_outputs:
            sample_outputs['Weighted Mean - Mean Entropy'] = wa_results[1]
            wa_results = wa_results[0]
        wa_results['method'] = 'Weighted Mean - Mean Entropy'
        
        sample_results = sample_results.append(wa_results, ignore_index=True, sort=False)

        confidence_functions = [lambda x: nt.get_confidence(x, method='convolution_entropy', convolution_size=3)]*prop.shape[0]
        wa_results = run_simple_aggregation(prop, gt, mask, agg_method='weighted_mean_confidence', confidence_functions=confidence_functions, noise_samples=0, noise_std=None, label_names=label_names, binary_strategy='maximum', return_outputs=return_outputs)
        if return_outputs:
            sample_outputs['Weighted Mean - 3x3 Conv Entropy'] = wa_results[1]
            wa_results = wa_results[0]
        wa_results['method'] = 'Weighted Mean - 3x3 Conv Entropy'
        
        sample_results = sample_results.append(wa_results, ignore_index=True, sort=False)

        confidence_functions = [lambda x: nt.get_confidence(x, method='convolution_entropy', convolution_size=5)]*prop.shape[0]
        wa_results = run_simple_aggregation(prop, gt, mask, agg_method='weighted_mean_confidence', confidence_functions=confidence_functions, noise_samples=0, noise_std=None, label_names=label_names, binary_strategy='maximum', return_outputs=return_outputs)
        if return_outputs:
            sample_outputs['Weighted Mean - 5x5 Conv Entropy'] = wa_results[1]
            wa_results = wa_results[0]
        wa_results['method'] = 'Weighted Mean - 5x5 Conv Entropy'
        
        sample_results = sample_results.append(wa_results, ignore_index=True, sort=False)

        # Negotiation based methods (Only keeps the last agreement)
        # Pixelwise entropy
        confidence_functions = [lambda x: nt.get_confidence(x, method='pixelwise_entropy')]*prop.shape[0]
        neg_results = run_negotiation(prop, gt, mask, confidence_functions=confidence_functions, noise_samples=0, noise_std=None, label_names=label_names, agent_names=agent_names, MAX_STEPS=MAX_STEPS, return_outputs=return_outputs)
        if return_outputs:
            sample_outputs['Negotiation - Pixelwise Entropy'] = {'agr': neg_results[1], 'prop': neg_results[2]}
            neg_results = neg_results[0]
        neg_results['method'] = 'Negotiation - Pixelwise Entropy'
        
        sample_results = sample_results.append(neg_results, ignore_index=True, sort=False)

        # Mean entropy
        confidence_functions = [lambda x: nt.get_confidence(x, method='mean_entropy')]*prop.shape[0]
        neg_results = run_negotiation(prop, gt, mask, confidence_functions=confidence_functions, noise_samples=0, noise_std=None, label_names=label_names, agent_names=agent_names, MAX_STEPS=MAX_STEPS, return_outputs=return_outputs)
        if return_outputs:
            sample_outputs['Negotiation - Mean Entropy'] = {'agr': neg_results[1], 'prop': neg_results[2]}
            neg_results = neg_results[0]
        neg_results['method'] = 'Negotiation - Mean Entropy'
        
        sample_results = sample_results.append(neg_results, ignore_index=True, sort=False)

        # Convolution entropy 3x3
        confidence_functions = [lambda x: nt.get_confidence(x, method='convolution_entropy', convolution_size=3)]*prop.shape[0]
        neg_results = run_negotiation(prop, gt, mask, confidence_functions=confidence_functions, noise_samples=0, noise_std=None, label_names=label_names, agent_names=agent_names, MAX_STEPS=MAX_STEPS, return_outputs=return_outputs)
        if return_outputs:
            sample_outputs['Negotiation - 3x3 Conv Entropy'] = {'agr': neg_results[1], 'prop': neg_results[2]}
            neg_results = neg_results[0]
        neg_results['method'] = 'Negotiation - 3x3 Conv Entropy'
        
        sample_results = sample_results.append(neg_results, ignore_index=True, sort=False)

        # Convolution entropy 5x5
        confidence_functions = [lambda x: nt.get_confidence(x, method='convolution_entropy', convolution_size=5)]*prop.shape[0]
        neg_results = run_negotiation(prop, gt, mask, confidence_functions=confidence_functions, noise_samples=0, noise_std=None, label_names=label_names, agent_names=agent_names, MAX_STEPS=MAX_STEPS, return_outputs=return_outputs)
        if return_outputs:
            sample_outputs['Negotiation - 5x5 Conv Entropy'] = {'agr': neg_results[1], 'prop': neg_results[2]}
            neg_results = neg_results[0]
        neg_results['method'] = 'Negotiation - 5x5 Conv Entropy'
        
        sample_results = sample_results.append(neg_results, ignore_index=True, sort=False)
        #sample_results['sample'] = sample_id
        
        
        results = results.append(sample_results, ignore_index=True)
        outputs.append(sample_outputs)
    if not return_mean:
        if not return_outputs:
            return results
        else:
            return results, outputs
        
    
    mean_values = pd.DataFrame()
    for method, df in results.groupby('method'):
        mv = df.mean(axis=0).to_frame().transpose()
        mv['method'] = method
        mean_values = mean_values.append(mv, ignore_index=True)
    if not return_outputs:
        return mean_values
    else:
        return mean_values, outputs