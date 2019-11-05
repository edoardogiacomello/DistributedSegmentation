from SyntheticSamples import *

template_name='bin_blob'
template = templates[template_name]
N_LABELS = 2
gt = generate_ground_truth(template, N_LABELS)
SAMPLES = 30
# Agents unbalanced on C1
log = pd.DataFrame()
for std in [0.01,0.05,0.1,0.2]:
    for unb_agents in [2, 4, 8]: 
        for mu_balanced in [.6, .75, .9]:
            for mu_1_unbalanced in [.6, .75, .9]:
                for mu_2_unbalanced in [0.4, 0.5]:
                    print("{} - 1 balanced agent (mu={}) vs {} unbalanced (mu1={}, mu2={}), noise std={}".format(template_name, mu_balanced, unb_agents, mu_1_unbalanced, mu_2_unbalanced, std) )
                    prediction_runs = list()
                    for i in range(SAMPLES):
                        balanced = agent_binary_balanced(mu=mu_balanced, std=std)
                        agents = [balanced] + [agent_binary_unbalanced(mu_1=mu_1_unbalanced, mu_2=mu_2_unbalanced, std=std) for a in range(unb_agents)]        
                        predictions = np.stack([generate_predictions(template, mu_matrix, std_matrix) for (mu_matrix, std_matrix) in agents])
                        prediction_runs.append(predictions)
                    run_result = exp.run_experiment_on_list(prediction_runs, [gt]*SAMPLES)
                    run_result['unbalanced_agents'] = unb_agents
                    run_result['mu_balanced'] = mu_balanced
                    run_result['mu_1_unbalanced'] = mu_1_unbalanced
                    run_result['mu_2_unbalanced'] = mu_2_unbalanced
                    run_result['std'] = std
                    log = log.append(run_result, ignore_index=True)

log.to_csv('results/{}_balanced_vs_unbalanced_c1.csv'.format(template_name))

# Agents unbalanced on C2
log = pd.DataFrame()
for std in [0.01,0.05,0.1,0.2]:
    for unb_agents in [2, 4, 8]: 
        for mu_balanced in [.6, .75, .9]:
            for mu_1_unbalanced in [0.4, 0.5]:
                for mu_2_unbalanced in [.6, .75, .9]:
                    print("{} - 1 balanced agent (mu={}) vs {} unbalanced (mu1={}, mu2={}), noise std={}".format(template_name, mu_balanced, unb_agents, mu_1_unbalanced, mu_2_unbalanced, std) )
                    prediction_runs = list()
                    for i in range(SAMPLES):
                        balanced = agent_binary_balanced(mu=mu_balanced, std=std)
                        agents = [balanced] + [agent_binary_unbalanced(mu_1=mu_1_unbalanced, mu_2=mu_2_unbalanced, std=std) for a in range(unb_agents)]        
                        predictions = np.stack([generate_predictions(template, mu_matrix, std_matrix) for (mu_matrix, std_matrix) in agents])
                        prediction_runs.append(predictions)
                    run_result = exp.run_experiment_on_list(prediction_runs, [gt]*SAMPLES)
                    run_result['unbalanced_agents'] = unb_agents
                    run_result['mu_balanced'] = mu_balanced
                    run_result['mu_1_unbalanced'] = mu_1_unbalanced
                    run_result['mu_2_unbalanced'] = mu_2_unbalanced
                    run_result['std'] = std
                    log = log.append(run_result, ignore_index=True)

log.to_csv('results/{}_balanced_vs_unbalanced_c2.csv'.format(template_name))

# Half agents unbalanced on C1 and half on C2
log = pd.DataFrame()
for std in [0.01,0.05,0.1,0.2]:
    for unb_agents in [2, 4, 8]: 
        for mu_balanced in [.6, .75, .9]:
            for mu_1_unbalanced in [.6, .75, .9]:
                for mu_2_unbalanced in [0.4, 0.5]:
                    print("{} - 1 balanced agent (mu={}) vs {} unbalanced (mu1={}, mu2={}), noise std={}".format(template_name, mu_balanced, unb_agents, mu_1_unbalanced, mu_2_unbalanced, std) )
                    prediction_runs = list()
                    for i in range(SAMPLES):
                        balanced = agent_binary_balanced(mu=mu_balanced, std=std)
                        unbalanced_c1 = [agent_binary_unbalanced(mu_1=mu_1_unbalanced, mu_2=mu_2_unbalanced, std=std) for a in range(unb_agents//2)]
                        unbalanced_c2 = [agent_binary_unbalanced(mu_1=mu_2_unbalanced, mu_2=mu_1_unbalanced, std=std) for a in range(unb_agents//2)]
                        agents = [balanced] + unbalanced_c1 + unbalanced_c2
                        predictions = np.stack([generate_predictions(template, mu_matrix, std_matrix) for (mu_matrix, std_matrix) in agents])
                        prediction_runs.append(predictions)
                    run_result = exp.run_experiment_on_list(prediction_runs, [gt]*SAMPLES)
                    run_result['unbalanced_agents'] = unb_agents
                    run_result['mu_balanced'] = mu_balanced
                    run_result['mu_1_unbalanced'] = mu_1_unbalanced
                    run_result['mu_2_unbalanced'] = mu_2_unbalanced
                    run_result['std'] = std
                    log = log.append(run_result, ignore_index=True)

log.to_csv('results/{}_balanced_vs_unbalanced_half.csv'.format(template_name))