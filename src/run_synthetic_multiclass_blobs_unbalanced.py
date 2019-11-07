from SyntheticSamples import *

template_name='blobs'
template = templates[template_name]
N_LABELS = 4
gt = generate_ground_truth(template, N_LABELS)
SAMPLES = 30

mu_star_unb_list = [num/N_LABELS for num in [1+1e-6, 1.5] + list(range(N_LABELS)[2:])]
mu_unb_list =[.4, .5, .6]
gamma_unb_list = [n/N_LABELS for n in range(N_LABELS) if n != 0]

# N Unbalanced agents for each class
log = pd.DataFrame()
for std in [.01, .05, .1, .2]:
    for unb_agents in [1, 2, 3]: 
        for mu_star in mu_star_unb_list:
            for gamma in gamma_unb_list:
                for mu in mu_unb_list:
                    if mu > mu_star:
                        continue
                    prediction_runs = list()
                    agents = list()
                    for l in range(N_LABELS):
                        agents += [agent_multiclass_unbalanced(mu_star, mu, gamma, std, c_star=l, n_labels=N_LABELS)]*unb_agents
                        
                    for i in range(SAMPLES):
                        predictions = np.stack([generate_predictions(template, mu_matrix, std_matrix) for (mu_matrix, std_matrix) in agents])
                        prediction_runs.append(predictions)
                        
                    run_result, outputs = exp.run_experiment_on_list(prediction_runs, [gt]*SAMPLES, return_outputs=True)
                    run_result['mu_unbalanced'] = mu
                    run_result['mu_star_unbalanced'] = mu_star
                    run_result['gamma_unbalanced'] = gamma
                    run_result['unbalanced_agents'] = unb_agents
                    run_result['std'] = std
                    
                    log = log.append(run_result, ignore_index=True)
                    
log.to_csv('results/{}_n_unbalanced_each_label.csv'.format(template_name))