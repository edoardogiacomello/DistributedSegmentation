from SyntheticSamples import *

template_name='chk'
template = templates[template_name]
N_LABELS = 4
gt = generate_ground_truth(template, N_LABELS)
SAMPLES = 30

mu_star_unb_list = [num/N_LABELS for num in [1+1e-6, 1.5] + list(range(N_LABELS)[2:])]
mu_unb_list =[.4, .5, .6]
gamma_unb_list = [n/N_LABELS for n in range(N_LABELS) if n != 0]

mu_star_expert_list = [num/N_LABELS for num in [1+1e-6, 1.5] + list(range(N_LABELS)[2:])]
mu_expert_list = [num/N_LABELS for num in [1, 1.1, 1.5] + list(range(N_LABELS-1)[3:])]

log = pd.DataFrame()
for std in [.01, .05, .1, .2]:
    for unb_agents in [2, 4, 8]: 
        for mu_star_exp in mu_star_expert_list:
            for mu_exp in mu_expert_list:       
                for mu_star_unb in mu_star_unb_list:
                    for gamma in gamma_unb_list:
                        for mu_unb in mu_unb_list:
                            if mu_unb > mu_star_unb:
                                continue

                            prediction_runs = list()
                            
                            #####
                            expert = [agent_multiclass_expert(mu_star_exp, mu_exp, std, c_star=1, n_labels=N_LABELS)]
                            unbalanced = [agent_multiclass_unbalanced(mu_star_unb, mu_unb, gamma, std, c_star=2, n_labels=N_LABELS)]*unb_agents
                            agents = expert + unbalanced
                            
                            for i in range(SAMPLES):
                                predictions = np.stack([generate_predictions(template, mu_matrix, std_matrix) for (mu_matrix, std_matrix) in agents])
                                prediction_runs.append(predictions)

                            run_result, outputs = exp.run_experiment_on_list(prediction_runs, [gt]*SAMPLES, return_outputs=True)
                            run_result['mu_expert'] = mu_exp
                            run_result['mu_star_expert'] = mu_star_exp

                            run_result['mu_unbalanced'] = mu_unb
                            run_result['mu_star_unbalanced'] = mu_star_unb
                            run_result['gamma_unbalanced'] = gamma
                            run_result['unbalanced_agents'] = unb_agents
                            run_result['std'] = std

                            log = log.append(run_result, ignore_index=True)

log.to_csv('results/{}_one_expert_vs_N_unbalanced.csv'.format(template_name))
