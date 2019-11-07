from SyntheticSamples import *

template_name='chk'
template = templates[template_name]
N_LABELS = 4
gt = generate_ground_truth(template, N_LABELS)
SAMPLES = 30


mu_star_expert_list = [num/N_LABELS for num in [1+1e-6, 1.5] + list(range(N_LABELS)[2:])]
mu_expert_list = [num/N_LABELS for num in [1, 1.1, 1.5] + list(range(N_LABELS-1)[3:])]

# One expert agent for each label
log = pd.DataFrame()
for std in [.01, .05, .1, .2]:
    for mu_star in mu_star_expert_list:
        for mu in mu_expert_list:
            if mu > mu_star:
                continue
            prediction_runs = list()
            for i in range(SAMPLES):
                agents = [agent_multiclass_expert(mu_star, mu, std, c_star=l, n_labels=N_LABELS) for l in range(N_LABELS)]
                predictions = np.stack([generate_predictions(template, mu_matrix, std_matrix) for (mu_matrix, std_matrix) in agents])
                prediction_runs.append(predictions)
            run_result, outputs = exp.run_experiment_on_list(prediction_runs, [gt]*SAMPLES, return_outputs=True)
            run_result['mu_expert'] = mu
            run_result['mu_star_expert'] = mu_star
            run_result['std'] = std
            log = log.append(run_result, ignore_index=True)

log.to_csv('results/{}_one_expert_each_label.csv'.format(template_name))
