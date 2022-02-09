import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-muted')
plt.rcParams['text.usetex'] = True
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.size'] = 5.0
plt.rcParams['xtick.minor.size'] = 3.0
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 3.0
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 1.0 
plt.rcParams['legend.handlelength'] = 5.0

# metrics[(key, M, K, lam)] = ['rel_err', 'precision', 'recall', 'model_entr', 'base_entr',  'NDCG', 'model_rep', 'base_rep']

metrics = pickle.load(open('synthetic_save/metrics.pkl', 'rb'))
M_list = [20]
K_list = [10]
lam_values = [0, 0.5, 1, 2, 4, 8]
key_list = ['MCAT',
            'PPO',
            'MCATDeterministic',
            'MCAT_FC',
            'ModalityDropout',
            'CascadedResidualAutoencoder',
            'UnifiedRepresentationNetwork',
            'PGOS',
            'Random']
tradeoff_keys = ['MCAT',
            'MCAT_FC',
            'ModalityDropout',
            'CascadedResidualAutoencoder',
            'UnifiedRepresentationNetwork',
            'PGOS']
key_map = {'MCAT': 'MCAT',
            'PPO': 'PPO',
            'MCATDeterministic': 'Deterministic',
            'MCAT_FC': 'FC Score Net',
            'ModalityDropout': 'MD',
            'CascadedResidualAutoencoder': 'CRA',
            'UnifiedRepresentationNetwork': 'URN',
            'PGOS': 'PGOS',
            'Random': 'Random',
            'Optimal Enrollment': 'Max Enrollment',
            'MCAT_Full': 'FRAMM No Missing',
            'FRAMM_FC_Full': 'FC No Missing'}

#######
### Tradeoffs
#######

# Fixed M = 20, K = 10, Varying Lambda
rel_errs = [[metrics[(k, 20, 10, l)]['rel_err'] for l in lam_values] for k in tradeoff_keys]
entropies = [[metrics[(k, 20, 10, l)]['model_entr'] for l in lam_values] for k in tradeoff_keys]
for xs, ys, k in zip(rel_errs, entropies, tradeoff_keys):
    plt.plot(xs, ys, label = key_map[k])

plt.xlabel('Relative Error')
plt.ylabel('Entropy')
plt.title('Relative  Error vs. Entropy for Varying $\lambda$')
plt.legend(loc='lower right', ncol=2, prop={'size': 6})
plt.grid(False)
plt.axes().set_facecolor("white")
plt.tight_layout()
plt.savefig('synthetic_figures/tradeoffs_20_10.pdf')
plt.clf()



#######
### Aggregate Effect
#######

# Optimal Enrollment, Lambda = 1, Lambda = 4 Representations for M = 20, K = 10
labels = ['White', 'Hispanic', 'Black', 'Asian', 'Mixed','Others']
optimal_values = metrics[('MCAT', 20, 10, 0)]['base_rep']
lam1_values = metrics[('MCAT', 20, 10, 1)]['model_rep']
lam4_values = metrics[('MCAT', 20, 10, 4)]['model_rep']
lam1_percentages = [(lam1_values[i] - optimal_values[i]) / optimal_values[i] for i in range(len(labels))]
lam4_percentages = [(lam4_values[i] - optimal_values[i]) / optimal_values[i] for i in range(len(labels))]
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars
fig, ax = plt.subplots()
bars1 = ax.bar(x - 1.5*width, optimal_values, width, label='No Fairness Enrollment')
bars2 = ax.bar(x, lam1_values, width, label='FRAMM $\lambda = 1$')
for i, (b,v) in enumerate(zip(bars2, lam1_percentages)):
    if i == 0:
        ax.text(b.get_x() + (1.0 * b.get_width()/2.), 1.025*b.get_height(),
                "{0:+d}\%".format(int(100 * v)),
                ha='center', va='bottom', size='x-small', fontweight='bold')
    else:
        ax.text(b.get_x() + (0.8 * b.get_width()/2.), 1.05*b.get_height(),
                "{0:+d}\%".format(int(100 * v)),
                ha='center', va='bottom', size='x-small', fontweight='bold')
bars3 = ax.bar(x + 1.5*width, lam4_values, width, label='FRAMM $\lambda = 4$')
for i, (b,v) in enumerate(zip(bars3, lam4_percentages)):
    if i == 0:
        ax.text(b.get_x() + (1.0 * b.get_width()/2.), 1.025*b.get_height(),
                "{0:+d}\%".format(int(100 * v)),
                ha='center', va='bottom', size='x-small', fontweight='bold')
    else:
        ax.text(b.get_x() + (1.2 * b.get_width()/2.), 1.05*b.get_height(),
                "{0:+d}\%".format(int(100 * v)),
                ha='center', va='bottom', size='x-small', fontweight='bold')
ax.set_ylabel('Percentage of Enrolled Population')
ax.set_title('Diversity by Group and $\lambda$')
plt.xticks(x, labels)
ax.legend()
plt.grid(False)
ax.set_facecolor("white")
plt.tight_layout()
plt.savefig('synthetic_figures/aggregate_20_10.pdf')
plt.clf()

   
      
metrics_full = pickle.load(open('synthetic_save/metrics_full.pkl', 'rb'))
key_list_full = ['MCAT',
            'MCAT_Full',
            'FRAMM_FC_Full',
            'Random'] 

metrics_noHist = pickle.load(open('synthetic_save/metrics_noHist.pkl', 'rb'))
key_list_noHist = ['MCAT',
            'MCAT_Full',
            'FRAMM_FC_Full'] 
key_map_noHist = {'MCAT': 'FRAMM No Hist.',
            'MCAT_Full': 'FRAMM No Miss. No Hist.',
            'FRAMM_FC_Full': 'FC No Miss. No Hist.'}


#######
### Augmentation Tradeoffs
#######

# Fixed M = 20, K = 10, Varying Lambda
rel_errs = [[metrics_full[(k, 20, 10, l)]['rel_err'] for l in lam_values[1:]] for k in key_list_full if k not in ['Random']]
entropies = [[metrics_full[(k, 20, 10, l)]['model_entr'] for l in lam_values[1:]] for k in key_list_full if k not in ['Random']]
for xs, ys, k in zip(rel_errs, entropies, [key for key in key_list_full if key not in ['Random']]):
    plt.plot(xs, ys, label = (key_map[k] if k != 'MCAT' else 'FRAMM'))


plt.xlabel('Relative Error')
plt.ylabel('Entropy')
plt.title('Relative  Error vs. Entropy for Varying $\lambda$')
plt.legend(loc='lower right', prop={'size': 6})
plt.grid(False)
plt.axes().set_facecolor("white")
plt.tight_layout()
plt.savefig('synthetic_figures/augmentation_tradeoffs_20_10.pdf')
plt.clf()

#######
### Augmentation-Modality Tradeoffs
#######

plt.plot(rel_errs[0], entropies[0], label='FRAMM')

rel_errs = [[metrics_noHist[(k, 20, 10, l)]['rel_err'] for l in lam_values[1:]] for k in key_list_noHist if k not in ['Random']]
entropies = [[metrics_noHist[(k, 20, 10, l)]['model_entr'] for l in lam_values[1:]] for k in key_list_noHist if k not in ['Random']]
for xs, ys, k in zip(rel_errs, entropies, [key for key in key_list_noHist if key not in ['Random']]):
    plt.plot(xs, ys, label = key_map_noHist[k])
    
plt.xlabel('Relative Error')
plt.ylabel('Entropy')
plt.title('Relative  Error vs. Entropy for Varying $\lambda$')
plt.legend(loc='lower right', prop={'size': 6})
plt.grid(False)
plt.axes().set_facecolor("white")
plt.tight_layout()
plt.savefig('synthetic_figures/augmentation_modality_tradeoffs_20_10.pdf')
plt.clf()

#######
### Modality Tradeoffs
#######

for xs, ys, k in zip(rel_errs, entropies, [key for key in key_list_noHist if key not in ['Random']]):
    plt.plot(xs, ys, label = key_map_noHist[k])
    
plt.xlabel('Relative Error')
plt.ylabel('Entropy')
plt.title('Relative  Error vs. Entropy for Varying $\lambda$')
plt.legend(loc='lower right', prop={'size': 6})
plt.grid(False)
plt.axes().set_facecolor("white")
plt.tight_layout()
plt.savefig('synthetic_figures/modality_tradeoffs_20_10.pdf')
plt.clf()