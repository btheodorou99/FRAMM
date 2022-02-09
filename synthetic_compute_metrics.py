aqimport os
import torch
import pickle
import numpy as np
from models import *
from data_object import *
from loss_metrics import *
import torch.nn.functional as F
from scipy.stats import bootstrap
from torch.utils.data import DataLoader
import ranking_metrics as ranking_metrics
from torch.distributions import Categorical
from torch.utils.data.sampler import SubsetRandomSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
c
def compute_test_metrics(net, K, M):
	if net is not None:
		net.eval()
	
	rel_dif=[]
	precision=[]
	recall = []
	model_rep=[]
	base_rep = []
	model_entr = []
	base_entr = []
	NDCG = []
	for batch in test_dataloader:
		trial = batch['trial'].to(device)
		label = batch['label'].to(device).squeeze()
		eth_label = batch['eth_label'].to(device)
		inv_static = batch['inv_static'].to(device)
		inv_dx = F.one_hot(batch['inv_dx'].type(torch.int64), num_classes=260).to(device)
		inv_dx_len = batch['inv_dx_len'].cpu().type(torch.int64)
		inv_rx = F.one_hot(batch['inv_rx'].type(torch.int64), num_classes=100).to(device)
		inv_rx_len = batch['inv_rx_len'].cpu().type(torch.int64)
		inv_enroll = batch['inv_enroll'].to(device)
		inv_enroll_len = batch['inv_enroll_len'].cpu().type(torch.int64)
		inv_mask = batch['inv_mask'].to(device)

		if torch.sum(label) == 0:
			continue						

		with torch.no_grad():
			if net is None:
				score = torch.rand(label.size(0), M).to(device)
			elif isinstance(net, MCAT):
				score = net(trial, inv_static, inv_dx, inv_dx_len, inv_rx, inv_rx_len, inv_enroll, inv_enroll_len, inv_mask, M).squeeze(-1)
			elif isinstance(net, MCAT_FC):
				score = net(trial, inv_static, inv_dx, inv_dx_len, inv_rx, inv_rx_len, inv_enroll, inv_enroll_len, inv_mask, M).squeeze(-1)
			elif isinstance(net, ModalityDropout):
				score = net(trial, inv_static, inv_dx, inv_dx_len, inv_rx, inv_rx_len, inv_enroll, inv_enroll_len, inv_mask, M).squeeze(-1)
			elif isinstance(net, CascadedResidualAutoencoderNetwork):
				score = net(trial, inv_static, inv_dx, inv_dx_len, inv_rx, inv_rx_len, inv_enroll, inv_enroll_len, inv_mask, M).squeeze(-1)
			elif isinstance(net, UnifiedRepresentationNetwork):
				score = net(trial, inv_static, inv_dx, inv_dx_len, inv_rx, inv_rx_len, inv_enroll, inv_enroll_len, inv_mask, M).squeeze(-1)

		rel_dif.extend(ranking_metrics.rel_err(score, label, K))
		precision.extend(ranking_metrics.precision_K(score, label, K))
		recall.extend(ranking_metrics.recall_K(score, label, K))
		NDCG.extend(ranking_metrics.NDCG_K(score, label, K))
		[avg_repn, base_repn] = ranking_metrics.compute_repn(score, label, K, eth_label)
		model_entr.extend(Categorical(probs = avg_repn).entropy().tolist())
		base_entr.extend(Categorical(probs = base_repn).entropy().tolist())
		model_rep.extend(avg_repn)
		base_rep.extend(base_repn)
    
	model_rep = torch.mean(torch.stack(model_rep), dim=0).tolist()
	base_rep = torch.mean(torch.stack(base_rep), dim=0).tolist()

	rel_dif = bootstrap(np.expand_dims(np.array(rel_dif), axis=0), np.nanmean, vectorized=False).confidence_interval
	precision = bootstrap(np.expand_dims(np.array(precision), axis=0), np.nanmean, vectorized=False).confidence_interval
	recall = bootstrap(np.expand_dims(np.array(recall), axis=0), np.nanmean, vectorized=False).confidence_interval
	model_entr = bootstrap(np.expand_dims(np.array(model_entr), axis=0), np.nanmean, vectorized=False).confidence_interval
	base_entr = bootstrap(np.expand_dims(np.array(base_entr), axis=0), np.nanmean, vectorized=False).confidence_interval
	NDCG = bootstrap(np.expand_dims(np.array(NDCG), axis=0), np.nanmean, vectorized=False).confidence_interval
  
	return np.mean(rel_dif), np.mean(precision), np.mean(recall), np.mean(model_entr), np.mean(base_entr), np.mean(NDCG), model_rep, base_rep, np.mean(rel_dif) - rel_dif[0], np.mean(precision) - precision[0], np.mean(recall) - recall[0], np.mean(model_entr) - model_entr[0], np.mean(base_entr) - base_entr[0], np.mean(NDCG) - NDCG[0]
  
combos = {
    20: {10: [0, 0.5, 1, 2, 4, 8]}
}

key_list = ['MCAT',
			'MCAT_FC',
			'MCATDeterministic',
			'ModalityDropout',
			'CascadedResidualAutoencoder',
			'UnifiedRepresentationNetwork',
			'PGOS',
			'PPO',
			'Random']

if os.path.exists('synthetic_save/metrics.pkl'):
	metrics = pickle.load(open('synthetic_save/metrics.pkl', 'rb'))
else:
	metrics = {}

for M in combos:
	trials_file = f'synthetic_data/trial_features_M{M}_test.pt'
	labels_file = f'synthetic_data/labels_M{M}_test.pt'
	inv_feat_static_file = f'synthetic_data/inv_features_static_M{M}_test.pt'
	inv_feat_diagnosis_file = f'synthetic_data/inv_features_diagnosis_M{M}_test.pt'
	inv_feat_diagnosis_lens_file = f'synthetic_data/inv_features_diagnosis_lens_M{M}_test.pt'
	inv_feat_prescription_file = f'synthetic_data/inv_features_prescription_M{M}_test.pt'
	inv_feat_prescription_lens_file = f'synthetic_data/inv_features_prescription_lens_M{M}_test.pt'
	inv_feat_enrollment_file = f'synthetic_data/inv_features_history_M{M}_test.pt'
	inv_feat_enrollment_lens_file = f'synthetic_data/inv_features_history_lens_M{M}_test.pt'
	inv_masks_file = f'synthetic_data/inv_masks_M{M}_test.pt'
	eth_labels_file = f'synthetic_data/eth_labels_M{M}_test.pt'

	td = TrialDataset_missing(trials_file, inv_feat_static_file, inv_feat_diagnosis_file, inv_feat_diagnosis_lens_file, inv_feat_prescription_file, inv_feat_prescription_lens_file, inv_feat_enrollment_file, inv_feat_enrollment_lens_file, inv_masks_file, labels_file, eth_labels_file, 10)
	
	test_sampler = SubsetRandomSampler(list(range(len(td))))
	batch_size = 32
	test_dataloader = DataLoader(td, batch_size, sampler=test_sampler)

	trial_dim = len(td[0]['trial'])
	claim_seq_len = 500
	trial_seq_len = 50
	dx_dim = 260
	rx_dim = 100
	lstm_dim = 128
	embedding_dim = 128
	num_layers = 1
	hidden_dim = 64
	static_dim = len(td[0]['inv_static'][0])
	n_heads = 4
	final_layer = 'Linear'

	for K in combos[M]:
		if M > K:
			for lam in combos[M][K]:
				for key in key_list:
					print('\n\n\n')
					print(key, 'M =', M, 'K = ', K, 'lam: ', lam)    

					if (key, M, K, lam) not in metrics:
						if key == 'MCAT':
							net = MCAT(trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads).to(device, non_blocking=True)
						elif key == 'MCAT_FC':
							net = MCAT_FC(trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads).to(device, non_blocking=True)
						elif key == 'MCATDeterministic':
							net = MCAT(trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads).to(device, non_blocking=True)
						elif key == 'ModalityDropout':
							net = ModalityDropout(trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads).to(device, non_blocking=True)
						elif key == 'CascadedResidualAutoencoder':
							net = CascadedResidualAutoencoderNetwork(claim_seq_len, trial_seq_len, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads).to(device, non_blocking=True)
						elif key == 'UnifiedRepresentationNetwork':
							net = UnifiedRepresentationNetwork(claim_seq_len, trial_seq_len, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads).to(device, non_blocking=True)
						elif key == 'PGOS':
							net = MCAT(trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads).to(device, non_blocking=True)
						elif key == 'PPO':
							net = MCAT(trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads).to(device, non_blocking=True)
						elif key == 'Random':
							net = None
						
						if key != 'Random':
							model_path = f'synthetic_save/{key}_M_{M}_K_{K}_lam_{lam}.pt'
							mdl = torch.load(model_path)
							net.load_state_dict(mdl['model_state_dict'])
				
						results_keys = ['rel_err', 'precision', 'recall', 'model_entr', 'base_entr',  'NDCG', 'model_rep', 'base_rep', 'rel_err_ci', 'precision_ci', 'recall_ci', 'model_entr_ci', 'base_entr_ci',  'NDCG_ci']
						perf = compute_test_metrics(net, K, M)	
						results = {}
						results.update(zip(results_keys, perf))
						metrics[(key, M, K, lam)] = results

					print('Mean relative error: ', metrics[(key, M, K, lam)]['rel_err'])
					print('Mean precision: ', metrics[(key, M, K, lam)]['precision'])
					print('Mean recall: ', metrics[(key, M, K, lam)]['recall'])
					print('Mean model entropy: ', metrics[(key, M, K, lam)]['model_entr'])
					print('Mean base entropy: ', metrics[(key, M, K, lam)]['base_entr'])
					print('Mean NDCG: ', np.mean((metrics[(key, M, K, lam)]['NDCG'])))
					print('Mean model repn: ', metrics[(key, M, K, lam)]['model_rep'])
					print('Mean base repn: ', metrics[(key, M, K, lam)]['base_rep'])

pickle.dump(metrics, open('synthetic_save/metrics.pkl', 'wb'))