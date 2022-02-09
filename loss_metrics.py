import math
import torch
from models import *
import torch.nn.functional as F
from torch.distributions import Categorical

#######
### Loss Functions Components
#######
# ordered_score = (batch, numMC, M)
def compute_log_prob(rank_samples, score, num_MC, K):
	ordered_score = torch.gather(score.unsqueeze(1).repeat(1,num_MC,1), 2, rank_samples).to(device)
	first_k = ordered_score[:,:,:K]
	permuted_indices = torch.argsort(torch.rand(*first_k.shape), dim=-1).to(device)
	permuted_first_k = torch.gather(first_k, -1, permuted_indices)
	random_top_k_ordered_score = torch.cat((permuted_first_k, ordered_score[:,:,K:]), dim=-1)
	denominators = torch.flip(torch.cumsum(torch.flip(random_top_k_ordered_score, [-1]), -1), [-1])
	ranking_prob = torch.prod((random_top_k_ordered_score / denominators)[:,:,:K], -1)	
	prob = math.factorial(K) * ranking_prob
	return torch.log(prob).to(device)

def compute_prob(rank_samples, score, num_MC, K):
	ordered_score = torch.gather(score.unsqueeze(1).repeat(1,num_MC,1), 2, rank_samples).to(device)
	first_k = ordered_score[:,:,:K]
	permuted_indices = torch.argsort(torch.rand(*first_k.shape), dim=-1).to(device)
	permuted_first_k = torch.gather(first_k, -1, permuted_indices)
	random_top_k_ordered_score = torch.cat((permuted_first_k, ordered_score[:,:,K:]), dim=-1)
	denominators = torch.flip(torch.cumsum(torch.flip(random_top_k_ordered_score, [-1]), -1), [-1])
	ranking_prob = torch.prod((random_top_k_ordered_score / denominators)[:,:,:K], -1)	
	prob = math.factorial(K) * ranking_prob
	return prob.to(device)

def ranking_loss(rank_samples, relevance, num_MC, K):
	relevance = relevance/torch.sum(relevance, 1, keepdim=True)
	relevance[relevance != relevance] = 0 # Handle division by zero and set to zero
	relevance = torch.gather(relevance.unsqueeze(1).repeat(1,num_MC,1), 2, rank_samples).to(device)
	return torch.sum(relevance[:,:,:K], 2).to(device) - torch.sum(relevance[:,:,K:], 2).to(device)

def fairness_loss(rank_samples, eth_list, relevance, num_MC, K):
	relevance = torch.gather(relevance.unsqueeze(1).repeat(1,num_MC,1), 2, rank_samples).to(device)[:,:,:K]
	relevance = relevance/torch.sum(relevance, 2, keepdim=True) # Normalize the relevance into weights
	relevance[relevance != relevance] = 1 / K # Handle division by zero and set to even weighting
	eth_list = torch.gather(eth_list.unsqueeze(1).repeat(1,num_MC,1,1), 2, rank_samples.unsqueeze(-1).repeat(1,1,1,eth_list.size(-1)))[:,:,:K,:]
	eth_list = eth_list * relevance.unsqueeze(-1) # Weight the distributions
	delta_f = torch.sum(eth_list, dim=2) # Sum across the chosen K
	reward = Categorical(probs = delta_f).entropy()
	return reward

def fairness_loss_os(rank_samples, eth_list, eth_ind, relevance, num_MC, K):
	relevance = torch.gather(relevance.unsqueeze(1).repeat(1,num_MC,1), 2, rank_samples).to(device)[:,:,:K]
	relevance = relevance/torch.sum(relevance, 2, keepdim=True) / 100 # Normalize the relevance into weights
	relevance[relevance != relevance] = 1 / K # Handle division by zero and set to even weighting
	eth_list = torch.gather(eth_list.unsqueeze(1).repeat(1,num_MC,1,1), 2, rank_samples.unsqueeze(-1).repeat(1,1,1,eth_list.size(-1)))[:,:,:K,:]
	eth_list = eth_list * relevance.unsqueeze(-1) # Weight the distributions					
	delta_f = torch.sum(eth_list, dim=2)
	eth_ind = eth_ind.unsqueeze(1).repeat(1,num_MC,1)
	ls = torch.zeros((rank_samples.size(0), num_MC)).to(device)
	for k in range(eth_list.size(-1)):
		for j in range(k+1, eth_list.size(-1)):
			ls += F.relu(torch.gather(delta_f, 2, eth_ind[:,:,k].unsqueeze(-1)) - torch.gather(delta_f, 2, eth_ind[:,:,j].unsqueeze(-1))).squeeze(-1)
	return -1*ls

#######
### Complete Loss Functions
#######

# score = bs * M
# relevance = bs * M
# eth_list = bs * M * num_cat
# rank_samples = bs * MC * M
def loss_function(score, relevance, eth_list, final_layer, lam, K):	
	num_MC_samples = 25
	M = len(score[0])
	if final_layer != 'Softmax':
		score = F.softmax(score, dim=1)

	rank_samples = torch.stack([torch.multinomial(score, num_samples=M, replacement=False).to(device) for _ in range(num_MC_samples)], axis=1)

	importance_prob_values = compute_log_prob(rank_samples, score, num_MC_samples, K)
	delta_values = ranking_loss(rank_samples, relevance, num_MC_samples, K)
	delta_f_values = fairness_loss(rank_samples, eth_list, relevance, num_MC_samples, K)
	rewards = delta_values + lam*delta_f_values
	loss = -1 * torch.sum(importance_prob_values * rewards, 1)
	return loss/num_MC_samples, torch.mean(rewards), torch.mean(delta_values), torch.mean(delta_f_values)

def loss_function_deterministic(score, relevance, eth_list, final_layer, lam, K):	
	if final_layer != 'Softmax':
		score = F.softmax(score, dim=1)

	model_ind = torch.argsort(score, axis=1, descending=True).unsqueeze(1)
	delta_values = ranking_loss(model_ind, relevance, 1, K)
	delta_f_values = fairness_loss(model_ind, eth_list, relevance, 1, K)
	rewards = delta_values + lam*delta_f_values
	loss = -1 * torch.sum(rewards, 1)
	return loss, torch.mean(rewards), torch.mean(delta_values), torch.mean(delta_f_values)

def loss_function_os(score, relevance, eth_list, eth_ind, final_layer, lam, K):	
	num_MC_samples = 25
	M = len(score[0])
	if final_layer != 'Softmax':
		score = F.softmax(score, dim=1)

	rank_samples = torch.stack([torch.multinomial(score, num_samples=M, replacement=False).to(device) for _ in range(num_MC_samples)], axis=1)

	importance_prob_values = compute_log_prob(rank_samples, score, num_MC_samples, K)
	delta_values = ranking_loss(rank_samples, relevance, num_MC_samples, K)
	delta_f_values = fairness_loss_os(rank_samples, eth_list, eth_ind, relevance, num_MC_samples, K)
	rewards = delta_values + lam*delta_f_values
	loss = -1 * torch.sum(importance_prob_values * rewards, 1)
	return loss/num_MC_samples, torch.mean(rewards), torch.mean(delta_values), torch.mean(delta_f_values)

def loss_function_PPO(score, score_old, relevance, eth_list, final_layer, lam, K):	
	num_MC_samples = 25
	epsilon = 0.2
	M = len(score[0])
	if final_layer != 'Softmax':
		score = F.softmax(score, dim=1)

	rank_samples = torch.stack([torch.multinomial(score, num_samples=M, replacement=False).to(device) for _ in range(num_MC_samples)], axis=1)

	importance_prob_values = compute_prob(rank_samples, score, num_MC_samples, K)
	importance_prob_values_old = compute_prob(rank_samples, score_old, num_MC_samples, K)
	prob_proportion = importance_prob_values / importance_prob_values_old
	clipped_prob_proportion = torch.clip(prob_proportion, min = 1 - epsilon, max = 1 + epsilon)
	delta_values = ranking_loss(rank_samples, relevance, num_MC_samples, K)
	delta_f_values = fairness_loss(rank_samples, eth_list, relevance, num_MC_samples, K)
	rewards = delta_values + lam*delta_f_values
	original_objective = prob_proportion * rewards
	clipped_objective = clipped_prob_proportion * rewards
	min_objective = torch.minimum(original_objective, clipped_objective)
	loss = -1 * torch.sum(min_objective, 1)
	return loss/num_MC_samples, torch.mean(rewards)

def loss_function_fairness(score, relevance, eth_list, final_layer, lam, K):
	num_MC_samples = 25
	M = len(score[0])
	if final_layer != 'Softmax':
		score = F.softmax(score, dim=1)

	rank_samples = torch.stack([torch.multinomial(score,num_samples=M, replacement=False).to(device) for _ in range(num_MC_samples)], axis=1)
	
	importance_prob_values = compute_log_prob(rank_samples, score, num_MC_samples, K)
	delta_f_values = fairness_loss(rank_samples, eth_list, relevance, num_MC_samples, K)

	loss = -1 * torch.sum(importance_prob_values * (lam*delta_f_values), 1)
	return loss/num_MC_samples
	
#######
### Validation Metrics
#######

def util_validation(score, relevance, final_layer):
	num_MC_samples = 25
	K = 10
	lam  = 0
	M = len(score[0])
		
	if final_layer != 'Softmax':
		score = torch.exp(score).to(device)
		score = score / torch.sum(score, dim=1).unsqueeze(1)

	rank_samples = torch.stack([torch.multinomial(score,num_samples=M, replacement=False).to(device) for _ in range(num_MC_samples)], axis=1)
	
	delta_values = ranking_loss(rank_samples, relevance, num_MC_samples, K)
				
	return torch.mean(delta_values, 1)
	
def grp_rep_validation(score, eth_list, final_layer, relevance=None):
	num_MC_samples = 25
	K = 10
	lam  = 0
	num_groups = 6
	M = len(score[0])
		
	if final_layer != 'Softmax':
		score = torch.exp(score).to(device)
		score = score / torch.sum(score, dim=1).unsqueeze(1)

	rank_samples = torch.stack([torch.multinomial(score,num_samples=M, replacement=False).to(device) for _ in range(num_MC_samples)], axis=1)

	if relevance is not None:
		relevance = torch.gather(relevance.unsqueeze(1).repeat(1,num_MC_samples,1), 2, rank_samples).to(device)[:,:,:K]
		relevance = relevance/torch.sum(relevance, 2, keepdim=True) # Normalize the relevance into weights
		relevance[relevance != relevance] = 1 / K # Handle division by zero and set to even weighting
	else:
		relevance = torch.ones((len(score), num_MC_samples, K)) / K
	eth_list = torch.gather(eth_list.unsqueeze(1).repeat(1,num_MC_samples,1,1), 2, rank_samples.unsqueeze(-1).repeat(1,1,1,eth_list.size(-1)))[:,:,:K,:]
	eth_list = eth_list * relevance.unsqueeze(-1) # Weight the distributions
	delta_f = torch.sum(eth_list,dim=2) # Sum across the chosen K
	grp_rep = torch.sum(delta_f, dim=1) / num_MC_samples # Sum across the MC_samples
	
	return grp_rep.detach().numpy()