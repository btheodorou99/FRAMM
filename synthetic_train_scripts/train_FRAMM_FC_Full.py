import torch
import numpy as np
from models import *
from tqdm import tqdm
from data_object import *
from loss_metrics import *
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# train functions
def validate(val_dataloader, net, K, device, lam, final_layer, num_inv):
    net.eval()
    rel_dif = []
    num_overlap = []
    num_tot = []
    rewards = []
    for batch in val_dataloader:
        trial = batch['trial'].to(device)
        label = batch['label'].to(device)
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
            score = net(trial, inv_static, inv_dx, inv_dx_len, inv_rx, inv_rx_len, inv_enroll, inv_enroll_len, M).squeeze(-1)

        if not torch.isfinite(score).all():
            continue
            
        avg_delta = torch.sum(torch.gather(label, 1, torch.argsort(score, descending=True)[:,:K]), 1)
        max_delta = torch.sum(torch.sort(label, descending=True)[0][:, :K], 1)
        rel_dif.extend(((max_delta - avg_delta) / max_delta).tolist())

        model_ind = torch.argsort(score, descending=True)[:,:K].detach().cpu().tolist()
        gt_ind = torch.argsort(label, descending=True)[:,:K].detach().cpu().tolist()
        num_overlap.extend([len(set(m_idx) & set(g_idx)) for m_idx, g_idx in zip(model_ind, gt_ind)])
        num_tot.extend([len(gt) for gt in gt_ind])
        _, reward, _, _, = loss_function(score.to(device, non_blocking=True), label, eth_label, final_layer, lam, K)
        rewards.append(np.nanmean(reward.cpu().numpy()))
    return np.nanmean(rewards), np.nanmean(rel_dif)#, sum(num_overlap)/sum(num_tot)
  
def train_policy(num_epochs, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads, M, train_dataloader, val_dataloader, K, final_layer, lr, lam):
    net = MCATFullData(trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads).to(device, non_blocking=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    max_val_reward = -1e5
    PATH = 'synthetic_save/FRAMM_FC_Full' + '_M_' + str(M) + '_K_' + str(K) + '_lam_' + str(lam) + '.pt'

    utility_values = []
    fairness_values = []
    reward_values = []
    loss_values = []
    val_loss = []
    for epoch in tqdm(range(num_epochs)):
        net.train()
        running_loss = []
        running_reward = []
        running_util = []
        running_fair = []
        for batch in train_dataloader:
            trial = batch['trial'].to(device)
            label = batch['label'].to(device)
            eth_label = batch['eth_label'].to(device)
            inv_static = batch['inv_static'].to(device)
            inv_dx = F.one_hot(batch['inv_dx'].type(torch.int64), num_classes=260).to(device)
            inv_dx_len = batch['inv_dx_len'].cpu().type(torch.int64)
            inv_rx = F.one_hot(batch['inv_rx'].type(torch.int64), num_classes=100).to(device)
            inv_rx_len = batch['inv_rx_len'].cpu().type(torch.int64)
            inv_enroll = batch['inv_enroll'].to(device)
            inv_enroll_len = batch['inv_enroll_len'].cpu().type(torch.int64)
            inv_mask = batch['inv_mask'].to(device)

            if torch.count_nonzero(label) == 0:
                continue
  
            score = net(trial, inv_static, inv_dx, inv_dx_len, inv_rx, inv_rx_len, inv_enroll, inv_enroll_len, M).squeeze(-1)
            if not torch.isfinite(score).all():
                continue
    
            loss, reward, util, fair = loss_function(score.to(device, non_blocking=True), label, eth_label, final_layer, lam, K)

            loss = torch.mean(loss)
            reward = torch.mean(reward)
            util = torch.mean(util)
            fair = torch.mean(fair)

            if loss.requires_grad == False:
                print('Loss grad not present')
            else:
                running_loss.append(loss.cpu().detach().item())
                running_reward.append(reward.cpu().detach().item())
                running_util.append(util.cpu().detach().item())
                running_fair.append(fair.cpu().detach().item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epoch % 5 == 0:
            print('Epoch: ', epoch, 'Training reward: ', np.mean(running_reward), 'Training utility: ', np.mean(running_util), 'Training fairness: ', np.mean(running_fair))
        
        utility_values.append(np.mean(running_util))
        fairness_values.append(np.mean(running_fair))
        reward_values.append(np.mean(running_reward))
        loss_values.append(np.mean(running_loss))

        with torch.no_grad():
            val_metrics = validate(val_dataloader, net, K, device, lam, final_layer, M)
            val_loss.append(val_metrics[0])
            if val_metrics[0] > max_val_reward:
                max_val_reward = val_metrics[0]
                print("Saving New Model")
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'utility_values': utility_values,
                    'fairness_values': fairness_values,
                    'loss_values': loss_values,
                    'utility_values_validation': val_loss
                }, PATH)

            print('Epoch: ', epoch, 'Validation Reward: ', val_metrics[0], 'Average Relative Difference: ', val_metrics[1])#, 'Average Overlap: ', val_metrics[2])

combos = {
    20: {10: [0, 0.5, 1, 2, 4, 8]}
}

for M in combos:
    trials_file = f'synthetic_data/trial_features_M{M}_train.pt'
    labels_file = f'synthetic_data/labels_M{M}_train.pt'
    inv_feat_static_file = f'synthetic_data/inv_features_static_M{M}_train.pt'
    inv_feat_diagnosis_file = f'synthetic_data/inv_features_diagnosis_M{M}_train.pt'
    inv_feat_diagnosis_lens_file = f'synthetic_data/inv_features_diagnosis_lens_M{M}_train.pt'
    inv_feat_prescription_file = f'synthetic_data/inv_features_prescription_M{M}_train.pt'
    inv_feat_prescription_lens_file = f'synthetic_data/inv_features_prescription_lens_M{M}_train.pt'
    inv_feat_enrollment_file = f'synthetic_data/inv_features_history_M{M}_train.pt'
    inv_feat_enrollment_lens_file = f'synthetic_data/inv_features_history_lens_M{M}_train.pt'
    inv_masks_file = f'synthetic_data/inv_masks_M{M}_train_full.pt'
    eth_labels_file = f'synthetic_data/eth_labels_M{M}_train.pt'

    td = TrialDataset(trials_file, inv_feat_static_file, inv_feat_diagnosis_file, inv_feat_diagnosis_lens_file, inv_feat_prescription_file, inv_feat_prescription_lens_file, inv_feat_enrollment_file, inv_feat_enrollment_lens_file, inv_masks_file, labels_file, eth_labels_file)
    td.labels[td.labels != td.labels] = 0  # correct some nans
    
    dataset_size = len(td)
    indices = list(range(dataset_size))

    validation_split = .1
    shuffle_dataset = True
    random_seed = 4

    split_val = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split_val:], indices[:split_val]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    batch_size = 32
    train_dataloader = DataLoader(td, batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(td, batch_size, sampler=val_sampler)

    trial_dim = len(td[0]['trial'])
    dx_dim = 260
    rx_dim = 100
    lstm_dim = 128
    embedding_dim = 128
    num_layers = 1
    hidden_dim = 64
    static_dim = len(td[0]['inv_static'][0])
    n_heads = 4

    num_epochs = 35
    lr = 1e-5
    final_layer = 'Linear'

    for K in combos[M]:
        for lam in combos[M][K]:
            print('policy ', 'M =', M, 'K = ', K, 'lam: ', lam)
            train_policy(num_epochs, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads, M, train_dataloader, val_dataloader, K, final_layer, lr, lam)