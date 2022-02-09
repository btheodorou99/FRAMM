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
def validate(val_dataloader, net, device, M):
    net.eval()
    losses = []
    for batch in val_dataloader:
        trial = batch['trial'].to(device)
        label = batch['label'].to(device)
        inv_static = batch['inv_static'].to(device)
        inv_dx = F.one_hot(batch['inv_dx'].type(torch.int64), num_classes=260).to(device)
        inv_dx_len = batch['inv_dx_len'].cpu().type(torch.int64)
        inv_rx = F.one_hot(batch['inv_rx'].type(torch.int64), num_classes=100).to(device)
        inv_rx_len = batch['inv_rx_len'].cpu().type(torch.int64)
        inv_enroll = batch['inv_enroll'].to(device)
        inv_enroll_len = batch['inv_enroll_len'].cpu().type(torch.int64)
            
        if torch.sum(label) == 0:
            continue

        with torch.no_grad():
            enrollments = net(trial, inv_static, inv_dx, inv_dx_len, inv_rx, inv_rx_len, inv_enroll, inv_enroll_len, M).squeeze(-1)

        if not torch.isfinite(enrollments).all():
            continue
            
        loss_fn = nn.MSELoss()
        loss = loss_fn(label, enrollments)
        losses.append(loss.cpu().item())

    return np.mean(losses)
  
def train_labeler(num_epochs, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, M, train_dataloader, val_dataloader, lr):
    net = DataLabeler(trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim).to(device, non_blocking=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    min_val_loss = 1e5
    PATH = 'save/DataLabeler.pt'

    loss_values = []
    val_loss = []
    for epoch in tqdm(range(num_epochs)):
        net.train()
        running_loss = []
        for batch in train_dataloader:
            trial = batch['trial'].to(device)
            label = batch['label'].to(device)
            inv_static = batch['inv_static'].to(device)
            inv_dx = F.one_hot(batch['inv_dx'].type(torch.int64), num_classes=260).to(device)
            inv_dx_len = batch['inv_dx_len'].cpu().type(torch.int64)
            inv_rx = F.one_hot(batch['inv_rx'].type(torch.int64), num_classes=100).to(device)
            inv_rx_len = batch['inv_rx_len'].cpu().type(torch.int64)
            inv_enroll = batch['inv_enroll'].to(device)
            inv_enroll_len = batch['inv_enroll_len'].cpu().type(torch.int64)

            if torch.count_nonzero(label) == 0:
                continue
  
            enrollments = net(trial, inv_static, inv_dx, inv_dx_len, inv_rx, inv_rx_len, inv_enroll, inv_enroll_len, M).squeeze(-1)
            loss_fn = nn.MSELoss()
            loss = loss_fn(label, enrollments)

            if loss.requires_grad == False:
                print('Loss grad not present')
            else:
                running_loss.append(loss.cpu().detach().item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print('Epoch: ', epoch, 'Training Loss: ', np.mean(running_loss))
        
        loss_values.append(np.mean(running_loss))

        with torch.no_grad():
            val_l = validate(val_dataloader, net, device, M)
            val_loss.append(val_l)
            if val_l < min_val_loss:
                min_val_loss = val_l
                print("Saving New Model")
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_values': loss_values,
                    'utility_values_validation': val_loss
                }, PATH)

            print('Epoch: ', epoch, 'Validation Loss: ', val_l)

M = 10

trials_file = f'data/trial_features_M{M}_train.pt'
labels_file = f'data/labels_M{M}_train.pt'
inv_feat_static_file = f'data/inv_features_static_M{M}_train.pt'
inv_feat_diagnosis_file = f'data/inv_features_diagnosis_M{M}_train.pt'
inv_feat_diagnosis_lens_file = f'data/inv_features_diagnosis_lens_M{M}_train.pt'
inv_feat_prescription_file = f'data/inv_features_prescription_M{M}_train.pt'
inv_feat_prescription_lens_file = f'data/inv_features_prescription_lens_M{M}_train.pt'
inv_feat_enrollment_file = f'data/inv_features_history_M{M}_train.pt'
inv_feat_enrollment_lens_file = f'data/inv_features_history_lens_M{M}_train.pt'
inv_masks_file = f'data/inv_masks_M{M}_train_full.pt'
eth_labels_file = f'data/eth_labels_M{M}_train.pt'

td = TrialDataset(trials_file, inv_feat_static_file, inv_feat_diagnosis_file, inv_feat_diagnosis_lens_file, inv_feat_prescription_file, inv_feat_prescription_lens_file, inv_feat_enrollment_file, inv_feat_enrollment_lens_file, inv_masks_file, labels_file, eth_labels_file)
td.labels[td.labels != td.labels] = 0  # correct some nans
td.labels[td.labels > 50] = 50  # remove outliers that we don't want in synthetic data

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
batch_size = 64
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

num_epochs = 150
lr = 1e-4

train_labeler(num_epochs, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, M, train_dataloader, val_dataloader, lr)