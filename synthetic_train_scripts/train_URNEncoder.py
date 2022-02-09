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
def validate_auto(val_dataloader, net, device, num_inv):
    net.eval()
    losses = []
    for batch in val_dataloader:
        inv_static = batch['inv_static'].to(device)
        inv_dx = F.one_hot(batch['inv_dx'].type(torch.int64), num_classes=260).to(device)
        inv_dx_len = batch['inv_dx_len'].cpu().type(torch.int64)
        inv_rx = F.one_hot(batch['inv_rx'].type(torch.int64), num_classes=100).to(device)
        inv_rx_len = batch['inv_rx_len'].cpu().type(torch.int64)
        inv_enroll = batch['inv_enroll'].to(device)
        inv_enroll_len = batch['inv_enroll_len'].unsqueeze(-1).cpu().type(torch.int64)
        inv_mask = batch['inv_mask'].to(device)

        with torch.no_grad():
            loss = net.autoencode(inv_static, inv_dx, inv_dx_len, inv_rx, inv_rx_len, inv_enroll, inv_enroll_len, inv_mask, num_inv)

        losses.append(loss.cpu().numpy())
    return np.nanmean(losses)

def train_policy(num_epochs, claim_seq_len, trial_seq_len, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads, M, train_dataloader, val_dataloader, lr):
    net = UnifiedRepresentationNetwork(claim_seq_len, trial_seq_len, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads).to(device, non_blocking=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    min_val_loss = 1e5
    PATH = 'synthetic_save/UnifiedRepresentationNetwork_Auto.pt'

    loss_values_auto = []
    val_loss_auto = []
    for epoch in tqdm(range(num_epochs)):
        net.train()
        running_loss_auto = []
        for batch in train_dataloader:
            inv_static = batch['inv_static'].to(device)
            inv_dx = F.one_hot(batch['inv_dx'].type(torch.int64), num_classes=260).to(device)
            inv_dx_len = batch['inv_dx_len'].cpu().type(torch.int64)
            inv_rx = F.one_hot(batch['inv_rx'].type(torch.int64), num_classes=100).to(device)
            inv_rx_len = batch['inv_rx_len'].cpu().type(torch.int64)
            inv_enroll = batch['inv_enroll'].to(device)
            inv_enroll_len = batch['inv_enroll_len'].unsqueeze(-1).cpu().type(torch.int64)
            inv_mask = batch['inv_mask'].to(device)

            loss = net.autoencode(inv_static, inv_dx, inv_dx_len, inv_rx, inv_rx_len, inv_enroll, inv_enroll_len, inv_mask, M)

            if loss.requires_grad == False:
                print('Loss grad not present')
            else:
                running_loss_auto.append(loss.cpu().detach().item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print('Auto Epoch: ', epoch, 'Auto Training loss: ', np.nanmean(running_loss_auto))
        loss_values_auto.append(np.nanmean(running_loss_auto))
        
        if epoch % 5 == 4:
            with torch.no_grad():
                val_l = validate_auto(val_dataloader, net, device, M)
                val_loss_auto.append(val_l)
                if val_l < min_val_loss:
                    min_val_loss = val_l
                    print("Saving New Model")
                    torch.save({
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, PATH)
                print('Auto Epoch: ', epoch, 'Auto Validation loss: ', val_l)

M = 20
trials_file = f'synthetic_data/trial_features_M20_train.pt'
labels_file = f'synthetic_data/labels_M20_train.pt'
inv_feat_static_file = f'synthetic_data/inv_features_static_M20_train.pt'
inv_feat_diagnosis_file = f'synthetic_data/inv_features_diagnosis_M20_train.pt'
inv_feat_diagnosis_lens_file = f'synthetic_data/inv_features_diagnosis_lens_M20_train.pt'
inv_feat_prescription_file = f'synthetic_data/inv_features_prescription_M20_train.pt'
inv_feat_prescription_lens_file = f'synthetic_data/inv_features_prescription_lens_M20_train.pt'
inv_feat_enrollment_file = f'synthetic_data/inv_features_history_M20_train.pt'
inv_feat_enrollment_lens_file = f'synthetic_data/inv_features_history_lens_M20_train.pt'
inv_masks_file = f'synthetic_data/inv_masks_M20_train.pt'
eth_labels_file = f'synthetic_data/eth_labels_M20_train.pt'

td = TrialDataset_missing(trials_file, inv_feat_static_file, inv_feat_diagnosis_file, inv_feat_diagnosis_lens_file, inv_feat_prescription_file, inv_feat_prescription_lens_file, inv_feat_enrollment_file, inv_feat_enrollment_lens_file, inv_masks_file, labels_file, eth_labels_file, 10)
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
batch_size = 16
train_dataloader = DataLoader(td, batch_size, sampler=train_sampler)
val_dataloader = DataLoader(td, batch_size, sampler=val_sampler)

claim_seq_len = 500
trial_seq_len = 50
trial_dim = len(td[0]['trial'])
dx_dim = 260
rx_dim = 100
lstm_dim = 128
embedding_dim = 128
num_layers = 1
hidden_dim = 64
static_dim = len(td[0]['inv_static'][0])
n_heads = 4

num_epochs = 50
lr = 1e-5
final_layer = 'Linear'

train_policy(num_epochs, claim_seq_len, trial_seq_len, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads, M, train_dataloader, val_dataloader, lr)