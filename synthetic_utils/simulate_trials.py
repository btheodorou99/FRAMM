import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from models import DataLabeler 
import torch.nn.functional as F

RANDOM_SEED = 4
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

trial_pool = torch.load('../synthetic_data/trial_pool.pt')
trial_pool = trial_pool[torch.randperm(len(trial_pool))]
inv_pool = pickle.load(open('../synthetic_data/inv_pool.pkl', 'rb'))
inv_history = {}
inv_history_len = {}

trial_dim = trial_pool.shape[1]
dx_dim = 260
rx_dim = 100
lstm_dim = 128
embedding_dim = 128
num_layers = 1
hidden_dim = 64
static_dim = len(inv_pool[0][0])
hist_dim = 292
device = 'cuda' if torch.cuda.is_available() else 'cpu'
labeler = DataLabeler(trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim).to(device, non_blocking=True)
mdl = torch.load('../synthetic_save/DataLabeler.pt')
labeler.load_state_dict(mdl['model_state_dict'])

trial_features_train = []
inv_features_static_train = []
inv_features_diagnosis_train = []
inv_features_diagnosis_lens_train = []
inv_features_prescription_train = []
inv_features_prescription_lens_train = []
inv_features_history_train = []
inv_features_history_lens_train = []
eth_labels_train = []
labels_train = []

trial_features_test = []
inv_features_static_test = []
inv_features_diagnosis_test = []
inv_features_diagnosis_lens_test = []
inv_features_prescription_test = []
inv_features_prescription_lens_test = []
inv_features_history_test = []
inv_features_history_lens_test = []
eth_labels_test = []
labels_test = []

NUM_MODALITIES = 4
NUM_AUGMENTATIONS = 10
TEST_SPLIT = int(0.2 * trial_pool.shape[0])
for M in [20]:
    for i, t in enumerate(trial_pool):
        if i%10 == 0:
            print(i)

        t_for_hist = t[:-1536]
        trial_input = t.unsqueeze(0).to(device)
        inv_choices = random.sample(list(inv_pool.keys()), M)
        static_input = torch.stack([inv_pool[idx][0] for idx in inv_choices], dim=0).unsqueeze(0).to(device)
        dx = torch.stack([inv_pool[idx][1] for idx in inv_choices], dim=0).unsqueeze(0)
        dx_input = F.one_hot(dx.type(torch.int64), num_classes=260).to(device)
        dx_len_input = torch.tensor([500] * M).unsqueeze(0).cpu().type(torch.int64)
        rx = torch.stack([inv_pool[idx][2] for idx in inv_choices], dim=0).unsqueeze(0)
        rx_input = F.one_hot(rx.type(torch.int64), num_classes=100).to(device)
        rx_len_input = torch.tensor([500] * M).unsqueeze(0).cpu().type(torch.int64)
        hist_input = torch.stack([inv_history[idx] if idx in inv_history else torch.zeros(50, hist_dim) for idx in inv_choices], dim=0).unsqueeze(0).to(device)
        hist_len_input = torch.tensor([inv_history_len[idx] if idx in inv_history else 0 for idx in inv_choices]).unsqueeze(0).cpu().type(torch.int64)

        eth_labels = static_input[:,:,-6:].squeeze().detach().cpu()
        labels_output = torch.ceil(labeler(trial_input, static_input, dx_input, dx_len_input, rx_input, rx_len_input, hist_input, hist_len_input, M).squeeze(-1)).detach().cpu()

        for j, idx in enumerate(inv_choices):
            hist_addition = torch.cat((t_for_hist, labels_output[:,j]), dim=-1)
            if idx in inv_history:
                if inv_history_len[idx] == 50:
                    inv_history[idx] = torch.cat((inv_history[idx][0:49], hist_addition), dim=0)
                else:
                    inv_history[idx][inv_history_len[idx]] = hist_addition
                    inv_history_len[idx] = inv_history_len[idx] + 1
            else:
                new_hist = torch.zeros(50, hist_dim)
                new_hist[0] = hist_addition
                inv_history[idx] = new_hist
                inv_history_len[idx] = 1

        if i > TEST_SPLIT:
            trial_features_train.append(t)
            inv_features_static_train.append(static_input.squeeze().detach().cpu())
            inv_features_diagnosis_train.append(dx.squeeze().detach().cpu())
            inv_features_diagnosis_lens_train.append(dx_len_input.squeeze().detach().cpu())
            inv_features_prescription_train.append(rx.squeeze().detach().cpu())
            inv_features_prescription_lens_train.append(rx_len_input.squeeze().detach().cpu())
            inv_features_history_train.append(hist_input.squeeze().detach().cpu())
            inv_features_history_lens_train.append(hist_len_input.squeeze().detach().cpu())
            eth_labels_train.append(eth_labels)
            labels_train.append(labels_output.squeeze())
        else:
            trial_features_test.append(t)
            inv_features_static_test.append(static_input.squeeze().detach().cpu())
            inv_features_diagnosis_test.append(dx.squeeze().detach().cpu())
            inv_features_diagnosis_lens_test.append(dx_len_input.squeeze().detach().cpu())
            inv_features_prescription_test.append(rx.squeeze().detach().cpu())
            inv_features_prescription_lens_test.append(rx_len_input.squeeze().detach().cpu())
            inv_features_history_test.append(hist_input.squeeze().detach().cpu())
            inv_features_history_lens_test.append(hist_len_input.squeeze().detach().cpu())
            eth_labels_test.append(eth_labels)
            labels_test.append(labels_output)

    # Save full data files
    trial_feat_filename_train = f'../synthetic_data/trial_features_M{M}_train.pt'
    labels_filename_train = f'../synthetic_data/labels_M{M}_train.pt'
    inv_feat_static_filename_train = f'../synthetic_data/inv_features_static_M{M}_train.pt'
    inv_feat_diagnosis_filename_train = f'../synthetic_data/inv_features_diagnosis_M{M}_train.pt'
    inv_feat_diagnosis_lens_filename_train = f'../synthetic_data/inv_features_diagnosis_lens_M{M}_train.pt'
    inv_feat_prescription_filename_train = f'../synthetic_data/inv_features_prescription_M{M}_train.pt'
    inv_feat_prescription_lens_filename_train = f'../synthetic_data/inv_features_prescription_lens_M{M}_train.pt'
    inv_feat_history_filename_train = f'../synthetic_data/inv_features_history_M{M}_train.pt'
    inv_feat_history_lens_filename_train = f'../synthetic_data/inv_features_history_lens_M{M}_train.pt'
    inv_masks_filename_train_full = f'../synthetic_data/inv_masks_M{M}_train_full.pt'
    eth_labels_filename_train = f'../synthetic_data/eth_labels_M{M}_train.pt'

    torch.save(torch.stack(trial_features_train), trial_feat_filename_train)
    torch.save(torch.stack(labels_train), labels_filename_train)
    torch.save(torch.stack(inv_features_static_train), inv_feat_static_filename_train)
    torch.save(torch.stack(inv_features_diagnosis_lens_train), inv_feat_diagnosis_lens_filename_train)
    torch.save(torch.stack(inv_features_prescription_lens_train), inv_feat_prescription_lens_filename_train)
    torch.save(torch.stack(inv_features_history_lens_train), inv_feat_history_lens_filename_train)
    torch.save(torch.stack(inv_features_diagnosis_train), inv_feat_diagnosis_filename_train)
    torch.save(torch.stack(inv_features_prescription_train), inv_feat_prescription_filename_train)
    torch.save(torch.stack(inv_features_history_train), inv_feat_history_filename_train)
    torch.save(torch.zeros(len(labels_train), M, NUM_MODALITIES), inv_masks_filename_train_full)
    torch.save(torch.stack(eth_labels_train), eth_labels_filename_train)

    trial_feat_filename_test = f'../synthetic_data/trial_features_M{M}_test.pt'
    labels_filename_test = f'../synthetic_data/labels_M{M}_test.pt'
    inv_feat_static_filename_test = f'../synthetic_data/inv_features_static_M{M}_test.pt'
    inv_feat_diagnosis_filename_test = f'../synthetic_data/inv_features_diagnosis_M{M}_test.pt'
    inv_feat_diagnosis_lens_filename_test = f'../synthetic_data/inv_features_diagnosis_lens_M{M}_test.pt'
    inv_feat_prescription_filename_test = f'../synthetic_data/inv_features_prescription_M{M}_test.pt'
    inv_feat_prescription_lens_filename_test = f'../synthetic_data/inv_features_prescription_lens_M{M}_test.pt'
    inv_feat_history_filename_test = f'../synthetic_data/inv_features_history_M{M}_test.pt'
    inv_feat_history_lens_filename_test = f'../synthetic_data/inv_features_history_lens_M{M}_test.pt'
    inv_masks_filename_test_full = f'../synthetic_data/inv_masks_M{M}_test_full.pt'
    eth_labels_filename_test = f'../synthetic_data/eth_labels_M{M}_test.pt'

    torch.save(torch.stack(trial_features_test), trial_feat_filename_test)
    torch.save(torch.stack(labels_test), labels_filename_test)
    torch.save(torch.stack(inv_features_static_test), inv_feat_static_filename_test)
    torch.save(torch.stack(inv_features_diagnosis_lens_test), inv_feat_diagnosis_lens_filename_test)
    torch.save(torch.stack(inv_features_prescription_lens_test), inv_feat_prescription_lens_filename_test)
    torch.save(torch.stack(inv_features_history_lens_test), inv_feat_history_lens_filename_test)
    torch.save(torch.stack(inv_features_diagnosis_test), inv_feat_diagnosis_filename_test)
    torch.save(torch.stack(inv_features_prescription_test), inv_feat_prescription_filename_test)
    torch.save(torch.stack(inv_features_history_test), inv_feat_history_filename_test)
    torch.save(torch.zeros(len(labels_test), M, NUM_MODALITIES), inv_masks_filename_test_full)
    torch.save(torch.stack(eth_labels_test), eth_labels_filename_test)

    # Save missing data masks
    inv_masks_train = torch.randint(0,5, (len(labels_train), NUM_AUGMENTATIONS, M, NUM_MODALITIES))
    inv_masks_train = torch.floor_divide(inv_masks_train, 4) # 4/5 chance of being present
    for i in range(inv_masks_train.shape[0]):
        for j in range(NUM_AUGMENTATIONS):
            for k in range(M):
                if torch.sum(inv_masks_train[i,j,k,:]) == NUM_MODALITIES:
                    modality = random.randint(0, NUM_MODALITIES-1)
                    inv_masks_train[i,j,k,modality] = 0
    inv_masks_train = torch.reshape(inv_masks_train, (len(labels_train) * NUM_AUGMENTATIONS, M, NUM_MODALITIES))

    inv_masks_test = torch.randint(0,5, (len(labels_test), NUM_AUGMENTATIONS, M, NUM_MODALITIES))
    inv_masks_test = torch.floor_divide(inv_masks_test, 4) # 4/5 chance of being present
    for i in range(inv_masks_test.shape[0]):
        for j in range(NUM_AUGMENTATIONS):
            for k in range(M):
                if torch.sum(inv_masks_test[i,j,k,:]) == NUM_MODALITIES:
                    modality = random.randint(0, NUM_MODALITIES-1)
                    inv_masks_test[i,j,k,modality] = 0
    inv_masks_test = torch.reshape(inv_masks_test, (len(labels_test) * NUM_AUGMENTATIONS, M, NUM_MODALITIES))

    inv_masks_filename_train = f'../synthetic_data/inv_masks_M{M}_train.pt'
    torch.save(inv_masks_train, inv_masks_filename_train)

    inv_masks_filename_test = f'../synthetic_data/inv_masks_M{M}_test.pt'
    torch.save(inv_masks_test, inv_masks_filename_test)