import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class TrialDataset_missing(Dataset):
    def __init__(self, trials_file, investigator_file, investigator_diagnosis_file, investigator_diagnosis_lens_file, investigator_prescription_file, investigator_prescription_lens_file, investigator_enrollment_file, investigator_enrollment_lens_file, investigator_mask_file, labels_file, eth_labels_file, num_augmentations):
        self.trial_features = torch.load(trials_file).repeat_interleave(num_augmentations, dim=0).to('cpu')
        self.labels = torch.load(labels_file).repeat_interleave(num_augmentations, dim=0).to('cpu')
        self.eth_labels = torch.load(eth_labels_file).repeat_interleave(num_augmentations, dim=0).to('cpu')
        self.inv_features = torch.load(investigator_file).repeat_interleave(num_augmentations, dim=0).to('cpu')
        self.inv_diagnosis = torch.load(investigator_diagnosis_file).repeat_interleave(num_augmentations, dim=0).to('cpu')
        self.inv_diagnosis_lens = torch.load(investigator_diagnosis_lens_file).repeat_interleave(num_augmentations, dim=0).to('cpu')
        self.inv_prescription = torch.load(investigator_prescription_file).repeat_interleave(num_augmentations, dim=0).to('cpu')
        self.inv_prescription_lens = torch.load(investigator_prescription_lens_file).repeat_interleave(num_augmentations, dim=0).to('cpu')
        self.inv_enrollment = torch.load(investigator_enrollment_file).repeat_interleave(num_augmentations, dim=0).to('cpu')
        self.inv_enrollment_lens = torch.load(investigator_enrollment_lens_file).repeat_interleave(num_augmentations, dim=0).to('cpu')
        self.inv_masks = torch.load(investigator_mask_file).to('cpu')

    def __len__(self):
        return len(set(self.trial_features))

    def __getitem__(self, idx):
        sample = {"trial": self.trial_features[idx], 
                  "label": self.labels[idx], 
                  "eth_label": self.eth_labels[idx],
                  "inv_static": self.inv_features[idx],
                  "inv_dx": self.inv_diagnosis[idx],
                  "inv_dx_len": self.inv_diagnosis_lens[idx],
                  "inv_rx": self.inv_prescription[idx],
                  "inv_rx_len": self.inv_prescription_lens[idx],
                  "inv_enroll": self.inv_enrollment[idx],
                  "inv_enroll_len": self.inv_enrollment_lens[idx],
                  "inv_mask": self.inv_masks[idx]}
        return sample

class TrialDataset(Dataset):
    def __init__(self, trials_file, investigator_file, investigator_diagnosis_file, investigator_diagnosis_lens_file, investigator_prescription_file, investigator_prescription_lens_file, investigator_enrollment_file, investigator_enrollment_lens_file, investigator_mask_file, labels_file, eth_labels_file):
        self.trial_features = torch.load(trials_file).to('cpu')
        self.labels = torch.load(labels_file).to('cpu')
        self.eth_labels = torch.load(eth_labels_file).to('cpu')
        self.inv_features = torch.load(investigator_file).to('cpu')
        self.inv_diagnosis = torch.load(investigator_diagnosis_file).to('cpu')
        self.inv_diagnosis_lens = torch.load(investigator_diagnosis_lens_file).to('cpu')
        self.inv_prescription = torch.load(investigator_prescription_file).to('cpu')
        self.inv_prescription_lens = torch.load(investigator_prescription_lens_file).to('cpu')
        self.inv_enrollment = torch.load(investigator_enrollment_file).to('cpu')
        self.inv_enrollment_lens = torch.load(investigator_enrollment_lens_file).to('cpu')
        self.inv_masks = torch.load(investigator_mask_file).to('cpu')

    def __len__(self):
        return len(set(self.trial_features))

    def __getitem__(self, idx):
        sample = {"trial": self.trial_features[idx], 
                  "label": self.labels[idx], 
                  "eth_label": self.eth_labels[idx],
                  "inv_static": self.inv_features[idx],
                  "inv_dx": self.inv_diagnosis[idx],
                  "inv_dx_len": self.inv_diagnosis_lens[idx],
                  "inv_rx": self.inv_prescription[idx],
                  "inv_rx_len": self.inv_prescription_lens[idx],
                  "inv_enroll": self.inv_enrollment[idx],
                  "inv_enroll_len": self.inv_enrollment_lens[idx],
                  "inv_mask": self.inv_masks[idx]}
        return sample
