import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

RANDOM_SEED = 4
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

lats = [25, 48]
longs = [-125, -65]
eths = torch.tensor([50, 17, 17, 10, 5, 1]).float()
specialty_probs = pickle.load(open('../synthetic_data/specialty_probs.pkl', 'rb'))
dx_from_spec_probs = pickle.load(open('../synthetic_data/dx_from_spec_probs.pkl', 'rb'))
dx_probs = pickle.load(open('../synthetic_data/dx_probs.pkl', 'rb'))
rx_from_spec_probs = pickle.load(open('../synthetic_data/rx_from_spec_probs.pkl', 'rb'))
rx_probs = pickle.load(open('../synthetic_data/rx_probs.pkl', 'rb'))

invs = {}
num_invs = 30000
for i in tqdm(range(num_invs)):
    #######
    ### Inv Static
    #######
    lat = [np.random.uniform(lats[0], lats[1])]
    long = [np.random.uniform(longs[0], longs[1])]
    eth = torch.relu(torch.normal(eths, torch.tensor([10,10,10,10,10,10]))) + 1
    eth = (eth / eth.sum() * 100).tolist()

    static_features = []
    
    gender = [0, 0]
    gender[random.randint(0,1)] = 1
    static_features.extend(gender)

    # Profession Type
    profession_type = [0] * 20
    profession_type[random.randint(0,19)] = 1
    static_features.extend(profession_type)

    # Primary Specialty
    primary_specialty = [0] * 202
    specialty_idx = random.choices(list(specialty_probs.keys()), list(specialty_probs.values()))[0]
    primary_specialty[specialty_idx] = 1
    static_features.extend(primary_specialty)

    # Provider Type
    provider_type = [0] * 8
    provider_type[random.randint(0,7)] = 1
    static_features.extend(provider_type)

    # Primary Specialty Type
    primary_specialty_type = [0] * 36
    primary_specialty_type[random.randint(0,35)] = 1
    static_features.extend(primary_specialty_type)

    # Primary Experience
    primary_experience = [0] * 369
    primary_experience[random.randint(0,368)] = 1
    static_features.extend(primary_experience)

    # Primary Experience Category
    primary_experience_category = [0] * 24
    primary_experience_category[random.randint(0,23)] = 1
    static_features.extend(primary_experience_category)

    inv_static = lat + long + static_features + eth

    #######
    ### Inv Diagnosis
    #######
    inv_diagnosis = [0] * 500
    curr_dx = random.choices(list(dx_from_spec_probs[specialty_idx].keys()), list(dx_from_spec_probs[specialty_idx].values()))[0]
    inv_diagnosis[0] = curr_dx
    for j in range(1,500):
        curr_dx = random.choices(list(dx_probs[curr_dx].keys()), list(dx_probs[curr_dx].values()))[0]
        inv_diagnosis[j] = curr_dx

    #######
    ### Inv Prescription
    #######
    inv_prescription = [0] * 500
    curr_rx = random.choices(list(rx_from_spec_probs[specialty_idx].keys()), list(rx_from_spec_probs[specialty_idx].values()))[0]
    inv_prescription[0] = curr_rx
    for j in range(1,500):
        curr_rx = random.choices(list(rx_probs[curr_rx].keys()), list(rx_probs[curr_rx].values()))[0]
        inv_prescription[j] = curr_rx
        
        

    #######
    ### Inv Features
    #######
    invs[i] = (torch.tensor(inv_static), torch.tensor(inv_diagnosis), torch.tensor(inv_prescription))

pickle.dump(invs, open('../synthetic_data/inv_pool.pkl', 'wb'))
