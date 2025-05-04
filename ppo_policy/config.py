import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

# -- Device setting  -- #
# gpu_id = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

#################################################################################
# Dataset 
################################################################################# 
# -- Dataset(After preprocessing) & Checkpoint Path -- #
datapath = {
    'path_data_root': './dataset',
    'path_init_data': os.path.join('./dataset', 'worded_data.pickle'),
    'path_dictionary': os.path.join('./dataset', 'dictionary.pickle'),
    'path_train_data': os.path.join('./dataset', 'our_dataset.pickle'),
}
Load_Pretrain = True

# -- Sequence Length -- #
MaxSeqLen = 1200

# -- Inference Condition -- #
TOKEN_COUNT = 150
Pretrain_CKPT = './ckpt/pretrain_actor.pth'
Output_File_Path = './gen_midi/pretrain_actor.mid'

#################################################################################
# Model Setting 
################################################################################# 

# -- Actor Modules  -- #
ActorConfig = {
    "D_MODEL": 512,
    "N_LAYER": 12,
    "N_HEAD": 8
}

# -- Critic Setting  -- #
CriticConfig = {
    "D_MODEL": 512,
    "N_LAYER": 12,
    "N_HEAD": 8
}

# -- Discriminator Modules  -- #
DiscriConfig = {
    "MAX_SEQ": 2048,
    "D_MODEL": 512,
    "N_LAYER": 12,
    "N_HEAD": 8
}

