import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader


# -- Agent Modules  -- #
AgentConfig = {
    "D_MODEL": 512,
    "N_LAYER": 12,
    "N_HEAD": 8
}



# -- Discriminator Modules  -- #
DiscriConfig = {
    "D_MODEL": 512,
    "N_LAYER": 6,
    "N_HEAD": 8
}


