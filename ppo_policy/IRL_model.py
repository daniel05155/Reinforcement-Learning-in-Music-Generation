import os 
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerConfig, LongformerForMultipleChoice, LongformerModel
from transformers import TrajectoryTransformerConfig, TrajectoryTransformerModel
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.builders import RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask
import pickle
from tqdm import tqdm

# -- Config -- # 
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 

# --- modules config --- #
MAX_SEQ_LEN = 1024
D_MODEL = 512
N_LAYER = 12
N_HEAD = 8    
path_exp = 'exp'
N_STATES = 50

# Pretrain_ckpt = '/data/Der_CODES/DQN-cp/ckpt/trainloss_22.pt' 

################################################################################
# Embedding Configuartion
###############################################################################
class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

################################################################################
# Discriminator-LongFormer 
################################################################################
class LongFormer(nn.Module):
    def __init__(self, n_token):
        super(LongFormer, self).__init__()

        self.n_token = n_token
        self.MAX_SEQ = MAX_SEQ_LEN * 2
        self.D_MODEL = D_MODEL
        self.N_layer = N_LAYER
        self.N_head  = N_HEAD
        self.CE_loss = nn.CrossEntropyLoss()

        self.emb_sizes =  [128, 256, 64, 512, 256, 256]  # CP Embedding

        print('Disc token >>>>> ', self.n_token)
        self.word_emb_tempo     = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.word_emb_chord     = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.word_emb_barbeat   = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.word_emb_pitch     = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.word_emb_duration  = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.word_emb_velocity  = Embeddings(self.n_token[5], self.emb_sizes[5])
    
        self.proj = nn.Linear(sum(self.emb_sizes), self.D_MODEL)

        # Features projection
        self.proj_tempo    = nn.Linear(self.D_MODEL, self.n_token[0])        
        self.proj_chord    = nn.Linear(self.D_MODEL, self.n_token[1])
        self.proj_barbeat  = nn.Linear(self.D_MODEL, self.n_token[2])
        self.proj_pitch    = nn.Linear(self.D_MODEL, self.n_token[3])
        self.proj_duration = nn.Linear(self.D_MODEL, self.n_token[4])
        self.proj_velocity = nn.Linear(self.D_MODEL, self.n_token[5])

        # Individual evaluation
        self.eval_tempo    = nn.Linear(self.n_token[0], 1)        
        self.eval_chord    = nn.Linear(self.n_token[1], 1)
        self.eval_barbeat  = nn.Linear(self.n_token[2], 1)
        self.eval_pitch    = nn.Linear(self.n_token[3], 1)
        self.eval_duration = nn.Linear(self.n_token[4], 1)
        self.eval_velocity = nn.Linear(self.n_token[5], 1)
        self.sigmoid = nn.Sigmoid()
        
        self.longformer_config = LongformerConfig(max_position_embeddings=self.MAX_SEQ,
                                                  hidden_size = self.D_MODEL,
                                                  num_hidden_layers = self.N_layer,
                                                  num_attention_heads = self.N_head,
                                                  hidden_act = "gelu",
                                                  hidden_dropout_prob = 0.1,
                                                  attention_probs_dropout_prob = 0.1,
                                                  position_embedding_type = "relative_key", # "absolute", "relative_key", "relative_key_query"
                                                  intermediate_size = 1024, 
                                                  attention_window = N_STATES,    # D_MODEL 512
                                                  )
        self.longformer = LongformerModel(self.longformer_config)
    
    def forward(self, data, masks):
        """
        args:
            data: (batch, windows_size, 6)
            masks: (batch, windows_size)        
        """
        tempo    = self.word_emb_tempo(data[..., 0])           # Take tempo in every note (取每個row的第1個col)
        chord    = self.word_emb_chord(data[..., 1])           # Take bar in every note   (取每個row的第2個col)
        barbeat  = self.word_emb_barbeat(data[..., 2])         # Take poistion in every note 
        pitch    = self.word_emb_pitch(data[..., 3])
        duration = self.word_emb_duration(data[..., 4])
        velocity = self.word_emb_velocity(data[..., 5])
        
        embedding = torch.cat([tempo, chord, barbeat, pitch, duration, velocity], dim=-1)
        x = self.proj(embedding)
        
        outputs = self.longformer(inputs_embeds=x, attention_mask=masks)
        sequence_output = outputs.last_hidden_state     # (batch, seq_len, hidden_size)
        sequence_mean   = sequence_output.mean(dim=1)   # (batch, hidden_size)
        element_score = self.score_classifier(sequence_mean)

        return element_score

    def compute_CEloss(self, predict, target, loss_mask):
        loss = self.CE_loss(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss
    
    def token_forward(self, data, target, loss_mask):
        tempo    = self.word_emb_tempo(data[..., 0])          # Take tempo in every note (取每個row的第1個col)
        chord    = self.word_emb_chord(data[..., 1])          # Take bar in every note   (取每個row的第2個col)
        barbeat  = self.word_emb_barbeat(data[..., 2])        # Take poistion in every note 
        pitch    = self.word_emb_pitch(data[..., 3])
        duration = self.word_emb_duration(data[..., 4])
        velocity = self.word_emb_velocity(data[..., 5])
        
        embedding = torch.cat([tempo, chord, barbeat, pitch, duration, velocity], dim=-1)
        cat_emb = self.proj(embedding)
        
        outputs = self.longformer(inputs_embeds=cat_emb, attention_mask=loss_mask)
        sequence_output = outputs.last_hidden_state         # (batch, seq_len, hidden_size)

        # Each features network
        y_tempo    = self.proj_tempo(sequence_output)
        y_chord    = self.proj_chord(sequence_output)
        y_barbeat  = self.proj_barbeat(sequence_output)
        y_pitch    = self.proj_pitch(sequence_output)
        y_duration = self.proj_duration(sequence_output)
        y_velocity = self.proj_velocity(sequence_output)

        tempo_hid    = self.eval_tempo(y_tempo).mean(dim=1)
        chord_hid    = self.eval_chord(y_chord).mean(dim=1)
        barbeat_hid  = self.eval_barbeat(y_barbeat).mean(dim=1)
        pitch_hid    = self.eval_pitch(y_pitch).mean(dim=1)
        duration_hid = self.eval_duration(y_duration).mean(dim=1)
        velocity_hid = self.eval_velocity(y_velocity).mean(dim=1)

        sum_score = 0
        attr_list = [tempo_hid, chord_hid, barbeat_hid, pitch_hid, duration_hid, velocity_hid]
        for idx, attr in enumerate(attr_list):
            attr_list[idx] = self.sigmoid(attr)
            sum_score += attr_list[idx]
        sum_score = sum_score / len(attr_list)

        return sum_score