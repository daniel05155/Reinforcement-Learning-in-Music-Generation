import sys
import os
import math
import time
import random
import pickle
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.builders import RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
from tqdm import tqdm
import wandb

from AIRL import RewardDiscri

################################################################################
# config
################################################################################
gid = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 

###--- data ---###
path_data_root = '/data/dataset_Pop1K7/representations/uncond/cp/ailab17k_from-scratch_cp'
path_train_data = os.path.join(path_data_root, 'train_data_linear.npz')
path_dictionary = os.path.join(path_data_root, 'dictionary.pkl')

###--- Pretrain ---###
Pretrain_ckpt = '/data/Der_CODES/DQN-cp/ckpt/trainloss_13.pt' 


################################################################################
# Model config 
################################################################################
D_MODEL = 512
N_LAYER = 12
N_HEAD = 8    
max_grad_norm = 3    
Target_update = 50            # Update freqency of target net
EPSILON = 0.9                 # greedy policy
GAMMA = 0.9                   # reward discount


################################################################################
# Training config 
################################################################################
NUM_SONGS = 1200          # NUM_EPOCH
EPISODES = 50
SEQ_LEN = 1000
N_STATES = 50
N_FEATURES = 6          # Features in each state
N_ACTIONS = 25

WINDOW_SIZE = 50        # STATE SIZE    
BUFFER_SIZE = 10000       # MEMORY CAPACITY
ACTION_DIM  = 6
NUM_ACTION  = 25        # Agent prediction

batch_size = 10
init_lr = 0.001


################################################################################
# File IO
################################################################################
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4

##########################################################################################################################
# Buffer 
##########################################################################################################################
class AgentMemory(object):
    def __init__(self):
        self.states_agent  = torch.zeros((BUFFER_SIZE, N_STATES, N_FEATURES))   # (buffer, 50, 6)
        self.actions_agent = torch.zeros((BUFFER_SIZE, N_ACTIONS, N_FEATURES))  
        self.rewards_agent = torch.zeros((BUFFER_SIZE, 1))                      # (buffer, 50, 1)
        self.next_states_agent = torch.zeros((BUFFER_SIZE, N_STATES, N_FEATURES)) 
        self.dones_agent   = torch.zeros((BUFFER_SIZE, 1))

        self.memory_counter = 0         # Count size of buffer

    def store_transition(self, state, action, reward, next_state, done):    
        index = self.memory_counter % BUFFER_SIZE   #

        self.states_agent[index,:,:]      = state
        self.actions_agent[index,:,:]     = action
        self.rewards_agent[index,:]       = reward
        self.next_states_agent[index,:,:] = next_state
        self.dones_agent[index,:]         = done

        self.memory_counter += 1                                                
    
    # -- Sampling batch size -- #
    def sampling(self, batch_size):
        # sample_idx  = np.random.randint(0, buffer_size-batch_size)
        sample_idx  = np.random.choice(BUFFER_SIZE, batch_size) 

        states_batch    = self.states_agent[sample_idx, :, :]
        actinos_batch   = self.actions_agent[sample_idx, :, :]
        rewards_batch   = self.rewards_agent[sample_idx, :]
        next_states_batch = self.next_states_agent[sample_idx, :, :]
        dones_batch     = self.dones_agent[sample_idx, :]

        return states_batch, actinos_batch, rewards_batch, next_states_batch, dones_batch
    
    # -- Take agent data in each episode -- #
    def get(self):
        agentdata_states      = self.states_agent[:, :, :].long().cuda()
        agentdata_actinos     = self.actions_agent[:, :, :].long().cuda()
        agentdata_rewards     = self.rewards_agent[:, :].long().cuda()
        agentdata_next_states = self.next_states_agent[:, :, :].long().cuda()
        agentdata_dones       = self.dones_agent[:, :].long().cuda()

        return agentdata_states, agentdata_actinos, agentdata_rewards, agentdata_next_states, agentdata_dones


# -- Expert Buffer -- #
class ExpertMemory(object):
    def __init__(self):
        self.states_exp  = torch.zeros((BUFFER_SIZE, N_STATES, N_FEATURES))   # (buffer, 50, 6)
        self.actions_exp = torch.zeros((BUFFER_SIZE, N_ACTIONS, N_FEATURES))  # (buffer, 1, 6)
        self.rewards_exp = torch.zeros((BUFFER_SIZE, 1))                      # (buffer, 1)
        self.next_states_exp  = torch.zeros((BUFFER_SIZE, N_STATES, N_FEATURES))  # (buffer, 50, 6)
        self.dones_exp =  torch.zeros((BUFFER_SIZE, 1))     # (buffer, 1)

        self.mask_state =  torch.zeros((BUFFER_SIZE, N_STATES))
        self.mask_next_state =  torch.zeros((BUFFER_SIZE, N_STATES))
        self.memory_counter = 0    

    def store_transition(self, state, action, reward, next_state, done, mask_state, mask_next_state):    
        index = self.memory_counter % BUFFER_SIZE   #

        self.states_exp[index, :, :]      = state
        self.actions_exp[index, :, :]     = action
        self.rewards_exp[index, :]        = reward
        self.next_states_exp[index, :, :] = next_state
        self.dones_exp[index, :]          = done
        self.mask_state[index, :]         = mask_state
        self.mask_next_state[index, :]    = mask_next_state

        self.memory_counter += 1   

    def sampling(self, batch_size):
        # sample_idx  = np.random.randint(0, buffer_size-batch_size)
        sample_idx  = np.random.choice(BUFFER_SIZE, batch_size) 

        states_batch    = self.states_exp[sample_idx, :, :]
        actinos_batch   = self.actions_exp[sample_idx, :, :]
        rewards_batch   = self.rewards_exp[sample_idx, :]
        next_states_batch = self.next_states_exp[sample_idx, :, :]
        dones_batch     = self.dones_exp[sample_idx, :]

        return states_batch, actinos_batch, rewards_batch, next_states_batch, dones_batch

    def get(self):
        expdata_states  = self.states_exp[:, :, :].long().cuda()
        expdata_actinos = self.actions_exp[:, :, :].long().cuda()
        expdata_rewards = self.rewards_exp[:, :]
        expdata_next_states = self.next_states_exp [:, :, :].long().cuda()
        expdata_dones   = self.dones_exp[:, :].long().cuda()
        
        expdata_mask_state      = self.mask_state[:, :].long().cuda()
        expdata_mask_next_state = self.mask_next_state[:, :].long().cuda()
        
        all_part = (expdata_states, expdata_actinos, expdata_rewards, expdata_next_states, 
                    expdata_dones, expdata_mask_state, expdata_mask_next_state)
        
        return all_part


################################################################################
# Sampling
################################################################################
# -- temperature -- #
def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs

def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word

# -- nucleus -- #
def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

def sampling(logit, p=None, t=1.0):
    logit = logit.squeeze().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)
    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


################################################################################
# Model
################################################################################
def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, n_token, is_training=True):
        super(TransformerModel, self).__init__()

        # --- params config --- #
        self.n_token = n_token   
        self.d_model = D_MODEL 
        self.n_layer = N_LAYER #
        self.dropout = 0.1
        self.n_head = N_HEAD #
        self.d_head = D_MODEL // N_HEAD
        self.d_inner = 2048
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.emb_sizes = [128, 256, 64, 512, 128, 128]              # self.emb_sizes = [128, 256, 64, 32, 512, 128, 128]

        # embeddings
        print('Token_class >>>>>:', self.n_token)
        self.word_emb_tempo     = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.word_emb_chord     = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.word_emb_barbeat   = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.word_emb_pitch     = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.word_emb_duration  = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.word_emb_velocity  = Embeddings(self.n_token[5], self.emb_sizes[5])
        self.pos_emb            = PositionalEncoding(self.d_model, self.dropout)

        # linear 
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), self.d_model)

        # encoder
        if is_training:
            # encoder (training)
            self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model//self.n_head,
                value_dimensions=self.d_model//self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="causal-linear",
            ).get()
        else:
            # encoder (inference)
            print(' [o] using RNN backend.')
            self.transformer_encoder = RecurrentEncoderBuilder.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model//self.n_head,
                value_dimensions=self.d_model//self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="causal-linear",
            ).get()

        # blend with type
        self.project_concat_type = nn.Linear(self.d_model, self.d_model)    # nn.Linear(self.d_model+32, self.d_model)

        # individual output
        self.proj_tempo    = nn.Linear(self.d_model, self.n_token[0])        
        self.proj_chord    = nn.Linear(self.d_model, self.n_token[1])
        self.proj_barbeat  = nn.Linear(self.d_model, self.n_token[2])
        self.proj_pitch    = nn.Linear(self.d_model, self.n_token[3])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[4])
        self.proj_velocity = nn.Linear(self.d_model, self.n_token[5])

    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def train_step(self, x, target, loss_mask):
        h  = self.forward_hidden(x)
        y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = self.forward_output(h, target)
        
        # reshape (b, s, f) -> (b, f, s)
        y_tempo     = y_tempo[:, ...].permute(0, 2, 1)
        y_chord     = y_chord[:, ...].permute(0, 2, 1)
        y_barbeat   = y_barbeat[:, ...].permute(0, 2, 1)
        y_pitch     = y_pitch[:, ...].permute(0, 2, 1)
        y_duration  = y_duration[:, ...].permute(0, 2, 1)
        y_velocity  = y_velocity[:, ...].permute(0, 2, 1)
        
        # loss
        loss_tempo = self.compute_loss(
                y_tempo, target[..., 0], loss_mask)
        loss_chord = self.compute_loss(
                y_chord, target[..., 1], loss_mask)
        loss_barbeat = self.compute_loss(
                y_barbeat, target[..., 2], loss_mask)
        loss_pitch = self.compute_loss(
                y_pitch, target[..., 3], loss_mask)
        loss_duration = self.compute_loss(
                y_duration, target[..., 4], loss_mask)
        loss_velocity = self.compute_loss(
                y_velocity, target[..., 5], loss_mask)

        return loss_tempo, loss_chord, loss_barbeat, loss_pitch, loss_duration, loss_velocity


    def forward_hidden(self, x, memory=None, is_training=True):
        '''
        linear transformer: b x s x f
        x.shape=(bs, nf)
        '''
        # embeddings
        emb_tempo =    self.word_emb_tempo(x[..., 0])
        emb_chord =    self.word_emb_chord(x[..., 1])
        emb_barbeat =  self.word_emb_barbeat(x[..., 2])         # emb_type = self.word_emb_type(x[..., 3])
        emb_pitch =    self.word_emb_pitch(x[..., 3])
        emb_duration = self.word_emb_duration(x[..., 4])
        emb_velocity = self.word_emb_velocity(x[..., 5])

        embs = torch.cat(
            [
                emb_tempo,
                emb_chord,
                emb_barbeat,
                emb_pitch,
                emb_duration,
                emb_velocity,
            ], dim=-1)

        emb_linear = self.in_linear(embs)
        pos_emb = self.pos_emb(emb_linear)

        # assert False
        
        # transformer
        if is_training:
            # mask
            attn_mask = TriangularCausalMask(pos_emb.size(1), device=x.device)
            h = self.transformer_encoder(pos_emb, attn_mask)                # y: b x s x d_model

            return h 
        
        else:
            pos_emb = pos_emb.squeeze(0)
            h, memory = self.transformer_encoder(pos_emb, memory=memory)    # y: s x d_model

            return h, memory 
    
    def forward_output(self, h, y):
        y_tempo    = self.proj_tempo(h)
        y_chord    = self.proj_chord(h)
        y_barbeat  = self.proj_barbeat(h)
        y_pitch    = self.proj_pitch(h)
        y_duration = self.proj_duration(h)
        y_velocity = self.proj_velocity(h)

        return  y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity


    def forward(self, x, target):
        hid = self.forward_hidden(x) 
        y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = self.forward_output(hid, target)
        return y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity 


################################################################################
# RL Policy
################################################################################
class DQN(object):
    def __init__(self, n_class, Pretrain=True):
        self.eval_net   = TransformerModel(n_class).cuda()
        self.target_net = TransformerModel(n_class).cuda()
        
        if Pretrain==True:
            checkpoint = torch.load(Pretrain_ckpt)
            self.eval_net.load_state_dict(checkpoint['model_state_dict'])
            self.eval_net.train()

        self.agent_buffer  = AgentMemory()   
        self.expert_buffer = ExpertMemory()
        self.optim = optim.Adam(self.eval_net.parameters(), lr=init_lr)
        self.sigmoid = nn.Sigmoid()
        self.count = 0

    def choose_action(self, x, target):
        
        h = self.eval_net.forward_hidden(x)     #　h.shape=(1,50,512)
        y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = self.eval_net.forward_output(h, target)
        
        m = nn.Softmax(dim=-1)
        y_tempo, y_chord, y_barbeat     = m(y_tempo), m(y_chord), m(y_barbeat)
        y_pitch, y_duration, y_velocity = m(y_pitch), m(y_duration), m(y_velocity)
        
        # Take index of max value 
        tempo, chord, barbeat  = torch.argmax(y_tempo, dim=-1), torch.argmax(y_chord, dim=-1), torch.argmax(y_barbeat, dim=-1)
        pitch, duration, velocity = torch.argmax(y_pitch, dim=-1), torch.argmax(y_duration, dim=-1), torch.argmax(y_velocity, dim=-1)

        # (1,7)-> (1,6)
        # action = torch.cat((tempo[:,-1], chord[:,-1], barbeat[:,-1],pitch[:,-1], duration[:,-1], velocity[:,-1]), dim=0)
        # action = action.unsqueeze(0)
        
        # Action(25)
        action = None
        for idx in range(N_ACTIONS):
            tmp = torch.cat((tempo[:,-idx], chord[:,-idx], barbeat[:,-idx],pitch[:,-idx], duration[:,-idx], velocity[:,-idx]), dim=0)
            tmp = tmp.unsqueeze(0)
            
            if action==None:      
                action = tmp
            else:
                action = torch.cat((action, tmp), dim=0)

        return action
    
    
    def update(self, agent_transition, expert_transition):
        # Update target network
        if self.count % Target_update  == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict()) 
        
        # Expert trend # 
        expert_state = expert_transition['state']
        expert_next_state = expert_transition['nextstate']

        # Eval Network #
        agent_state = agent_transition['state'].long().cuda()   # (batch,state,Feature)
        agent_next_state = agent_transition['nextstate'].long().cuda()
        agent_action = agent_transition['action'].long().cuda()
        agent_reward = agent_transition['reward'].float().cuda()
        agent_done = agent_transition['done'].long().cuda()

        y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = self.eval_net(agent_state, expert_state)  # state對應的q-value
        
        qval_tempo      = y_tempo.gather(2, agent_action[:,:,0].unsqueeze(0)).squeeze(0)           # (batch_size,1)
        qval_tempo      = self.sigmoid(qval_tempo)

        qval_chord      = y_chord.gather(2, agent_action[:,:,1].unsqueeze(0)).squeeze(0)           # (batch_size,1)
        qval_chord      = self.sigmoid(qval_chord)

        qval_barbeat    = y_barbeat.gather(2, agent_action[:,:,2].unsqueeze(0)).squeeze(0) 
        qval_barbeat    = self.sigmoid(qval_barbeat)

        qval_pitch      = y_pitch.gather(2, agent_action[:,:,3].unsqueeze(0)).squeeze(0) 
        qval_pitch      = self.sigmoid(qval_pitch)

        qval_duration   = y_duration.gather(2, agent_action[:,:,4].unsqueeze(0)).squeeze(0) 
        qval_duration   = self.sigmoid(qval_duration)

        qval_velocity   = y_velocity.gather(2, agent_action[:,:,5].unsqueeze(0)).squeeze(0) 
        qval_velocity   = self.sigmoid(qval_velocity)

        ###　Target Net ### 
        next_tempo, next_chord, next_barbeat, next_pitch, next_duration, next_velocity = self.target_net(agent_next_state, expert_next_state)

        ### Max Q-Value of next state in each column(each feature) ###
        next_tempo          = next_tempo.max(2)[0]
        max_next_q_tempo, _ = next_tempo.topk(N_ACTIONS, dim=1)

        next_chord          = next_chord.max(2)[0]
        max_next_q_chord, _ = next_chord.topk(N_ACTIONS, dim=1)

        next_barbeat          = next_barbeat.max(2)[0]
        max_next_q_barbeat, _ = next_barbeat.topk(N_ACTIONS, dim=1)

        next_pitch          = next_pitch.max(2)[0]
        max_next_q_pitch, _ = next_pitch.topk(N_ACTIONS, dim=1)
      
        next_duration          = next_duration.max(2)[0]
        max_next_q_duration, _ = next_duration.topk(N_ACTIONS, dim=1)

        next_velocity          = next_velocity.max(2)[0]   
        max_next_q_velocity, _ = next_velocity.topk(N_ACTIONS, dim=1)
        
        # TD Loss # 
        q_targets_tempo     = agent_reward + GAMMA * (1-agent_done) * max_next_q_tempo
        q_targets_tempo     = self.sigmoid(q_targets_tempo)

        q_targets_chord     = agent_reward + GAMMA * (1-agent_done) * max_next_q_chord
        q_targets_chord     = self.sigmoid(q_targets_chord)
        
        q_targets_barbeat   = agent_reward + GAMMA * (1-agent_done) * max_next_q_barbeat
        q_targets_barbeat   = self.sigmoid(q_targets_barbeat)

        q_targets_pitch     = agent_reward + GAMMA * (1-agent_done) * max_next_q_pitch
        q_targets_pitch    = self.sigmoid(q_targets_pitch)


        q_targets_duration  = agent_reward + GAMMA * (1-agent_done) * max_next_q_duration
        q_targets_duration  = self.sigmoid(q_targets_duration)

        q_targets_velocity  = agent_reward + GAMMA * (1-agent_done) * max_next_q_velocity
        q_targets_velocity  = self.sigmoid(q_targets_velocity)


        tempo_loss      = F.mse_loss(qval_tempo,    q_targets_tempo)
        chord_loss      = F.mse_loss(qval_chord,    q_targets_chord)
        barbeat_loss    = F.mse_loss(qval_barbeat,  q_targets_barbeat)
        pitch_loss      = F.mse_loss(qval_pitch,    q_targets_pitch)
        duration_loss   = F.mse_loss(qval_duration, q_targets_duration)
        velocity_loss   = F.mse_loss(qval_velocity, q_targets_velocity )
        
        loss = (tempo_loss + chord_loss + barbeat_loss + pitch_loss + duration_loss + velocity_loss) / 6
        self.optim.zero_grad() 
        loss.backward()  
        self.optim.step()
        
        self.count+=1

        return loss


if __name__ == '__main__':

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary
    
    train_data = np.load(path_train_data)

    # run = wandb.init(
    #     project="DQN-RL-Music",
        
    #     config={
    #         "learning_rate": init_lr,
    #         "epochs": NUM_SONGS,
    #         'batch_size': batch_size,
    #         'SEQ_LEN': SEQ_LEN,
    #         'BUFFER_SIZE': BUFFER_SIZE,

    #         })

    # Number of each feature(Delete 'type'(index=3)): [56, 135, 18, 87, 18, 25]
    n_class = []
    for key in event2word.keys():
        if key !='type':
            n_class.append(len(dictionary[0][key]))
    
    ###--- Buffer ---###
    AgentBuffer  = AgentMemory()
    ExpertBuffer = ExpertMemory()

    ###--- RL Policy ---###
    Agent    = DQN(n_class, Pretrain=True)
    Rewarder = RewardDiscri(n_class, Pretrain=False)

    ###--- Unpack Training data ---###
    train_x = train_data['x']           # (batch,3584,7):(1625,3584,7)
    train_y = train_data['y']           # (batch,3584,7):(1625,3584,7)    
    train_mask = train_data['mask']     # (batch,3584)
    num_batch = len(train_x) // batch_size
    
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    train_mask = torch.from_numpy(train_mask)

    ###--- Delete 'type'(index=3) ---###
    train_x = torch.cat( (train_x[:,:,:3], train_x[:,:,4:]), dim=-1)
    train_y = torch.cat( (train_y[:,:,:3], train_y[:,:,4:]), dim=-1)
    
    # SEQ_LEN
    data_x    = train_x[:,:SEQ_LEN,:].long().cuda()
    data_y    = train_y[:,:SEQ_LEN*2,:].long().cuda()
    data_mask = train_mask[:,:SEQ_LEN+1000].float().cuda()

    gene_reward = []
    TD_loss = []
    for epoch in range(NUM_SONGS):  
        # Env.reset
        state_x  = data_x[epoch,:WINDOW_SIZE,:]
        expert_x = data_y[epoch,:,:]

        done = torch.tensor(1).long().cuda()       # Agent Done 
        cnt_reward = 0
        for num in range(0, EPISODES): 
            # idx  = np.random.randint(0, SEQ_LEN-WINDOW_SIZE, size=1)[0]
            Expert_state      = expert_x[num: num+WINDOW_SIZE]    
            Expert_next_state = expert_x[num+50: num+50+WINDOW_SIZE]
            Expert_action     = expert_x[num+WINDOW_SIZE]       # expert_x[num+WINDOW_SIZE]
            Expert_reward     = torch.tensor(1.0).float().cuda()            
            Expert_done       = torch.tensor(0).long().cuda()            
            Expert_mask_state = train_mask[epoch, num: num+WINDOW_SIZE]          # State mask
            Expert_mask_nextstate = train_mask[epoch, num+1: num+1+WINDOW_SIZE]  # Next state mask

            done = torch.tensor(0).long().cuda()       
            action = Agent.choose_action(state_x.unsqueeze(0), Expert_state.unsqueeze(0))
            
            next_state  = torch.cat((state_x[:N_ACTIONS,:], action), dim=0)           # (state_x, action): Same Dim.

            agent_reward = torch.tensor(0.5).float().cuda()         # Temporary reward
            
            ## store_transition  ## 
            AgentBuffer.store_transition(state_x, action, agent_reward, next_state, done)            
            ExpertBuffer.store_transition(Expert_state, action, Expert_reward, 
                                          Expert_next_state, Expert_done, Expert_mask_state, Expert_mask_nextstate)
            
            state_x = next_state    # Turn to next state

            ## Update ## 
            if AgentBuffer.memory_counter > BUFFER_SIZE:

                ###--- Rewarder ---###  -> Plot
                agent_traj  = AgentBuffer.get()
                expert_traj = ExpertBuffer.get()
                traj_reward, answer_reward  = Rewarder.update_disc(agent_traj, expert_traj)

                AgentBuffer.rewards_agent[:,:] = traj_reward
                gene_reward.append(torch.sum(AgentBuffer.rewards_agent[:])/300)
                
                state, action, reward, next_state, done = AgentBuffer.sampling(batch_size)
                agent_transition = {'state':state, 'action':action, 'reward':reward, 'nextstate':next_state, 'done':done}
                
                expert_state, expert_action, expert_reward, expert_next_state, expert_done = ExpertBuffer.sampling(batch_size)
                expert_transition = {'state':state, 'action':action, 'reward':reward, 
                                     'nextstate':next_state, 'done':expert_done}

                loss = Agent.update(agent_transition, expert_transition)
                # wandb.log({"TD_loss": loss})
                
                TD_loss.append(loss)
                sys.stdout.write('Epoch: {}/{} | Episode: {}/{} | TD_Loss: {:03f} \r'.format(epoch , NUM_SONGS, num,EPISODES, loss))
                sys.stdout.flush()
            else:
                sys.stdout.write('Epoch: {}/{} | Episode: {}/{} | Buffer_Size:{}\r'.format(epoch , NUM_SONGS, num,EPISODES, AgentBuffer.memory_counter))
                sys.stdout.flush()
        
        
        if (epoch>10) and (epoch%5==0):
            os.makedirs('./ckpt',exist_ok=True)
            torch.save(Agent.eval_net.state_dict(), f'./ckpt/dqn_best.pt')

            # Save recording 
            record = {'Agent':gene_reward, 'TD_Loss': TD_loss}
            with open( './ckpt/avg_reward.pickle', 'wb') as file:
                pickle.dump(record, file)    


'''
1. Loss-global & lovcal
2. IRL-Reward
3. 

'''