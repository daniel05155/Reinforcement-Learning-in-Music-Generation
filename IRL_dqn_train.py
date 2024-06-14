import sys
import os
import math
import time
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from model import LinearTransformer

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
from tqdm import tqdm
import wandb
from AIRL import RewardDiscri
from utils import bi_loss_plot

################################################################################
# config
################################################################################
gid = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# print(f'Device: {device}')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 

###--- data ---###
path_data_root = '/data/dataset_Pop1K7/representations/uncond/cp/ailab17k_from-scratch_cp'
path_train_data = os.path.join(path_data_root, 'train_data_linear.npz')
path_dictionary = os.path.join(path_data_root, 'dictionary.pkl')

###--- Checkpoint path  ---###
Pretrain_ckpt = '/data/Der_CODES/DQN-cp/ckpt/trainloss_13.pt' 
save_ckpt_path = './ckpt/dqn_best.pt'    

################################################################################
# Model config 
################################################################################ 
Target_update = 50            # Update freqency of target net
EPSILON = 0.9                 # greedy policy
GAMMA = 0.95                  # reward discount

################################################################################
# Training config 
################################################################################
NUM_SONGS = 1500          # NUM_EPOCH
EPISODES = 50
SEQ_LEN = 1000
N_STATES = 50
N_FEATURES = 6          # Features in each state
N_ACTIONS = 25

WINDOW_SIZE = 50        # STATE SIZE    
BUFFER_SIZE = 20000     # MEMORY CAPACITY
ACTION_DIM  = 6
NUM_ACTION  = 25        # Agent prediction

batch_size = 30
init_lr = 0.01


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
        self.states_agent  = np.zeros((BUFFER_SIZE, N_STATES, N_FEATURES))   # (buffer, 50, 6)
        self.actions_agent = np.zeros((BUFFER_SIZE, N_ACTIONS, N_FEATURES))  
        self.rewards_agent = np.zeros((BUFFER_SIZE, 1))                      # (buffer, 50, 1)
        self.next_states_agent = np.zeros((BUFFER_SIZE, N_STATES, N_FEATURES)) 
        self.dones_agent   = np.zeros((BUFFER_SIZE, 1))

        self.memory_counter = 0         # Count size of buffer

    def store_transition(self, state, action, reward, next_state, done):    
        index = self.memory_counter % BUFFER_SIZE   #
        # tensor -> np.array
        state   = state.detach().cpu().numpy()
        action  = action.detach().cpu().numpy()
        reward  = reward.detach().cpu().numpy()
        next_state = next_state.detach().cpu().numpy()
        done = done.detach().cpu().numpy()

        self.states_agent[index,:,:]      = state
        self.actions_agent[index,:,:]     = action
        self.rewards_agent[index,:]       = reward
        self.next_states_agent[index,:,:] = next_state
        self.dones_agent[index,:]         = done
        
        self.memory_counter += 1                                                

    # -- Sampling the number of batch size -- #
    def sampling(self, batch_size):
        sample_idx  = np.random.choice(BUFFER_SIZE, batch_size)  # sample_idx  = np.random.randint(0, buffer_size-batch_size)

        states_batch    = self.states_agent[sample_idx, :, :]
        actinos_batch   = self.actions_agent[sample_idx, :, :]
        rewards_batch   = self.rewards_agent[sample_idx, :]
        next_states_batch = self.next_states_agent[sample_idx, :, :]
        dones_batch     = self.dones_agent[sample_idx, :]

        # np.array -> tensor
        states_batch  = torch.from_numpy(states_batch).long().cuda()
        actinos_batch = torch.from_numpy(actinos_batch).long().cuda()
        rewards_batch = torch.from_numpy(rewards_batch).float()
        next_states_batch = torch.from_numpy(next_states_batch).long().cuda()
        dones_batch   = torch.from_numpy(dones_batch).long().cuda()

        return states_batch, actinos_batch, rewards_batch, next_states_batch, dones_batch
    
    # -- Take agent data in each episode -- #
    def get(self):
        agentdata_states      = torch.from_numpy(self.states_agent[:, :, :]).long().cuda()
        agentdata_actinos     = torch.from_numpy(self.actions_agent[:, :, :]).long().cuda()
        agentdata_rewards     = torch.from_numpy(self.rewards_agent[:, :]).float().cuda()
        agentdata_next_states = torch.from_numpy(self.next_states_agent[:, :, :]).long().cuda()
        agentdata_dones       = torch.from_numpy(self.dones_agent[:, :]).long().cuda()

        return agentdata_states, agentdata_actinos, agentdata_rewards, agentdata_next_states, agentdata_dones


# -- Expert Buffer -- #
class ExpertMemory(object):
    def __init__(self):
        self.states_exp      = np.zeros((BUFFER_SIZE, N_STATES, N_FEATURES))   # (buffer, 50, 6)
        self.actions_exp     = np.zeros((BUFFER_SIZE, N_ACTIONS, N_FEATURES))  # (buffer, 1, 6)
        self.rewards_exp     = np.zeros((BUFFER_SIZE, 1))                      # (buffer, 1)
        self.next_states_exp = np.zeros((BUFFER_SIZE, N_STATES, N_FEATURES))   # (buffer, 50, 6)
        self.dones_exp       = np.zeros((BUFFER_SIZE, 1))     # (buffer, 1)

        self.mask_state      = torch.zeros((BUFFER_SIZE, N_STATES))
        self.mask_next_state = torch.zeros((BUFFER_SIZE, N_STATES))
        self.memory_counter = 0    

    def store_transition(self, state, action, reward, next_state, done, mask_state, mask_next_state):    
        index = self.memory_counter % BUFFER_SIZE   

        # tensor -> np.array
        state   = state.detach().cpu().numpy()
        action  = action.detach().cpu().numpy()
        reward  = reward.detach().cpu().numpy()
        next_state = next_state.detach().cpu().numpy()
        done = done.detach().cpu().numpy()

        self.states_exp[index, :, :]      = state
        self.actions_exp[index, :, :]     = action
        self.rewards_exp[index, :]        = reward
        self.next_states_exp[index, :, :] = next_state
        self.dones_exp[index, :]          = done

        self.mask_state[index, :]         = mask_state          # torch type
        self.mask_next_state[index, :]    = mask_next_state     # torch type

        self.memory_counter += 1   

    
    def sampling(self, batch_size):
        # sample_idx  = np.random.randint(0, buffer_size-batch_size)
        sample_idx  = np.random.choice(BUFFER_SIZE, batch_size) 

        states_batch    = self.states_exp[sample_idx, :, :]
        actinos_batch   = self.actions_exp[sample_idx, :, :]
        rewards_batch   = self.rewards_exp[sample_idx, :]
        next_states_batch = self.next_states_exp[sample_idx, :, :]
        dones_batch     = self.dones_exp[sample_idx, :]
        mask_state_batch = self.mask_state[sample_idx, :]             # torch type
        mask_next_state_batch  = self.mask_next_state[sample_idx, :]  # torch type

        # np.array -> tensor
        states_batch  = torch.from_numpy(states_batch).long().cuda()
        actinos_batch = torch.from_numpy(actinos_batch).long().cuda()
        rewards_batch = torch.from_numpy(rewards_batch).float().cuda()
        next_states_batch = torch.from_numpy(next_states_batch).long().cuda()
        dones_batch   = torch.from_numpy(dones_batch).long().cuda()

        return states_batch, actinos_batch, rewards_batch, next_states_batch, dones_batch, mask_state_batch, mask_next_state_batch

    def get(self):
        expdata_states  = torch.from_numpy(self.states_exp[:, :, :]).long().cuda()
        expdata_actinos = torch.from_numpy(self.actions_exp[:, :, :]).long().cuda()
        expdata_rewards = torch.from_numpy(self.rewards_exp[:, :]).float().cuda()
        expdata_next_states = torch.from_numpy(self.next_states_exp [:, :, :]).long().cuda()
        expdata_dones   = torch.from_numpy(self.dones_exp[:, :]).long().cuda()
        
        expdata_mask_state      = self.mask_state[:, :].long().cuda()
        expdata_mask_next_state = self.mask_next_state[:, :].long().cuda()
        
        all_part = (expdata_states, expdata_actinos, expdata_rewards, expdata_next_states, 
                    expdata_dones, expdata_mask_state, expdata_mask_next_state)
        
        return all_part


################################################################################
# DQN Policy
################################################################################
class DQN(object):
    def __init__(self, n_class, Pretrain=True):
        self.eval_net   = LinearTransformer(n_class).cuda()
        self.target_net = LinearTransformer(n_class).cuda()
        
        if Pretrain==True:
            print(f'Load Pretrain from: {Pretrain_ckpt}')
            checkpoint = torch.load(Pretrain_ckpt)
            self.eval_net.load_state_dict(checkpoint['model_state_dict'])
        
        self.eval_net.train()
        self.target_net.train()

        self.agent_buffer  = AgentMemory()   
        self.expert_buffer = ExpertMemory()
        self.optim      = optim.Adam(self.eval_net.parameters(), lr=init_lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[20,40], gamma=0.1)

        self.sigmoid    = nn.Sigmoid()
        self.target_count = 0
        
        self.cnt_update = 0         
        self.mse_val = 0.0
        self.ce_val = 0.0
        self.total_val = 0.0

        self.record_fore_epoch = 0  # Write into list
        self.update_flag = False    # DQN Update


    def choose_action(self, x, target):
        h = self.eval_net.forward_hidden(x)     #　h.shape=(1,50,512)
        y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = self.eval_net.forward_output(h, target)
        
        m = nn.Softmax(dim=-1)
        y_tempo, y_chord, y_barbeat     = m(y_tempo), m(y_chord), m(y_barbeat)
        y_pitch, y_duration, y_velocity = m(y_pitch), m(y_duration), m(y_velocity)
        
        # Take index of max value 
        tempo, chord, barbeat  = torch.argmax(y_tempo, dim=-1), torch.argmax(y_chord, dim=-1), torch.argmax(y_barbeat, dim=-1)
        pitch, duration, velocity = torch.argmax(y_pitch, dim=-1), torch.argmax(y_duration, dim=-1), torch.argmax(y_velocity, dim=-1)

        # action = torch.cat((tempo[:,-1], chord[:,-1], barbeat[:,-1],pitch[:,-1], duration[:,-1], velocity[:,-1]), dim=0)
        # action = action.unsqueeze(0)

        # Action = (N_ACTIONS, 1)
        action = None
        for idx in range(N_ACTIONS):
            tmp = torch.cat((tempo[:,-idx], chord[:,-idx], barbeat[:,-idx],pitch[:,-idx], duration[:,-idx], velocity[:,-idx]), dim=0)
            tmp = tmp.unsqueeze(0)
            if action==None:      
                action = tmp
            else:
                action = torch.cat((action, tmp), dim=0)

        return action
    
    
    def update(self, agent_transition, expert_transition, mask_next_states, update_flag, epoch):
        # Update target network
        if self.target_count  % Target_update  == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict()) 

        self.target_count+=1
        
        # Expert 
        expert_state      = expert_transition['state']
        expert_next_state = expert_transition['nextstate']

        # Eval Network #
        agent_state      = agent_transition['state'].long().cuda()# (batch,state,Feature)
        agent_next_state = agent_transition['nextstate'].long().cuda()
        agent_action     = agent_transition['action'].long().cuda()
        agent_reward     = agent_transition['reward'].float().cuda()
        agent_done       = agent_transition['done'].long().cuda()

        # state對應的q-value
        y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = self.eval_net(agent_state, expert_state)  
        
        qval_tempo     = y_tempo.gather(2, agent_action[:,:,0].unsqueeze(0)).squeeze(0)           # (batch_size,1)
        qval_chord     = y_chord.gather(2, agent_action[:,:,1].unsqueeze(0)).squeeze(0)           # (batch_size,1)
        qval_barbeat   = y_barbeat.gather(2, agent_action[:,:,2].unsqueeze(0)).squeeze(0) 
        qval_pitch     = y_pitch.gather(2, agent_action[:,:,3].unsqueeze(0)).squeeze(0) 
        qval_duration  = y_duration.gather(2, agent_action[:,:,4].unsqueeze(0)).squeeze(0) 
        qval_velocity  = y_velocity.gather(2, agent_action[:,:,5].unsqueeze(0)).squeeze(0) 

        ###　Target Networks ### 
        next_tempo, next_chord, next_barbeat, next_pitch, next_duration, next_velocity = self.target_net(agent_next_state, expert_next_state)

        ### Max Q-Value of next state in each column (each feature) ###
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
        q_targets_chord     = agent_reward + GAMMA * (1-agent_done) * max_next_q_chord
        q_targets_barbeat   = agent_reward + GAMMA * (1-agent_done) * max_next_q_barbeat
        q_targets_pitch     = agent_reward + GAMMA * (1-agent_done) * max_next_q_pitch
        q_targets_duration  = agent_reward + GAMMA * (1-agent_done) * max_next_q_duration
        q_targets_velocity  = agent_reward + GAMMA * (1-agent_done) * max_next_q_velocity
        
        
        tempo_loss      = F.mse_loss(qval_tempo,    q_targets_tempo)
        chord_loss      = F.mse_loss(qval_chord,    q_targets_chord)
        barbeat_loss    = F.mse_loss(qval_barbeat,  q_targets_barbeat)
        pitch_loss      = F.mse_loss(qval_pitch,    q_targets_pitch)
        duration_loss   = F.mse_loss(qval_duration, q_targets_duration)
        velocity_loss   = F.mse_loss(qval_velocity, q_targets_velocity)
        MSEloss = (tempo_loss+ chord_loss+ barbeat_loss + pitch_loss + duration_loss + velocity_loss) / 6

        CE_tempo, CE_chord, CE_barbeat, CE_pitch, CE_duration, CE_velocity = self.eval_net.train_step(agent_state, expert_next_state, mask_next_states)
        CEloss = (CE_tempo+ CE_chord+ CE_barbeat+ CE_pitch+ CE_duration+ CE_velocity) / 6
        total_loss = 0.2*MSEloss + 0.8*CEloss

        self.ce_val  += CEloss.item()
        self.mse_val += MSEloss.item()
        self.total_val += total_loss.item()

        self.optim.zero_grad() 
        total_loss.backward()  
        self.optim.step()
        self.scheduler.step()

        self.cnt_update+=1
        wandb.log({"MSELoss": MSEloss, "CELoss": CEloss, "AgentLoss": total_loss})   
        tqdm.write('Epoch: {}/{}| Episode: {}/{}| MSE_Loss: {:03f}| CE_Loss: {:03f}| TD_Loss: {:03f}'.format(epoch, NUM_SONGS,num,EPISODES, MSEloss, CEloss, total_loss))
        

        if self.record_fore_epoch < epoch:
            self.record_fore_epoch+=1 

            self.mse_val/=self.cnt_update
            self.ce_val/=self.cnt_update
            self.total_val/=self.cnt_update

            first_loss.append(self.mse_val)
            sec_loss.append(self.ce_val)
            global_loss.append(self.total_val)


        if update_flag==True and epoch>=410:
            os.makedirs('./ckpt',exist_ok=True)
            torch.save({
                    'epoch':  epoch,
                    'model_state_dict': self.eval_net.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                }, save_ckpt_path )
            
            wandb.save(save_ckpt_path)


        # Plotting
        if update_flag==True and epoch>=410:
            name = ['MSE', 'CE', 'Global']
            one = [i for i in first_loss]
            two = [j for j in sec_loss]
            three = [k for k in global_loss]
            bi_loss_plot(one, two, three, name, './exp/agent_loss.png')

            # Save recording 
            record = {'Agent':agent_reward, 'first_loss': first_loss, 'sec_loss':sec_loss,' global_loss':  global_loss}
            with open( './ckpt/agent_info.pickle', 'wb') as file:
                pickle.dump(record, file)



if __name__ == '__main__':

    # Load 
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary
    train_data = np.load(path_train_data)

    run = wandb.init(
        project="DQN-RL-Music",
        config={
            "learning_rate": init_lr,
            "epochs": NUM_SONGS,
            'batch_size': batch_size,
            'SEQ_LEN': SEQ_LEN,
            'BUFFER_SIZE': BUFFER_SIZE,
            })

    # Number of each feature(Delete'type'): [56, 135, 18, 87, 18, 25]
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
    
    first_loss  = []
    sec_loss    = []
    global_loss = []

    for epoch in tqdm(range(NUM_SONGS), desc='RL'):  
        # Env.reset
        state_x  = data_x[epoch,:WINDOW_SIZE,:]
        expert_x = data_y[epoch,:,:]

        done = torch.tensor(1).long().cuda() # Agent Done 
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

            done    = torch.tensor(0).long().cuda()
            action  = Agent.choose_action(state_x.unsqueeze(0), Expert_state.unsqueeze(0))
            
            next_state  = torch.cat((state_x[:N_ACTIONS,:], action), dim=0)      # (state_x, action): Same Dim.

            agent_reward = torch.tensor(0.5).float().cuda()  # Temporary reward
            
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

                traj_reward, answer_reward  = Rewarder.update_disc(agent_traj, expert_traj, train=False)
                AgentBuffer.rewards_agent[:,:] = traj_reward
                gene_reward.append(np.sum(AgentBuffer.rewards_agent[:])/300)
                
                state, action, reward, next_state, done = AgentBuffer.sampling(batch_size)
                agent_transition = {'state':state, 'action':action, 'reward':reward, 'nextstate':next_state, 'done':done}
                
                expert_state, expert_action, expert_reward, expert_next_state, expert_done,_, mask_next_states = ExpertBuffer.sampling(batch_size)
                mask_next_states = mask_next_states.cuda()
                expert_transition = {'state':state, 'action':action, 'reward':reward, 
                                     'nextstate':next_state, 'done':expert_done}
                
                update_flag = True
                Agent.update(agent_transition, expert_transition, mask_next_states, update_flag, epoch)

                # sys.stdout.write('Epoch: {}/{} | Episode: {}/{} | TD_Loss: {:03f} \r'.format(epoch,NUM_SONGS,num,EPISODES, loss))
                # sys.stdout.flush()
            else:
                tqdm.write('Epoch: {}/{} | Episode: {}/{} | Buffer_Size:{}\r'.format(epoch,NUM_SONGS,num,EPISODES,AgentBuffer.memory_counter))
                # sys.stdout.write('Epoch: {}/{} | Episode: {}/{} | Buffer_Size:{}\r'.format(epoch,NUM_SONGS,num,EPISODES,AgentBuffer.memory_counter))
                # sys.stdout.flush()

