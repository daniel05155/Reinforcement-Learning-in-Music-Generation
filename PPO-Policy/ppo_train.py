import sys
import os
import time
import wandb
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from model import Actor_Transformer, Critic_Transformer, LongFormer
from tqdm import tqdm
from config import device, datapath, Load_Pretrain

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 

###--- data ---###
path_data_root = '/data/dataset_Pop1K7/representations/uncond/cp/ailab17k_from-scratch_cp'
path_train_data = os.path.join(path_data_root, 'train_data_linear.npz')
path_dictionary = os.path.join(path_data_root, 'dictionary.pkl')

###--- Checkpoint path  ---###
Pretrain_agent_ckpt      = './ckpt/pretrain_actor.pth' 
Pretrain_eval_model_ckpt = './ckpt/pretrain_eval.pth'
save_ckpt_path           = './ckpt/ppo_best.pt'    

################################################################################
# Model config 
################################################################################ 
Target_update = 50            # Update freqency of target net
EPSILON = 0.9                 # greedy policy
GAMMA = 0.95                  # reward discount
PPO_STEPS = 10       
PPO_CLIP = 0.2
DISCOUNT_FACTOR = 0.99

################################################################################
# Training config 
################################################################################
NUM_SONGS = 1000          # NUM_EPOCH
EPISODES = 30
SEQ_LEN = 1000
N_STATES = 50
N_FEATURES = 6          # Features in each state
N_ACTIONS = 25

WINDOW_SIZE = 50        # STATE SIZE    
BUFFER_SIZE = EPISODES    # MEMORY CAPACITY
ACTION_DIM  = 6
NUM_ACTION  = 25        # Agent prediction

batch_size = 6
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
        self.states_agent       = np.zeros((BUFFER_SIZE, N_STATES, N_FEATURES))  # (buffer, 50, 6)
        self.value_agent        = np.zeros((BUFFER_SIZE, 1))    # (buffer, 1)

        self.actions_agent      = np.zeros((BUFFER_SIZE, N_ACTIONS, N_FEATURES))  
        self.log_actions_agent  = np.zeros((BUFFER_SIZE, N_ACTIONS, N_FEATURES))
        
        self.rewards_agent      = np.zeros((BUFFER_SIZE, 1))                     # (buffer, 50, 1)
        self.next_states_agent  = np.zeros((BUFFER_SIZE, N_STATES, N_FEATURES)) 
        self.dones_agent        = np.zeros((BUFFER_SIZE, 1))
        self.memory_counter = 0             # Count size of buffer

    def store_transition(self, state, action, log_action, value_state, reward, next_state, done):    
        index = self.memory_counter % BUFFER_SIZE   #
        # tensor -> np.array
        state       = state.detach().cpu().numpy()
        action      = action.detach().cpu().numpy()
        log_action  = log_action.detach().cpu().numpy()
        value_state = value_state.detach().cpu().numpy()   
        reward      = reward.detach().cpu().numpy()
        next_state  = next_state.detach().cpu().numpy()
        done        = done.detach().cpu().numpy()

        self.states_agent[index,:,:]      = state
        self.value_agent[index,:]      = value_state 

        self.actions_agent[index,:,:]     = action
        self.log_actions_agent[index,:,:] = log_action  
       
        self.rewards_agent[index,:]       = reward
        self.next_states_agent[index,:,:] = next_state
        self.dones_agent[index,:]         = done
        
        self.memory_counter += 1                                                

    # -- Sampling the number of batch size -- #
    def sampling(self, batch_size):
        sample_idx  = np.random.choice(BUFFER_SIZE, batch_size) 

        states_batch      = self.states_agent[sample_idx, :, :]
        actinos_batch     = self.actions_agent[sample_idx, :, :]
        log_actions_batch = self.log_actions_agent[sample_idx, :, :]

        values_batch      = self.value_agent[sample_idx, :]  
        rewards_batch     = self.rewards_agent[sample_idx, :]
        
        next_states_batch = self.next_states_agent[sample_idx, :, :]
        dones_batch       = self.dones_agent[sample_idx, :]

        # np.array -> tensor
        states_batch        = torch.from_numpy(states_batch).long().to(device)
        actinos_batch       = torch.from_numpy(actinos_batch).long().to(device)
        log_actions_batch   = torch.from_numpy(log_actions_batch).long().to(device)
        values_batch        = torch.from_numpy(values_batch).float()
        rewards_batch       = torch.from_numpy(rewards_batch).float()
    
        next_states_batch   = torch.from_numpy(next_states_batch).long().to(device)
        dones_batch         = torch.from_numpy(dones_batch).long().to(device)

        return states_batch, actinos_batch, log_actions_batch, values_batch, rewards_batch, next_states_batch, dones_batch
    
    # -- Take agent data in each episode -- #
    def get(self):
        states_agentdata      = torch.from_numpy(self.states_agent[:, :, :]).long().to(device)
        actinos_agentdata     = torch.from_numpy(self.actions_agent[:, :, :]).long().to(device)
        log_actions_agentdata = torch.from_numpy(self.log_actions_agent[:, :, :]).long().to(device)
        values_agentdata      = torch.from_numpy(self.value_agent[:, ...]).float().to(device)
        rewards_agentdata     = torch.from_numpy(self.rewards_agent[:, :]).float().to(device)
        next_states_agentdata = torch.from_numpy(self.next_states_agent[:, :, :]).long().to(device)
        dones_agentdata       = torch.from_numpy(self.dones_agent[:, :]).long().to(device)

        all_agent_part = {'states':states_agentdata, 'actions':actinos_agentdata, 'log_actions':log_actions_agentdata,
                        'values':values_agentdata, 'rewards':rewards_agentdata, 'next_states':next_states_agentdata, 'dones':dones_agentdata}
        
        return all_agent_part

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
        done    = done.detach().cpu().numpy()

        self.states_exp[index, :, :]      = state
        self.actions_exp[index, :, :]     = action
        self.rewards_exp[index, :]        = reward
        self.next_states_exp[index, :, :] = next_state
        self.dones_exp[index, :]          = done

        self.mask_state[index, :]         = mask_state          # torch type
        self.mask_next_state[index, :]    = mask_next_state     # torch type
        self.memory_counter += 1   

    def sampling(self, batch_size):
        sample_idx  = np.random.choice(BUFFER_SIZE, batch_size) 

        states_batch    = self.states_exp[sample_idx, :, :]
        actinos_batch   = self.actions_exp[sample_idx, :, :]
        rewards_batch   = self.rewards_exp[sample_idx, :]
        next_states_batch = self.next_states_exp[sample_idx, :, :]
        dones_batch     = self.dones_exp[sample_idx, :]

        mask_state_batch = self.mask_state[sample_idx, :]             # torch type
        mask_next_state_batch  = self.mask_next_state[sample_idx, :]  # torch type

        # np.array -> tensor
        states_batch  = torch.from_numpy(states_batch).long().to(device)
        actinos_batch = torch.from_numpy(actinos_batch).long().to(device)
        rewards_batch = torch.from_numpy(rewards_batch).float().to(device)
        next_states_batch = torch.from_numpy(next_states_batch).long().to(device)
        dones_batch   = torch.from_numpy(dones_batch).long().to(device)

        return states_batch, actinos_batch, rewards_batch, next_states_batch, dones_batch, mask_state_batch, mask_next_state_batch


    def get(self):
        expdata_states  = torch.from_numpy(self.states_exp[:, :, :]).long().to(device)
        expdata_actinos = torch.from_numpy(self.actions_exp[:, :, :]).long().to(device)
        expdata_rewards = torch.from_numpy(self.rewards_exp[:, :]).float().to(device)
        expdata_next_states = torch.from_numpy(self.next_states_exp [:, :, :]).long().to(device)
        # expdata_dones   = torch.from_numpy(self.dones_exp[:, :]).long().to(device)
        
        expdata_mask_state      = self.mask_state[:, :].long().to(device)
        expdata_mask_next_state = self.mask_next_state[:, :].long().to(device)

        expert_all = {'states':expdata_states, 'actions':expdata_actinos,'rewards': expdata_rewards, 
                     'next_states':expdata_next_states, 'mask_state':expdata_mask_state, 'mask_next_state':expdata_mask_next_state}
        return expert_all

################################################################################
# PPO Policy
################################################################################
class PPO(object):
    def __init__(self, n_class, Pretrain=Load_Pretrain):
        self.actor_net  = Actor_Transformer(n_class).to(device)
        self.critic_net = Critic_Transformer(n_class).to(device)
        self.eval_net   = LongFormer(n_class).to(device)

        if Pretrain==True:
            print(f'Load pretrain From: {Pretrain_agent_ckpt}')
            agent_ckpt = torch.load(Pretrain_agent_ckpt, map_location=device)
            self.actor_net.load_state_dict(agent_ckpt, strict=False)    # , strict=False
            # self.critic_net.load_state_dict(agent_ckpt, strict=False)

            print(f'Reward Model From: {Pretrain_eval_model_ckpt}')
            eval_ckpt = torch.load(Pretrain_eval_model_ckpt, map_location=device)
            self.eval_net.load_state_dict(eval_ckpt, strict=False)

        self.actor_net.train()
        self.critic_net.train()
        # self.eval_net.train()

        self.agent_buffer  = AgentMemory()   
        self.expert_buffer = ExpertMemory()
        self.actor_optim   = optim.Adam(self.actor_net.parameters(), lr=init_lr)
        # self.scheduler     = optim.lr_scheduler.MultiStepLR(self.actor_net, milestones=[20,40], gamma=0.1)
        self.critic_optim  = optim.Adam(self.critic_net.parameters(), lr=init_lr) 
        
        self.target_count = 0
        self.cnt_update = 0         
        self.mse_val = 0.0
        self.ce_val  = 0.0
        self.total_val = 0.0

        self.record_for_epoch = 0       # Write into list

    def choose_action(self, state_x):
        h = self.actor_net.forward_hidden(state_x)     #　h.shape=(1,50,512)
        y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = self.actor_net.forward_output(h)

        value_state = self.actor_net.value_funtion(h.squeeze(0))   
        value_state = value_state.squeeze(0)            # (50, 6)
        # value_state = torch.sum(value_state)/N_STATES

        m = nn.Softmax(dim=-1)
        y_tempo, y_chord, y_barbeat     = m(y_tempo), m(y_chord), m(y_barbeat)
        y_pitch, y_duration, y_velocity = m(y_pitch), m(y_duration), m(y_velocity)

        token_attri = [y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity]
    
        # -- Take index of max value  -- #
        tempo, chord, barbeat  = torch.argmax(y_tempo, dim=-1), torch.argmax(y_chord, dim=-1), torch.argmax(y_barbeat, dim=-1)
        pitch, duration, velocity = torch.argmax(y_pitch, dim=-1), torch.argmax(y_duration, dim=-1), torch.argmax(y_velocity, dim=-1)
        
        action = None
        for idx in range(1, N_ACTIONS+1):
            tmp = torch.cat((tempo[:,-idx], chord[:,-idx], barbeat[:,-idx], pitch[:,-idx], duration[:,-idx], velocity[:,-idx]), dim=0)
            tmp = tmp.unsqueeze(0)
            prob_act = torch.cat((  y_tempo[0, -idx, tempo[:,idx]], 
                                    y_chord[0, -idx, chord[:,idx]], 
                                    y_barbeat[0, -idx, barbeat[:,-idx]],
                                    y_pitch[0, -idx, pitch[:,-idx]],
                                    y_duration[0, -idx, duration[:,-idx]],
                                    y_velocity[0, -idx, velocity[:,-idx]]
                                ), dim=0)
            log_prob = torch.log(prob_act)
            log_prob = log_prob.unsqueeze(0)

            if action==None:      
                action = tmp
                log_prob_res = log_prob
            else:
                action = torch.cat((action, tmp), dim=0)
                log_prob_res = torch.cat((log_prob_res, log_prob), dim=0)
            
        return action, log_prob_res

    # Update Policy 
    def select_udpate(self, state_x):
        h = self.actor_net.forward_hidden(state_x)     #　h.shape=(Batch,50,512)
        y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = self.actor_net.forward_output(h)

        # value_state = self.actor_net.value_funtion(h)   # Error
        value_state = self.critic_net.value_produce(state_x)

        m = nn.Softmax(dim=-1)
        y_tempo, y_chord, y_barbeat     = m(y_tempo), m(y_chord), m(y_barbeat)
        y_pitch, y_duration, y_velocity = m(y_pitch), m(y_duration), m(y_velocity)
        token_attri = [y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity]
    
        # -- Take index of max value  -- #
        tempo, chord, barbeat  = torch.argmax(y_tempo, dim=-1), torch.argmax(y_chord, dim=-1), torch.argmax(y_barbeat, dim=-1)
        pitch, duration, velocity = torch.argmax(y_pitch, dim=-1), torch.argmax(y_duration, dim=-1), torch.argmax(y_velocity, dim=-1)
        
        result_action = None
        result_logprob_action = None

        for batch in range(state_x.shape[0]):
            action = None
            log_prob_res = None
            for idx in range(1, N_ACTIONS+1): 
                #  action
                tmp = torch.hstack((tempo[batch, -idx], chord[batch,-idx], barbeat[batch,-idx], pitch[batch,-idx], duration[batch,-idx], velocity[batch,-idx]))
                tmp = tmp.unsqueeze(0)

                # log prob.of action
                prob_act = torch.hstack((   y_tempo[batch, -idx, tempo[batch, -idx]], 
                                            y_chord[batch, -idx, chord[batch, -idx]], 
                                            y_barbeat[batch,-idx, barbeat[batch,-idx]],
                                            y_pitch[batch,  -idx, pitch[batch,-idx]],
                                            y_duration[batch, -idx, duration[batch,-idx]],
                                            y_velocity[batch, -idx, velocity[batch,-idx]]
                                        ))
                log_prob = torch.log(prob_act)
                log_prob = log_prob.unsqueeze(0)

                if action==None:      
                    action = tmp
                    log_prob_res = log_prob
                else:
                    action = torch.cat((action, tmp), dim=0)
                    log_prob_res = torch.cat((log_prob_res, log_prob), dim=0)

            ### Batch size
            if result_action==None:      
                result_action = action.unsqueeze(0)
                result_logprob_action = log_prob_res.unsqueeze(0)
            else:
                result_action = torch.cat((result_action, action.unsqueeze(0)), dim=0)
                result_logprob_action = torch.cat((result_logprob_action, log_prob_res.unsqueeze(0)), dim=0)
            # print(f'{result_action.shape}, {result_logprob_action.shape}')
        return action, log_prob_res, value_state
    
    def calculate_returns(self, rewards, discount_factor, normalize=True):
        returns = []
        R = 0
        for reward_val in (rewards):
            R = reward_val + R * discount_factor
            returns.insert(0, R) 
        returns = torch.tensor(returns).unsqueeze(1).to(device)
        if normalize:
            returns = (returns - returns.mean()) / returns.std()
        return returns
    
    def calculate_advantages(self, returns, values, normalize=True):
        advantages = returns - values
        if normalize:
            advantages = (advantages - advantages.mean()) / advantages.std()
        return advantages

    def update_policy(self, ppo_steps, ppo_clip, advantages, returns):
        # -- Sampling from agent buffer -- #
        # states, actinos, log_actions, values, rewards, next_states, dones = AgentBuffer.sampling(batch_size) 
        # agent_transition = {'state':states, 'action':actinos, 'log_actions':log_actions, 
        #                     'reward':rewards, 'value':values, 'nextstate':next_states, 'done':dones}
        agent_all = AgentBuffer.get()
        expert_all = ExpertBuffer.get()
        
        log_actions = agent_all['log_actions'].detach()
        advantages  = advantages.to(device).detach()
        returns     = returns.to(device).detach()
        
        # Actor_Update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        for epoch in tqdm(range(ppo_steps), desc='Actor_Update'):
            # Get new log prob of actions from all input states
            input_state = agent_all['states']

            # new_action, new_log_prob_action, value_pred  = Agent.choose_action(input_state)  # Error
            new_action, new_log_prob_action, value_pred = Agent.select_udpate(input_state)  
            
            ## Actor update ## 
            log_actions_val = log_actions.squeeze(0)
            policy_ratio = (new_log_prob_action - log_actions_val).exp()

            policy_loss_1 = 0.2 * advantages
            policy_loss_2 = torch.clamp(policy_ratio, min = 1.0-ppo_clip, max = 1.0+ppo_clip) * (advantages.unsqueeze(2))

            ## Calculate actor loss ## 
            policy_loss_1  = policy_loss_1.unsqueeze(2)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()    # Take max
            
            ce_tuple = self.actor_net.train_step(input_state, expert_all['states'], expert_all['mask_state'])
            ce_loss = (ce_tuple[0]+ce_tuple[1]+ce_tuple[2]+ce_tuple[3]+ce_tuple[4]+ce_tuple[5])/6
            actor_loss = policy_loss + ce_loss 

            value_loss = F.mse_loss(returns, value_pred).sum()   

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            total_policy_loss+=actor_loss.item()

            self.critic_optim.zero_grad()
            value_loss.backward()           
            self.critic_optim.step()
            total_value_loss+=value_loss.item()
            
            tqdm.write('Update_PPO:{}/{}| Actor_loss:{:03f}| Critic_loss:{:.03f}'.format(epoch,ppo_steps,actor_loss,value_loss))
            # wandb.log({"Actor_Loss": actor_loss,"Value_Loss": value_loss})   

        return total_policy_loss/ppo_steps

if __name__ == '__main__':
    
    # run = wandb.init(
    #     project="Run-Music-on-PPO",
    #     config={
    #         "learning_rate": init_lr,
    #         "epochs": NUM_SONGS,
    #         'batch_size': batch_size,
    #         'SEQ_LEN': SEQ_LEN,
    #         'BUFFER_SIZE': BUFFER_SIZE,
    #         })

    ##### Load Data #####
    with open(datapath['path_dictionary'] , 'rb') as f:
        dictionary = pickle.load(f)
    event2word, word2event = dictionary

    with open(datapath['path_train_data'], 'rb') as f:
        my_dataset = pickle.load(f)
    
    token_class = []
    for key in event2word.keys():
            token_class.append(len(dictionary[0][key]))
    
    ###--- RL Policy ---###
    Agent = PPO(token_class, Pretrain=Load_Pretrain)

    ###--- Unpack Training data ---###
    train_x = my_dataset['train_x']             # (B,1200,6)
    train_y = my_dataset['train_y']             # (B,1200,6)    
    train_mask =  my_dataset['mask']            # (B,1200)
    num_batch  = len(my_dataset['train_x']) // batch_size
    
    train_x = torch.from_numpy(train_x).to(device)
    train_y = torch.from_numpy(train_y).to(device)
    train_mask = torch.from_numpy(train_mask).to(device)

    gene_reward = []
    policy_loss_list = [] 
    value_loss_list = []

    for epoch in tqdm(range(NUM_SONGS), desc='RL'):  
        # Env.reset()
        state_x  = train_x[epoch,:WINDOW_SIZE,:]
        expert_x = train_y[epoch,:,:]
        done = torch.tensor(1).long().to(device)    # Agent Done 
        cnt_reward = 0
        
        ###--- Buffer ---###
        AgentBuffer  = AgentMemory()
        ExpertBuffer = ExpertMemory()

        for num in range(0, EPISODES): 
            Expert_state      = expert_x[num: num+WINDOW_SIZE]    
            Expert_next_state = expert_x[num+50: num+50+WINDOW_SIZE]
            Expert_action     = expert_x[num+WINDOW_SIZE]                       # expert_x[num+WINDOW_SIZE]
            Expert_reward     = torch.tensor(1.0).float().to(device)         
            Expert_done       = torch.tensor(0).long().to(device)          
            Expert_mask_state = train_mask[epoch, num: num+WINDOW_SIZE]          # State mask
            Expert_mask_nextstate = train_mask[epoch, num+1: num+1+WINDOW_SIZE]  # Next state mask

            done = torch.tensor(0).long().to(device)
            action, log_prob_res = Agent.choose_action(state_x.unsqueeze(0))

            next_state = torch.cat((state_x[:N_ACTIONS,:], action), dim=0)     # (state_x, action): Same Dim.
            # agent_reward = torch.tensor(0.5).float().to(device)                     # Temporary reward
            
            # -- Turn to next state  -- #  
            state_x = next_state   

            # -- value,reward -- #  
            value_state = Agent.critic_net.value_produce(state_x.unsqueeze(0))
            agent_reward = Agent.eval_net.token_forward(state_x.unsqueeze(0), Expert_state, Expert_mask_state.unsqueeze(0))
        
            # -- store_transition -- #  
            AgentBuffer.store_transition(state_x, action, log_prob_res, value_state, agent_reward, next_state, done)            
            ExpertBuffer.store_transition(Expert_state, action, Expert_reward, Expert_next_state, 
                                          Expert_done, Expert_mask_state, Expert_mask_nextstate)            
        
        agent_all = AgentBuffer.get()
        rewards_episode = agent_all['rewards']
        values_episode  = agent_all['values']

        returns_result = Agent.calculate_returns(rewards_episode, DISCOUNT_FACTOR)
        advantages     = Agent.calculate_advantages(returns_result, values_episode)   
        
        # -- Update Policy -- #
        policy_avg_loss = Agent.update_policy(PPO_STEPS, PPO_CLIP, advantages, returns_result)
        policy_loss_list.append(policy_avg_loss)
        tqdm.write('Overall Progress...Epoch:{}/{} | Episode: {}/{} |\r'.format(epoch,NUM_SONGS,num,EPISODES))
        # sys.stdout.write('Epoch: {}/{}| Policy_Loss: {:03f}| Value_Loss: {:03f}\r'.format(epoch, NUM_SONGS, policy_loss, value_loss))
        # sys.stdout.flush()

        if epoch%5==0:
            os.makedirs('./ckpt',exist_ok=True)
            torch.save(Agent.actor_net.state_dict(), save_ckpt_path )

        if epoch%20==0:
            record_loss = {'policy_loss': policy_loss_list}
            with open('./ckpt/policy_loss.pickle', 'wb') as file:
                pickle.dump(record_loss, file)

            # -- Plotting -- #
            plt.figure()
            plt.plot(policy_loss_list)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Policy Loss')
            plt.savefig('./Loss_policy.png')
            
