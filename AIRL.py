import os 
import pickle
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from AIRL_model import LongFormer   # Discriminator


# -- Config -- # 
gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 

# --- modules config --- #
MAX_SEQ_LEN = 1024
D_MODEL = 512
N_LAYER = 6
N_HEAD = 8    
path_exp = 'exp'
N_STATES = 50

Pretrain_ckpt = '/data/Der_CODES/DQN-cp/ckpt/trainloss_22.pt' 

# -- Discriminator -- #
class RewardDiscri(nn.Module):
    def __init__(self, n_token, Pretrain=True):
        super().__init__()
        self.disc_model = LongFormer(n_token).to(device)

        ## Load Pretrain
        if Pretrain==True:
            checkpoint = torch.load(Pretrain_ckpt)
            self.disc_model.load_state_dict(checkpoint['model_state_dict'])

        self.BCE_criterion = nn.BCELoss()
        self.CrossEntropy = nn.CrossEntropyLoss()
        self.avg_score = nn.Sequential(
            nn.Linear(6,1).to(device),
            nn.Sigmoid()
        )
        self.last_unit = nn.Linear(6,1).to(device)

        self.gamma = 0.9
        self.init_lr = 0.1
        self.epoch_disc = 10
        self.batch_size = 10

        self.optim_disc = optim.Adam(self.disc_model.parameters(), lr=self.init_lr)
        self.sched_disc = torch.optim.lr_scheduler.StepLR(self.optim_disc, step_size=10, gamma=0.1)
        self.reward_path   = './ckpt/IRL_reward.pickle'
        self.IRL_ckpt_path = './ckpt/disc_IRL.pt' 

    # -- Sequence reward  -- #  
    def all_forward(self, states_batch, dones_batch, next_states_batch, mask_states_batch, mask_next_states_batch):
        torch.cuda.empty_cache()
        self.disc_model.train()
    
        element_score = self.disc_model(states_batch, mask_states_batch)
        sum_score     = self.avg_score(element_score)       
        return element_score, sum_score      # return -F.logsigmoid(vs)  

    
    # -- For inference -- #
    def calculate_reward(self, states, dones, next_states, mask_states, mask_next_states):
        data_length = states.shape[0]   # Data Length
        pred_val =  torch.ones((data_length, 1))  

        checkpoint = torch.load(self.IRL_ckpt_path)
        self.disc_model.load_state_dict(checkpoint['model_state_dict'])
        self.disc_model.eval()
        with torch.no_grad():
            for idx in range(data_length//self.batch_size): 
                st_idx =  idx * self.batch_size
                ed_idx = (idx+1) * self.batch_size

                states_part = states[st_idx:ed_idx, :, :].long().to(device)
                dones_part  = dones[st_idx:ed_idx, :].long().to(device)
                next_states_part = next_states[st_idx:ed_idx, :, :].long().to(device)
                mask_states_part = mask_states[st_idx:ed_idx, :].long().to(device)
                mask_next_state_part = mask_next_states[st_idx:ed_idx, :].long().to(device)
                
                _, sum_score  = self.all_forward(states_part, dones_part , next_states_part, mask_states_part, mask_next_state_part)
                pred_val[st_idx:ed_idx, :] =sum_score

        return pred_val
    

    def update_disc(self, agent_episode, expert_episode):

        torch.cuda.empty_cache()
        agent_state_action, _, _,  agent_nextstate_action, agent_done = agent_episode
        exp_state_action, _, _, exp_nextstate_action, exp_done, mask_states, mask_next_states = expert_episode

        # rewards = self.calculate_reward(states, dones, next_states, mask_states, mask_next_states)
        agent_label = torch.zeros((self.batch_size, 1)).float().to(device) # Agent Label
        exp_label   = torch.ones((self.batch_size, 1)).float().to(device) # Expert Label
        
        data_length = agent_state_action.shape[0]   # Data Length
        li_loss = []

        print(f'Running IRL.....')
        for epoch in range(self.epoch_disc):
            record_loss=0.0
            for idx in tqdm(range(data_length//self.batch_size), desc='IRL Training'):
                bidx_st =  idx * self.batch_size
                bidx_ed = (idx+1) * self.batch_size
                
                self.optim_disc.zero_grad()
                
                #  -- Expert  -- #
                states_exp_batch = exp_state_action[bidx_st:bidx_ed,: ,:].long().to(device)
                dones_exp_batch  = exp_done[bidx_st:bidx_ed,:].long().to(device)
                next_states_exp_batch = exp_nextstate_action[bidx_st:bidx_ed,: ,:].long().to(device)
                mask_states_batch = mask_states[bidx_st:bidx_ed, :].long().to(device)
                mask_next_state_batch = mask_next_states[bidx_st:bidx_ed, :].long().to(device) 
                
                exp_each_score, exp_sum_logits = self.all_forward(states_exp_batch, dones_exp_batch, next_states_exp_batch, 
                                              mask_states_batch, mask_next_state_batch)
                
                exp_BCELoss  = self.BCE_criterion(exp_sum_logits, exp_label)  
                exp_BCELoss.backward()
                
                #  -- Agent  -- #
                states_batch = agent_state_action[bidx_st:bidx_ed, :, :].long().to(device)
                dones_batch  = agent_done[bidx_st:bidx_ed, :].long().to(device)
                next_states_batch = agent_nextstate_action[bidx_st:bidx_ed, :, :].long().to(device)
                
                agent_each_score, agent_sum_logits = self.all_forward(states_batch.detach(), dones_batch.detach(),next_states_batch.detach(),
                                                 mask_states_batch.detach(), mask_next_state_batch.detach())
                
                # CELoss = self.disc_model.token_forward(states_batch, states_exp_batch, mask_states_batch)
                agent_BCELoss  = self.BCE_criterion(agent_sum_logits, agent_label)   # torch.zeros_like(exp_logits)
                agent_BCELoss.backward()  

                global_loss = agent_BCELoss + exp_BCELoss
                record_loss = record_loss + global_loss.item()
                # loss.backward()            

                self.optim_disc.step()
                self.sched_disc.step()

                # tqdm.write('Epoch: {}/{} | Loss:{}\r'.format(epoch, self.epoch_disc, global_loss))
                
                #  -- Save checkpoint  -- #
                if epoch%5 == 0:
                    torch.save({
                        'epoch': self.epoch_disc,
                        'model_state_dict': self.disc_model.state_dict(),
                        'optimizer_state_dict': self.optim_disc.state_dict(),
                    }, self.IRL_ckpt_path)
                
                    # wandb.save(self.IRL_ckpt_path)

                #  -- wanbd log -- #
                # wandb.log({
                #             'Expert_Loss': exp_BCELoss,
                #             'Agent_Loss': agent_BCELoss,
                #             'Gobal_Loss': global_loss  
                #            })
            
            #  -- Loss Plotting  -- #
            avg_epoch_loss = record_loss / (data_length//self.batch_size)
            tqdm.write('Epoch: {}/{} | Loss:{}\r'.format(epoch, self.epoch_disc, global_loss))

            li_loss.append(avg_epoch_loss)  
        

        #  -- Reward Calculation -- #  
        traj_reward = self.calculate_reward(agent_state_action, agent_done , agent_nextstate_action, 
                                             mask_states, mask_next_states)
        answer_reward = self.calculate_reward(exp_state_action, exp_done, exp_nextstate_action, 
                                            mask_states, mask_next_states)

        # Save recording 
        record = {'Agent':traj_reward , 'Expert':answer_reward , 'loss':li_loss}
        with open(self.reward_path, 'wb') as file:
            pickle.dump(record, file)    

        print(f'=== End Runnning & Saving ===')
    
        return traj_reward, answer_reward
