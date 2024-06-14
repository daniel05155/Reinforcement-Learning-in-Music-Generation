import os 
import pickle
import wandb
from tqdm import tqdm
gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(f'Device: {device}')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from AIRL_model import LongFormer   # Discriminator
from utils import tri_loss_plot, score_plotting

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 

# --- modules config --- #
MAX_SEQ_LEN = 1024
D_MODEL = 512
N_LAYER = 10
N_HEAD = 8    
path_exp = 'exp'
N_STATES = 50

Pretrain_ckpt = '/data/Der_CODES/DQN-cp/ckpt/trainloss_22.pt' 

# -- Discriminator -- #
class RewardDiscri(nn.Module):
    def __init__(self, n_token, Pretrain=True):
        super().__init__()
        self.disc_model = LongFormer(n_token).cuda()

        ## Load Pretrain
        if Pretrain==True:
            checkpoint = torch.load(Pretrain_ckpt)
            self.disc_model.load_state_dict(checkpoint['model_state_dict'])

        self.BCE_criterion = nn.BCELoss()
        self.CrossEntropy = nn.CrossEntropyLoss()
        self.avg_score = nn.Sequential(
            nn.Linear(6,1).cuda(),
            nn.Sigmoid()
        )
        self.last_unit = nn.Linear(6,1).cuda()

        self.init_lr = 0.001
        self.epoch_disc = 10
        self.batch_size = 100

        self.optim_disc = optim.Adam(self.disc_model.parameters(), lr=self.init_lr)
        self.sched_disc = torch.optim.lr_scheduler.StepLR(self.optim_disc, step_size=10, gamma=0.1)
        self.reward_path   = './exp/IRL_reward.pickle'
        self.IRL_ckpt_path = './ckpt/disc_IRL.pt' 

    # -- Sequence reward  -- #  
    def all_forward(self, states_batch, dones_batch, next_states_batch, mask_states_batch, mask_next_states_batch):
        # torch.cuda.empty_cache()
        self.disc_model.train()
        seq_score = self.disc_model(states_batch, mask_states_batch)
       
        return seq_score   # return -F.logsigmoid(vs)  

    
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

                states_part = states[st_idx:ed_idx, :, :].long().cuda()
                dones_part  = dones[st_idx:ed_idx, :].long().cuda()
                next_states_part = next_states[st_idx:ed_idx, :, :].long().cuda()
                mask_states_part = mask_states[st_idx:ed_idx, :].long().cuda()
                mask_next_state_part = mask_next_states[st_idx:ed_idx, :].long().cuda()
                
                seq_score  = self.all_forward(states_part, dones_part , next_states_part, mask_states_part, mask_next_state_part)
                pred_val[st_idx:ed_idx, :] = seq_score

        return pred_val
    

    def update_disc(self, agent_episode, expert_episode, train=True):
        # torch.cuda.empty_cache()
        agent_state_action, _, _,  agent_nextstate_action, agent_done = agent_episode
        exp_state_action, _, _, exp_nextstate_action, exp_done, mask_states, mask_next_states = expert_episode

        agent_label = torch.zeros((self.batch_size, 1)).float().cuda() # Agent Label
        exp_label   = torch.ones((self.batch_size, 1)).float().cuda()  # Expert Label
        
        data_length = agent_state_action.shape[0]   # Data Length
        li_loss  = []
        exp_loss = []
        agent_loss = []
        ce_loss  = []

        if train==True:
            for epoch in range(self.epoch_disc):
                record_loss = 0.0
                first_exp_loss = 0.0
                sec_agent_loss = 0.0
                third_ce_loss    = 0.0

                for idx in tqdm(range(data_length//self.batch_size), desc='IRL Training'):
                    bidx_st =  idx * self.batch_size
                    bidx_ed = (idx+1) * self.batch_size
                    
                    self.optim_disc.zero_grad()
                    
                    #  -- Expert  -- #
                    states_exp_batch = exp_state_action[bidx_st:bidx_ed,: ,:].long().cuda() 
                    dones_exp_batch  = exp_done[bidx_st:bidx_ed,:].long().cuda() 
                    next_states_exp_batch = exp_nextstate_action[bidx_st:bidx_ed,: ,:].long().cuda() 
                    mask_states_batch = mask_states[bidx_st:bidx_ed, :].long().cuda() 
                    mask_next_state_batch = mask_next_states[bidx_st:bidx_ed, :].long().cuda()  
                    
                    exp_logits = self.all_forward(states_exp_batch, dones_exp_batch, next_states_exp_batch, 
                                                mask_states_batch, mask_next_state_batch)

                    exp_BCELoss  = self.BCE_criterion(exp_logits, exp_label)  
                    # exp_BCELoss.backward()
                    
                    #  -- Agent  -- #
                    states_batch = agent_state_action[bidx_st:bidx_ed, :, :].long().cuda() 
                    dones_batch  = agent_done[bidx_st:bidx_ed, :].long().cuda() 
                    next_states_batch = agent_nextstate_action[bidx_st:bidx_ed, :, :].long().cuda() 
                    
                    CE_loss  = self.disc_model.token_forward(states_batch.detach(), states_exp_batch.detach(), mask_states_batch.detach())
                    agent_logits = self.all_forward(states_batch.detach(), dones_batch.detach(),next_states_batch.detach(),
                                                    mask_states_batch.detach(), mask_next_state_batch.detach())
                    
                    agent_BCELoss  = self.BCE_criterion(agent_logits, agent_label)   # torch.zeros_like(exp_logits)
                    # agent_BCELoss.backward()  

                    global_loss = exp_BCELoss +(agent_BCELoss+ CE_loss)
                    global_loss.backward()           
                    
                    self.optim_disc.step()
                    self.sched_disc.step()
                    
                    first_exp_loss = first_exp_loss + exp_BCELoss.item()
                    sec_agent_loss = sec_agent_loss + agent_BCELoss.item()
                    third_ce_loss  = third_ce_loss + CE_loss.item()

                    record_loss    = record_loss + global_loss.item()
                    
                    #  -- Save checkpoint  -- #
                    if epoch%5 == 0:
                        torch.save({
                            'epoch': self.epoch_disc,
                            'model_state_dict': self.disc_model.state_dict(),
                            'optimizer_state_dict': self.optim_disc.state_dict(),
                        }, self.IRL_ckpt_path)
                        wandb.save('disc_IRL.pt')

                    #  -- wandb log -- #
                    wandb.log({
                                'Expert_Loss': exp_BCELoss,
                                'Agent_Loss': agent_BCELoss,
                                'Gobal_Loss': global_loss  
                            })
                
                #  -- Loss Plotting  -- #
                exp_avg_loss    = first_exp_loss / (data_length//self.batch_size)
                agent_avg_loss  = sec_agent_loss / (data_length//self.batch_size)
                ce_avg_loss     = third_ce_loss / (data_length//self.batch_size)
                avg_record_loss = record_loss / (data_length//self.batch_size)
                
                exp_loss.append(exp_avg_loss)  
                agent_loss.append(agent_avg_loss)
                ce_loss.append(ce_avg_loss)
                li_loss.append(avg_record_loss)
                
                tqdm.write('Epoch:{}/{}| Exp_L:{}| Gene_L:{}| CE_L:{}| Global_L:{}\r'.format(epoch, self.epoch_disc, 
                                                                                    exp_avg_loss, agent_avg_loss, ce_avg_loss, global_loss))

        traj_reward = self.calculate_reward(agent_state_action, agent_done , agent_nextstate_action, 
                                            mask_states, mask_next_states)
        answer_reward = self.calculate_reward(exp_state_action, exp_done, exp_nextstate_action, 
                                            mask_states, mask_next_states)
        
        #  -- Plotting  -- #
        if len(exp_loss)!=0:
            save_loss_path = './exp/IRL_loss.png'
            name = ['Expert', 'Agent', 'CE', 'Total']
            tri_loss_plot(exp_loss, agent_loss, ce_loss, li_loss, name, save_loss_path)

        gene    = [i.item() for i in traj_reward]
        answer  = [j.item() for j in answer_reward]
        save_distri_path = './exp/score.png'
        score_plotting(gene, answer, save_distri_path)

        #  -- Save recording   -- #
        record = {'Agent':traj_reward , 'Expert':answer_reward}
        with open(self.reward_path, 'wb') as file:
            pickle.dump(record, file)    

        # print(f'=== End IRL Runnning & Saving ===\n')
        return traj_reward, answer_reward
