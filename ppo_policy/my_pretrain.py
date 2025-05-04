###--- Pretrained- Linear Transformer  ---###
import os
import numpy as np
import time
import argparse
from datetime import datetime, timedelta
import sys
import pickle
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from model import LinearTransformer, network_paras, LongFormer 
from utils_file import write_config_log
from config import device, datapath, MaxSeqLen     

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 

###--- Our Dataset  ---###
train_datapath = './dataset/our_dataset.pickle'   

###--- Trainable Parameter ---###
BATCH_SIZE  = 12
NUM_EPOCH   = 1000
Init_lr     = 0.01


def pretrain(model, my_dataset, optimizer, scheduler, flag, config_path, ckpt_path):

    # --- Load Dataset --- #
    train_x = my_dataset['train_x']
    train_y = my_dataset['train_y']
    mask    = my_dataset['mask']
    num_batch = len(my_dataset['train_x']) // BATCH_SIZE

    print('    num_batch:', num_batch)
    print('    train_x:', train_x.shape)
    print('    train_y:', train_y.shape)
    print('    train_mask:', mask.shape)

    if flag==True:
        purpose = 'Pretrain with Longformer for reward model'
    else:
        purpose = 'Pretrain with Linearformer for agent.'

    len_train_x = train_x.shape
    len_train_y = train_y.shape
    write_config_log(config_path, purpose, len_train_x, len_train_y, NUM_EPOCH, BATCH_SIZE, Init_lr)

    # run
    start_time = time.time()
    record_loss = []
    for epoch in range(NUM_EPOCH):
        acc_loss = 0   
        arr_losses = np.zeros(6)  

        # num_batch 
        for bidx in range(num_batch): 
            # batch index
            bidx_st = BATCH_SIZE * bidx
            bidx_ed = BATCH_SIZE * (bidx + 1)

            # unpack batch data
            batch_x = train_x[bidx_st:bidx_ed]
            batch_y = train_y[bidx_st:bidx_ed]
            batch_mask = mask[bidx_st:bidx_ed]
            
            # to tensor
            batch_x = torch.from_numpy(batch_x).long().to(device)
            batch_y = torch.from_numpy(batch_y).long().to(device)
            batch_mask = torch.from_numpy(batch_mask).float().to(device)

            # run
            losses = model.train_step(batch_x, batch_y, batch_mask)
            CELoss = (losses[0]+losses[1]+losses[2]+losses[3]+losses[4]+losses[5]) / 6
        
            # Update
            optimizer.zero_grad()
            CELoss.backward()
            optimizer.step()
            scheduler.step()

            # Accuracy
            arr_losses += np.array([l.item() for l in losses])
            acc_loss += CELoss.item()

            #  -- wandb log -- #
            wandb.log({ 
                    'Tempo_Loss':losses[0], 'Chord_Loss':losses[1], 'Bar_Loss':losses[2], 
                    'Pitch_Loss':losses[3], 'Duration_Loss':losses[4], 'Velocity_Loss':losses[5],
                    'Total_Loss':CELoss,
                })

            # print
            sys.stdout.write('Epoch: {}/{} | Batch:{}/{} | Loss: {:06f} | {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\r'.format(
                epoch, NUM_EPOCH, bidx, num_batch, CELoss, losses[0], losses[1], losses[2], losses[3], losses[4], losses[5]))
            sys.stdout.flush()

        # epoch loss
        runtime = time.time() - start_time
        epoch_loss = acc_loss / num_batch
        record_loss.append(CELoss.item())
        arr_losses = arr_losses / num_batch

        # print('------------------------------------\n')
        # print('Epoch: {}/{} | Avg_Loss: {} | Time: {}'.format(epoch, NUM_EPOCH, epoch_loss, str(timedelta(seconds=runtime))))
        # each_loss_str = '{:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\r'.format(
        #       arr_losses[0], arr_losses[1], arr_losses[2], arr_losses[3], arr_losses[4], arr_losses[5])
        # print('Each_Loss ->', each_loss_str)

        # -- Save Checkpoint -- #
        if epoch%10==0:
            torch.save(model.state_dict(), os.path.join(ckpt_path, 'pretrain_best.pth'))
            # torch.save({
            #         'model_state': model.state_dict(),
            #         'opt_state': optimizer.state_dict(),
            #     }, os.path.join(ckpt_path, 'pretrain_best.pth'))

    plot_path = os.path.join(ckpt_path, 'pretrain_loss.png')
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.plot(record_loss, label='CE_Loss')
    plt.legend()
    plt.savefig(plot_path)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward_pretrain', help='Reward Model Pretrain', action='store_true', default=False)
    args = parser.parse_args()

    # run = wandb.init(
    #     project="Pretrain-PPO-Music",
    #     config={
    #         "learning_rate": Init_lr,
    #         "epochs": NUM_EPOCH,
    #         'batch_size': BATCH_SIZE,
    #         'SEQ_LEN': MaxSeqLen
    #         })
    
    ##### File Operations #####
    # Experiment Directory 
    exp_name =  datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    exp_dir = os.path.join('./Exp-Pretrain', exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save Model Path 
    model_save_path = os.path.join('./Exp-Pretrain', exp_name, 'model')
    os.makedirs(model_save_path, exist_ok=True)

    # Config Directory 
    log_dir = os.path.join(exp_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    
    # Write config 
    config_path = os.path.join(log_dir, 'config_log.txt')
    ############################## 

    ##### Load Data & Model #####
    with open(datapath['path_dictionary'] , 'rb') as f:
        dictionary = pickle.load(f)
    event2word, word2event = dictionary

    with open(train_datapath, 'rb') as f:
        my_dataset = pickle.load(f)

    num_token = []
    for etype in event2word.keys():
        num_token.append(len(dictionary[0][etype])+1)  # +1:Cause pad_word
    print('Num of token class >>', num_token)

    if args.reward_pretrain:
        model = LongFormer(num_token).to(device)
        model.train()
        print('Reward Model Pretraining...')
        flag = True
    else:
        model = LinearTransformer(num_token, is_training=True).to(device)
        model.train()
        print('Agent Pretraining...')
        flag=False
    
    n_parameters = network_paras(model)
    print('Model_parameters: {:,}'.format(n_parameters))

    optimizer = optim.Adam(model.parameters(), lr=Init_lr )
    scheduler  = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, NUM_EPOCH], gamma=0.1)

    ##### run pretraining #####
    pretrain(model, my_dataset, optimizer, scheduler, flag, config_path, ckpt_path=model_save_path)

if __name__ == '__main__':
    main()
    