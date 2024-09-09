import numpy as np
import os 
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from config import datapath, MaxSeqLen


def process_data():

    # print(f'Loading from...{datapath['path_train_data']}')
    with open(datapath['path_init_data'], 'rb') as file:
        dataset = pickle.load(file)
    print("Number of midis:", len(dataset))

    with open(datapath['path_dictionary'], 'rb') as f:
        dictionary = pickle.load(f)
    event2word, word2event = dictionary

    num_token=[]
    pad_word = []      
    for etype in event2word.keys():
        num_token.append(len(dictionary[0][etype]))
    print(f'num_token: {num_token}')
    
    #  padding word 
    pad_word = [0 for i in range(len(num_token))]    
    print(f'pad_word: {pad_word}')
    
    # --- Creat Custom Dataset --- #
    our_data = [] 
    mask_data = []
    x_lens = []    
    for x in dataset:
        mask = [1]*len(x)
        x_lens.append(len(x))       # Record the length of seq before padding
        if len(x) <= MaxSeqLen:
            while(len(x))<MaxSeqLen:
                x.append(pad_word)
                mask.append(0)
        else:
            x = x[:MaxSeqLen]
            mask = mask[:MaxSeqLen]
        
        our_data.append(x)
        mask_data.append(mask)

    # --- Shuffle  --- #
    our_data = np.array(our_data)
    index = np.arange(len(our_data))
    np.random.shuffle(index)
    our_data = our_data[index]

    # --- Split data --- #
    data_length = len(our_data)//2
    train_data = our_data[:data_length]
    test_data = our_data[data_length+1:]

    # List->Array
    train_x = np.array(train_data)      # (B,SeqLen,6)
    train_y = np.array(test_data)  
    train_mask = np.array(mask_data)    # loss mask

    # --- Save data --- #
    with open('./dataset/our_dataset.pickle','wb') as file:
        custom_data = {
            'train_x': train_x,
            'train_y': train_y,
            'mask':  train_mask
        }
        pickle.dump(custom_data, file)

if __name__ =='__main__':
    process_data()
