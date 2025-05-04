import os
import numpy as np
import time
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import prepare_data
from config import device, datapath, TOKEN_COUNT, Output_File_Path
from model import LinearTransformer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 

path_outfile = './gen_midi/111.mid'


# --- write tool --- #
def to_midi(data, word2event, path_outfile):
    tes = []    # tuple events
    for e in data:
        e = [word2event[etype][e[i]] for i, etype in enumerate(word2event)]
        te = prepare_data.GroupEvent(Tempo=int(e[0].split(' ')[1]),
                                     Bar=e[1].split(' ')[1],
                                     Position=e[2].split(' ')[1],
                                     Pitch=int(e[3].split(' ')[1]),
                                     Duration=int(e[4].split(' ')[1]),
                                     Velocity=int(e[5].split(' ')[1])
                                     )
        tes.append(te)
    prepare_data.tuple_events_to_midi(tes, path_outfile)


def fun():
    midi_dir = './gen_midi/'
    os.makedirs(midi_dir, exist_ok=True)

    with open(datapath['path_dictionary'] , 'rb') as f:
        dictionary = pickle.load(f)
    event2word, word2event = dictionary

    with open(datapath['path_train_data'], 'rb') as file:
        dataset = pickle.load(file)

    num_token = []
    for key in event2word.keys():
        num_token.append(len(dictionary[0][key]))
    print('Num of class of token:', num_token)

    ane = dataset['train_x']  #  my_dataset['train_y'] 
    to_midi(ane[10], word2event, path_outfile)


if __name__ =='__main__':
    fun()