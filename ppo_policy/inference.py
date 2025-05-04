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
from config import device, datapath, TOKEN_COUNT, Output_File_Path, Pretrain_CKPT
from model import Actor_Transformer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 

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

# -- Sampling -- #
def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs

def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word

# -- nucleus sampling -- #
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
    cur_word = torch.tensor([cur_word]).to(device)
    return cur_word


def testing():

    midi_dir = './gen_midi/'
    os.makedirs(midi_dir, exist_ok=True)

    with open(datapath['path_dictionary'] , 'rb') as f:
        dictionary = pickle.load(f)
    event2word, word2event = dictionary

    num_token = []
    for key in event2word.keys():
        num_token.append(len(dictionary[0][key]))
    print('Num of class of token:', num_token)

    model = Actor_Transformer(num_token, is_training=False).to(device)
    ckpt_model = torch.load(Pretrain_CKPT, map_location=device)
    model.load_state_dict(ckpt_model) # ckpt_model['model_state']
    model.eval()

    init_state = np.array([[0, 0, 0, 0, 0, 0]])
    init_state = torch.from_numpy(init_state).long().to(device)
    init_state = init_state.unsqueeze(0)

    count_note = 0

    gen_song = None
    print('------ generate ------')
    with torch.no_grad():
        while True:
            h, memory = model.forward_hidden(init_state, memory=None, is_training=False)
            y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = model.forward_output(h)

            # sampling gen_cond
            # tempo_action    = sampling(y_tempo, t=1.2, p=0.8)      # sampling(y_tempo, p=0.8)
            # barbeat_action  = sampling(y_barbeat,t=1.2, p=0.8)     # sampling(y_barbeat,p=0.8)
            # chord_action    = sampling(y_chord, p=0.9)
            # pitch_action    = sampling(y_pitch, p=0.9)
            # velocity_action = sampling(y_velocity, p=0.9) 
            # duration_action = sampling(y_duration, t=1.5, p=0.9)   # sampling(y_duration, t=2, p=0.9)
            
            m = nn.Softmax(dim=-1)
            tempo_prob, chord_prob, barbeat_prob     = m(y_tempo), m(y_chord), m(y_barbeat)
            pitch_prob, duration_prob, velocity_prob = m(y_pitch), m(y_duration), m(y_velocity)
            
            tempo_dist      = torch.distributions.Categorical(tempo_prob)
            chord_dist      = torch.distributions.Categorical(chord_prob)
            barbeat_dist    = torch.distributions.Categorical(barbeat_prob)
            pitch_dist      = torch.distributions.Categorical(pitch_prob)
            duration_dist   = torch.distributions.Categorical(duration_prob)
            velocity_dist   = torch.distributions.Categorical(velocity_prob)
            
            tempo_action    = tempo_dist.sample()
            chord_action    = chord_dist.sample()
            barbeat_action  = barbeat_dist.sample()
            pitch_action    = pitch_dist.sample()
            duration_action = duration_dist.sample()
            velocity_action = velocity_dist.sample()
            
            # Take the index of max value 
            # tempo, chord, barbeat  = torch.argmax(y_tempo, dim=-1), torch.argmax(y_chord, dim=-1), torch.argmax(y_barbeat, dim=-1)
            # pitch, duration, velocity = torch.argmax(y_pitch, dim=-1), torch.argmax(y_duration, dim=-1), torch.argmax(y_velocity, dim=-1)
            # action = torch.cat((tempo[:], chord[:], barbeat[:],pitch[:], duration[:], velocity[:]), dim=0)
            # action = action.unsqueeze(0)
            
            action = torch.cat((tempo_action, chord_action,barbeat_action, pitch_action, duration_action, velocity_action), dim=0)
            next_state = action.unsqueeze(0).unsqueeze(0)
            init_state = next_state 
            # next_step = torch.cat((init_state, action), dim=1)
            # init_state = next_step
            
            if count_note==0:
                next_seq = action.unsqueeze(0)
                gen_song = next_seq.cpu().detach().numpy()
            else:
                next_seq = action.unsqueeze(0)
                gen_song = np.concatenate((gen_song, next_seq.cpu().detach().numpy()), axis=0)

            count_note+=1
            print(f'Count note: {count_note}')
            if count_note>=TOKEN_COUNT:
                break

    to_midi(gen_song, word2event, path_outfile=Output_File_Path)
    print('====== Finish ====== ')

if __name__ =='__main__':
    testing()