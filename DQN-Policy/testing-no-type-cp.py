import sys
import os
import math
import time
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
from model import LinearTransformer

gid = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)

################################################################################
# config
################################################################################
###--- data ---###
path_data_root = '/data/dataset_Pop1K7/representations/uncond/cp/ailab17k_from-scratch_cp'
path_train_data = os.path.join(path_data_root, 'train_data_linear.npz')
path_dictionary = os.path.join(path_data_root, 'dictionary.pkl')

###--- Load checkpoint ---###
path_saved_ckpt = './ckpt/dqn_best.pt' 

###--- Generated config  ---###
generate_songs = 5
bar_production = 50

###--- Directory  ---###
path_gendir = 'gen_midis'
os.makedirs(path_gendir, exist_ok=True)

###--- Parameter ---###
D_MODEL = 512
N_LAYER = 10
N_HEAD = 8    
batch_size = 4
init_lr = 0.001

BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4


################################################################################
# Inference
################################################################################

def write_midi(words, path_outfile, word2event):
    class_keys = word2event.keys()
    # words = np.load(path_infile)
    midi_obj = miditoolkit.midi.parser.MidiFile()

    bar_cnt = 0
    cur_pos = 0

    all_notes = []
    for i in range(len(words)):
        vals = []
        for kidx, key in enumerate(class_keys):
            vals.append(word2event[key][words[i][kidx]])
        # print(vals)

        note_num = (type(vals[3])==str and type(vals[4])==str and type(vals[5])==str)

        # Metrical (0, Bar, Beat)
        if note_num != True:
            
            if vals[2] == 'Bar':
                bar_cnt += 1
            
            if vals[2] == 0:
                pass
            
            elif 'Beat' in vals[2]:
                beat_pos = int(vals[2].split('_')[1])
                cur_pos = bar_cnt * BAR_RESOL + beat_pos * TICK_RESOL

                # chord
                if vals[1] != 'CONTI' and vals[1] != 0:
                    midi_obj.markers.append(
                        Marker(text=str(vals[1]), time=cur_pos))

                if vals[0] != 'CONTI' and vals[0] != 0:
                    tempo = int(vals[0].split('_')[-1])
                    midi_obj.tempo_changes.append(
                        TempoChange(tempo=tempo, time=cur_pos))
        # Note 
        else: 
            if (type(vals[3])==str and type(vals[4])==str and type(vals[5])==str):
                try:
                    pitch    = vals[3].split('_')[-1]
                    duration = vals[4].split('_')[-1]
                    velocity = vals[5].split('_')[-1]
                    
                    if int(duration) == 0:
                        duration = 60
                    end = cur_pos + int(duration)
                    
                    all_notes.append(
                        Note(
                            pitch=int(pitch), 
                            start=cur_pos, 
                            end=end, 
                            velocity=int(velocity))
                        )
                except:
                    continue
    
    # saving midi file
    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = all_notes
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_outfile)



def inference_from_scratch(model, word2event, bar_cond):
        
        classes = word2event.keys()
        def print_word_cp(cp):
            result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]
            for r in result:
                print('{:15s}'.format(str(r)), end=' | ')
            print('')

        init = np.array([
            [0, 0, 1, 0, 0, 0], # bar
        ])

        cnt_bar = 1   ##
        cnt_token = len(init)
        with torch.no_grad():
            final_res = []
            memory = None
            h = None
            
            init_t = torch.from_numpy(init).long().cuda()
            print('------ initiate ------')
            
            for step in range(init.shape[0]):
                print_word_cp(init[step, :])
                input_ = init_t[step, :].unsqueeze(0).unsqueeze(0)
                final_res.append(init[step, :][None, ...])

                h, memory = model.forward_hidden(input_, memory, is_training=False)

            print('------ generate ------')
            while(True):
                # sample others
                next_arr = model.forward_output_sampling(h)     
                final_res.append(next_arr[None, ...])
                print('bar:', cnt_bar, end= '  ==')
                print_word_cp(next_arr)

                # forward
                input_  = torch.from_numpy(next_arr).long().cuda()
                input_  = input_.unsqueeze(0).unsqueeze(0)
                h, memory = model.forward_hidden(input_, memory, is_training=False)

                if word2event['bar-beat'][next_arr[2]] == 'Bar':
                    cnt_bar += 1
                
                # end of sequence
                if cnt_bar == bar_cond:
                    break

        print('\n--------[Done]--------')
        final_res = np.concatenate(final_res)
        print(final_res.shape)
        return final_res
    

def generate(model, word2event): 
    
    start_time = time.time()
    song_time_list = []
    words_len_list = []

    cnt_tokens_all = 0 
    sidx = 0
    
    while sidx < generate_songs:
        try:
            start_time = time.time()
            print('current idx:', sidx)
            path_outfile = os.path.join(path_gendir, 'get_{}.mid'.format(str(sidx)))

            res = inference_from_scratch(model, word2event, bar_production)
            write_midi(res, path_outfile, word2event)

            song_time = time.time() - start_time
            word_len = len(res)
            print('song time:', song_time)
            print('word_len:', word_len)
            words_len_list.append(word_len)
            song_time_list.append(song_time)

            sidx += 1
        except KeyboardInterrupt:
            raise ValueError(' [x] terminated.')
        except:
            continue
    
    print('ave token time:', sum(words_len_list) / sum(song_time_list))
    print('ave song time:', np.mean(song_time_list))
    
    runtime_result = {
        'song_time':song_time_list,
        'words_len_list': words_len_list,
        'ave token time:': sum(words_len_list) / sum(song_time_list),
        'ave song time': float(np.mean(song_time_list)),
    }

    with open('runtime_stats.json', 'w') as f:
        json.dump(runtime_result, f)


if __name__ == '__main__':
    
    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary
    # Modify encoding 
    del event2word['type']
    del word2event['type']     # word2event['type']: {0: 'EOS', 1: 'Metrical', 2: 'Note'}

    # output dir
    os.makedirs(path_gendir, exist_ok=True)

    # Token Class 
    n_class = []
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    # Model
    model = LinearTransformer(n_class, is_training=False)
    model.cuda()
    model.eval()
    
    print(f'Load model from: {path_saved_ckpt}')
    checkpoint = torch.load(path_saved_ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])    

    generate(model, word2event)

