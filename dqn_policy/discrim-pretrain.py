'''
Model: Longformer
Embedding: Compound Word
'''
import sys
import os
import math
import time
import glob
import datetime
import random
import pickle
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.builders import RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask
from transformers import LongformerConfig, LongformerForMultipleChoice, LongformerModel
from transformers import TrajectoryTransformerConfig, TrajectoryTransformerModel

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
import saving

################################################################################
# config
################################################################################

MODE = 'train' 
# MODE = 'inference' 

###--- data ---###
path_data_root = '/data/dataset_Pop1K7/representations/uncond/cp/ailab17k_from-scratch_cp'
path_train_data = os.path.join(path_data_root, 'train_data_linear.npz')
path_dictionary = os.path.join(path_data_root, 'dictionary.pkl')

###--- training config ---###
D_MODEL = 512
MAX_SEQ = 4096
N_LAYER = 12
N_HEAD = 8   
SEQ_LEN = 1024 
path_exp = 'exp'
batch_size = 2
init_lr = 0.01
path_gendir = 'gen_midis'
num_songs = 5


################################################################################
# File IO
################################################################################
gid = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4


def write_midi(words, path_outfile, word2event):
    
    class_keys = word2event.keys()
    # words = np.load(path_infile)
    midi_obj = miditoolkit.midi.parser.MidiFile()

    bar_cnt = 0
    cur_pos = 0

    all_notes = []

    cnt_error = 0
    for i in range(len(words)):
        vals = []
        for kidx, key in enumerate(class_keys):
            vals.append(word2event[key][words[i][kidx]])
        # print(vals)

        if vals[3] == 'Metrical':
            if vals[2] == 'Bar':
                bar_cnt += 1
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
            else:
                pass
        elif vals[3] == 'Note':

            try:
                pitch = vals[4].split('_')[-1]
                duration = vals[5].split('_')[-1]
                velocity = vals[6].split('_')[-1]
                
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
        else:
            pass
    
    # save midi
    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = all_notes
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_outfile)


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
# Eembedding 
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


################################################################################
# Model
################################################################################
class LongFormer(nn.Module):
    def __init__(self, n_token):
        super(LongFormer, self).__init__()

        self.n_token = n_token
        self.MAX_SEQ = MAX_SEQ
        self.D_MODEL = D_MODEL
        self.N_layer = N_LAYER
        self.N_head  = N_HEAD
        self.CE_loss = nn.CrossEntropyLoss()

        self.emb_sizes = [128, 256, 64, 512, 256, 128]    # CP Embedding

        # print('Disc token >>>>>', self.n_token)
        self.word_emb_tempo     = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.word_emb_chord     = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.word_emb_barbeat   = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.word_emb_pitch     = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.word_emb_duration  = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.word_emb_velocity  = Embeddings(self.n_token[5], self.emb_sizes[5])
    
        self.in_linear = nn.Linear(sum(self.emb_sizes), self.D_MODEL) 

        self.longformer_config = LongformerConfig(max_position_embeddings=self.MAX_SEQ,
                                                  hidden_size = self.D_MODEL,
                                                  num_hidden_layers = self.N_layer,
                                                  num_attention_heads = self.N_head,
                                                  hidden_act = "gelu",
                                                  hidden_dropout_prob = 0.1,
                                                  attention_probs_dropout_prob = 0.1,
                                                  position_embedding_type = "absolute", # "absolute", "relative_key", "relative_key_query"
                                                  intermediate_size = 1024, 
                                                  attention_window = D_MODEL,    # D_MODEL: 512
                                                  )
        self.longformer = LongformerModel(self.longformer_config)

        # blend with type
        self.project_concat_type = nn.Linear(np.sum(self.emb_sizes), self.D_MODEL)

        # Forward_out: individual prediction
        self.proj_tempo    = nn.Linear(self.D_MODEL, self.n_token[0])        
        self.proj_chord    = nn.Linear(self.D_MODEL, self.n_token[1])
        self.proj_barbeat  = nn.Linear(self.D_MODEL, self.n_token[2])
        self.proj_type     = nn.Linear(self.D_MODEL, self.n_token[3])
        self.proj_pitch    = nn.Linear(self.D_MODEL, self.n_token[4])
        self.proj_duration = nn.Linear(self.D_MODEL, self.n_token[5])
        self.proj_velocity = nn.Linear(self.D_MODEL, self.n_token[6])
        

    def compute_CEloss(self, predict, target, loss_mask):
        loss = self.CE_loss(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss


    def forward_hidden(self, data, masks):
        """
        args:
            data: (batch, windows_size, 7)
            masks: (batch, windows_size)        
        """
        emb_tempo    = self.word_emb_tempo(data[..., 0])           # Take tempo in every note (取每個row的第1個col)
        emb_chord    = self.word_emb_chord(data[..., 1])           # Take bar in every note   (取每個row的第2個col)
        emb_barbeat  = self.word_emb_barbeat(data[..., 2])         # Take poistion in every note 
        emb_pitch    = self.word_emb_pitch(data[..., 3])
        emb_duration = self.word_emb_duration(data[..., 4])
        emb_velocity = self.word_emb_velocity(data[..., 5])
        
        embedding = torch.cat([emb_tempo, emb_chord, emb_barbeat, emb_pitch, emb_duration, emb_velocity], dim=-1)
        inputs_x = self.in_linear(embedding)
        
        outputs = self.longformer(inputs_embeds=inputs_x, attention_mask=masks)
        sequence_output = outputs.last_hidden_state     # (batch, seq_len, hidden_size)
        
        # # type projection
        # y_type = self.proj_type(sequence_output)
        return sequence_output


    def forward_output(self, h, y):

        # tf_skip_type = self.word_emb_type(y[..., 3])
        # # project other
        # y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        # y_  = self.project_concat_type(y_concat_type)
        y_tempo    = self.proj_tempo(h)
        y_chord    = self.proj_chord(h)
        y_barbeat  = self.proj_barbeat(h)
        y_pitch    = self.proj_pitch(h)
        y_duration = self.proj_duration(h)
        y_velocity = self.proj_velocity(h)

        return  y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity


    def train_step(self, x, target, loss_mask):
        h, y_type  = self.forward_hidden(x, loss_mask)
        y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = self.forward_output(h, target)
        
        # reshape (b, s, f) -> (b, f, s)
        y_tempo     = y_tempo[:, ...].permute(0, 2, 1)
        y_chord     = y_chord[:, ...].permute(0, 2, 1)
        y_barbeat   = y_barbeat[:, ...].permute(0, 2, 1)
        y_type      = y_type[:, ...].permute(0, 2, 1)
        y_pitch     = y_pitch[:, ...].permute(0, 2, 1)
        y_duration  = y_duration[:, ...].permute(0, 2, 1)
        y_velocity  = y_velocity[:, ...].permute(0, 2, 1)
        
        # loss
        loss_tempo      = self.compute_CEloss(y_tempo, target[..., 0], loss_mask)
        loss_chord      = self.compute_CEloss(y_chord, target[..., 1], loss_mask)
        loss_barbeat    = self.compute_CEloss(y_barbeat, target[..., 2], loss_mask)
        loss_type       = self.compute_CEloss(y_type,  target[..., 3], loss_mask)
        loss_pitch      = self.compute_CEloss(y_pitch, target[..., 4], loss_mask)
        loss_duration   = self.compute_CEloss( y_duration, target[..., 5], loss_mask)
        loss_velocity   = self.compute_CEloss(y_velocity, target[..., 6], loss_mask)

        return loss_tempo, loss_chord, loss_barbeat, loss_type, loss_pitch, loss_duration, loss_velocity



##########################################################################################################################
# Script
##########################################################################################################################

def train():
    # hyper params
    n_epoch = 4000
    max_grad_norm = 3

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary
    train_data = np.load(path_train_data)

    # create saver
    saver_agent = saving.Saver(path_exp)

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))
    # log
    print('num of classes:', n_class)
    
    # init
    net = LongFormer(n_class).to(device)
    net.train()
    n_parameters = network_paras(net)
    print('n_parameters: {:,}'.format(n_parameters))
    saver_agent.add_summary_msg(' > params amount: {:,d}'.format(n_parameters))
    
    # optimizers
    optimizer = optim.Adam(net.parameters(), lr=init_lr)

    # unpack
    train_x = train_data['x']
    train_y = train_data['y']
    train_mask = train_data['mask']
    num_batch = len(train_x) // batch_size
    
    ###--- Delete 'type'(index=3) ---### 
    train_x = np.concatenate((train_x[:,:,:3], train_x[:,:,4:]), axis=2)
    train_y = np.concatenate((train_y[:,:,:3], train_y[:,:,4:]), axis=2)
    
    ###--- SEQ_LEN ---### 
    data_x    = train_x[:,:SEQ_LEN,:].long().cuda()
    data_y    = train_y[:,:SEQ_LEN*2,:].long().cuda()
    data_mask = train_mask[:,:SEQ_LEN+1000].float().cuda()
    print('    num_batch:', num_batch)
    print('    train_x:', train_x.shape)
    print('    train_y:', train_y.shape)
    print('    train_mask:', train_mask.shape)

    # run
    start_time = time.time()
    for epoch in range(n_epoch):
        acc_loss = 0
        acc_losses = np.zeros(7)   # 

        for bidx in range(num_batch): # num_batch 
            saver_agent.global_step_increment()
            
            # index
            bidx_st = batch_size * bidx
            bidx_ed = batch_size * (bidx + 1)

            # unpack batch data
            batch_x = train_x[bidx_st:bidx_ed]
            batch_y = train_y[bidx_st:bidx_ed]
            batch_mask = train_mask[bidx_st:bidx_ed]
            
            # to tensor
            batch_x = torch.from_numpy(batch_x).long().to(device)
            batch_y = torch.from_numpy(batch_y).long().to(device)
            batch_mask = torch.from_numpy(batch_mask).float().to(device)

            # run
            losses = net.train_step(batch_x, batch_y, batch_mask)
            loss = (losses[0] + losses[1] + losses[2] + losses[3] + losses[4] + losses[5] ) / 6
        
            # Update
            net.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            # print
            sys.stdout.write('Batch: {}/{} | Loss: {:06f} | {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\r'.format(
                bidx, num_batch, loss, losses[0], losses[1], losses[2], losses[3], losses[4], losses[5]))
            sys.stdout.flush()

            # acc
            acc_losses += np.array([l.item() for l in losses])
            acc_loss += loss.item()

            # log
            saver_agent.add_summary('batch loss', loss.item())
        

        # epoch loss
        runtime = time.time() - start_time
        epoch_loss = acc_loss / num_batch
        acc_losses = acc_losses / num_batch
        print('------------------------------------')
        print('Epoch: {}/{} | Loss: {} | time: {}'.format(
            epoch, n_epoch, epoch_loss, str(datetime.timedelta(seconds=runtime))))
        each_loss_str = '{:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\r'.format(
              acc_losses[0], acc_losses[1], acc_losses[2], acc_losses[3], acc_losses[4], acc_losses[5])
        print('    >', each_loss_str)

        saver_agent.add_summary('epoch loss', epoch_loss)
        saver_agent.add_summary('epoch each loss', each_loss_str)

        # save model, with policy
        loss = epoch_loss
        ckpt_dir = './ckpt'
        if 0.4 < loss <= 0.8:
            fn = int(loss * 10) * 10
            filename = 'trainloss_' +str(fn) +'.pt'
            ckpt_saved_path = os.path.join(ckpt_dir, filename)
            torch.save({
                    'epoch':  n_epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_saved_path)
            # saver_agent.save_model(net, name='loss_' + str(fn))

        elif 0.05 < loss <= 0.40:
            fn = int(loss * 100)
            filename = 'trainloss_' +str(fn) +'.pt'
            ckpt_saved_path = os.path.join(ckpt_dir, filename)
            
            torch.save({
                    'epoch':  n_epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_saved_path)
           
        elif loss <= 0.05:
            print('Finished')
            return  
        
        else:
            fn = int(loss * 100)
            filename = 'trainloss_' + str(fn) + '_high.pt'
            ckpt_saved_path = os.path.join(ckpt_dir, filename)

            torch.save({
                    'epoch':  n_epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_saved_path)
           


def generate():
    # path
    path_ckpt = './ckpt/' # path to ckpt dir
    path_saved_ckpt = os.path.join(path_ckpt + '_params.pt')

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary

    # outdir
    os.makedirs(path_gendir, exist_ok=True)

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    # init model
    net = LongFormer(n_class, is_training=False)
    net.cuda()
    net.eval()
    
    # load model
    print('[*] load model from:',  path_saved_ckpt)
    net.load_state_dict(torch.load(path_saved_ckpt))

    # gen
    start_time = time.time()
    song_time_list = []
    words_len_list = []

    cnt_tokens_all = 0 
    sidx = 0
    while sidx < num_songs:
        try:
            start_time = time.time()
            print('current idx:', sidx)
            path_outfile = os.path.join(path_gendir, 'get_{}.mid'.format(str(sidx)))

            res = net.inference_from_scratch(dictionary)
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
    # -- training -- #
    if MODE == 'train':
        train()

    # -- inference -- #
    elif MODE == 'inference':
        generate()
        
    else:
        pass
