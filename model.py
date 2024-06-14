import numpy as np 
import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.builders import RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask
from config import AgentConfig


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
# Model
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
# Model - LinearTransformer
################################################################################
class LinearTransformer(nn.Module):
    def __init__(self, n_token, is_training=True):
        super(LinearTransformer, self).__init__()

        # --- params --- #
        self.d_model = AgentConfig["D_MODEL"] 
        self.n_layer = AgentConfig["N_LAYER"]  
        self.n_head  = AgentConfig["N_HEAD"]  
        self.d_head  = AgentConfig["D_MODEL"] // AgentConfig["N_HEAD"]
        self.dropout = 0.1
        self.d_inner = 2048
        self.n_token = n_token   
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.emb_sizes = [128, 256, 64, 512, 128, 128]              # self.emb_sizes = [128, 256, 64, 32, 512, 128, 128]

        # embeddings
        print('Token_class >>>>>:', self.n_token)
        self.word_emb_tempo     = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.word_emb_chord     = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.word_emb_barbeat   = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.word_emb_pitch     = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.word_emb_duration  = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.word_emb_velocity  = Embeddings(self.n_token[5], self.emb_sizes[5])
        self.pos_emb            = PositionalEncoding(self.d_model, self.dropout)

        # linear 
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), self.d_model)

        # encoder
        if is_training:
            # encoder (training)
            self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model//self.n_head,
                value_dimensions=self.d_model//self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="causal-linear",
            ).get()
        else:
            # encoder (inference)
            print(' [o] using RNN backend.')
            self.transformer_encoder = RecurrentEncoderBuilder.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model//self.n_head,
                value_dimensions=self.d_model//self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="causal-linear",
            ).get()

        # blend with type
        self.project_concat_type = nn.Linear(self.d_model, self.d_model)        # nn.Linear(self.d_model+32, self.d_model)

        # individual output
        self.proj_tempo    = nn.Linear(self.d_model, self.n_token[0])        
        self.proj_chord    = nn.Linear(self.d_model, self.n_token[1])
        self.proj_barbeat  = nn.Linear(self.d_model, self.n_token[2])
        self.proj_pitch    = nn.Linear(self.d_model, self.n_token[3])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[4])
        self.proj_velocity = nn.Linear(self.d_model, self.n_token[5])

    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss


    def train_step(self, x, target, loss_mask):
        h  = self.forward_hidden(x)
        y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = self.forward_output(h, target)
        
        # reshape (b, s, f) -> (b, f, s)
        y_tempo     = y_tempo[:, ...].permute(0, 2, 1)
        y_chord     = y_chord[:, ...].permute(0, 2, 1)
        y_barbeat   = y_barbeat[:, ...].permute(0, 2, 1)
        y_pitch     = y_pitch[:, ...].permute(0, 2, 1)
        y_duration  = y_duration[:, ...].permute(0, 2, 1)
        y_velocity  = y_velocity[:, ...].permute(0, 2, 1)
        
        # loss
        loss_tempo = self.compute_loss(
                y_tempo, target[..., 0], loss_mask)
        loss_chord = self.compute_loss(
                y_chord, target[..., 1], loss_mask)
        loss_barbeat = self.compute_loss(
                y_barbeat, target[..., 2], loss_mask)
        loss_pitch = self.compute_loss(
                y_pitch, target[..., 3], loss_mask)
        loss_duration = self.compute_loss(
                y_duration, target[..., 4], loss_mask)
        loss_velocity = self.compute_loss(
                y_velocity, target[..., 5], loss_mask)

        
        return loss_tempo, loss_chord, loss_barbeat, loss_pitch, loss_duration, loss_velocity


    def forward_hidden(self, x, memory=None, is_training=True):
        '''
        linear transformer: b x s x f
        x.shape=(bs, nf)
        '''
        # embeddings
        emb_tempo    = self.word_emb_tempo(x[..., 0])
        emb_chord    = self.word_emb_chord(x[..., 1])
        emb_barbeat  = self.word_emb_barbeat(x[..., 2])         # emb_type = self.word_emb_type(x[..., 3])
        emb_pitch    = self.word_emb_pitch(x[..., 3])
        emb_duration = self.word_emb_duration(x[..., 4])
        emb_velocity = self.word_emb_velocity(x[..., 5])

        embs = torch.cat(
            [
                emb_tempo,
                emb_chord,
                emb_barbeat,
                emb_pitch,
                emb_duration,
                emb_velocity,
            ], dim=-1)

        emb_linear = self.in_linear(embs)
        pos_emb = self.pos_emb(emb_linear)

        # assert False
        
        # transformer
        if is_training:
            # mask
            attn_mask = TriangularCausalMask(pos_emb.size(1), device=x.device)
            h = self.transformer_encoder(pos_emb, attn_mask)                # y: b x s x d_model
            return h 
        
        else:
            pos_emb = pos_emb.squeeze(0)
            h, memory = self.transformer_encoder(pos_emb, memory=memory)    # y: s x d_model
            return h, memory 
    
    
    def forward_output(self, h, y):
        y_tempo    = self.proj_tempo(h)
        y_chord    = self.proj_chord(h)
        y_barbeat  = self.proj_barbeat(h)
        y_pitch    = self.proj_pitch(h)
        y_duration = self.proj_duration(h)
        y_velocity = self.proj_velocity(h)

        return  y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity
    

    def forward(self, x, target):
        hid = self.forward_hidden(x) 
        y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = self.forward_output(hid, target)
        return y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity 


    # For Inference
    def forward_output_sampling(self, h):
        '''
        for inference
        '''
        # sample type
        # y_type_logit = y_type[0, :]
        # cur_word_type = sampling(y_type_logit, p=0.90)
        # type_word_t = torch.from_numpy(np.array([cur_word_type])).long().cuda().unsqueeze(0)

        # tf_skip_type = self.word_emb_type(type_word_t).squeeze(0)
        # # concat
        # y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        # y_  = self.project_concat_type(y_concat_type)

        # project other
        y_tempo    = self.proj_tempo(h)
        y_chord    = self.proj_chord(h)
        y_barbeat  = self.proj_barbeat(h)
        y_pitch    = self.proj_pitch(h)
        y_duration = self.proj_duration(h)
        y_velocity = self.proj_velocity(h)
        
        # sampling gen_cond
        cur_word_tempo    = sampling(y_tempo, t=1.2, p=0.9)
        cur_word_barbeat  = sampling(y_barbeat, t=1.2)
        cur_word_chord    = sampling(y_chord, p=0.99)
        cur_word_pitch    = sampling(y_pitch, p=0.9)
        cur_word_duration = sampling(y_duration, t=2, p=0.9)
        cur_word_velocity = sampling(y_velocity, t=5)        

        # collect
        next_arr = np.array([
            cur_word_tempo,
            cur_word_chord,
            cur_word_barbeat,
            cur_word_pitch,
            cur_word_duration,
            cur_word_velocity,
            ])        
        return next_arr
    

    