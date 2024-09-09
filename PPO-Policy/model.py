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
from transformers import LongformerConfig, LongformerForMultipleChoice, LongformerModel
from config import ActorConfig, DiscriConfig


################################################################################
# Network Parameters
################################################################################
def network_paras(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

################################################################################
# Sampling Methods
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
# Trainable Params Compute trainable params
################################################################################
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
#  Actor - LinearTransformer 
################################################################################
class Actor_Transformer(nn.Module):
    def __init__(self, n_token, is_training=True):
        super(Actor_Transformer, self).__init__()

        # --- params --- #
        self.d_model = ActorConfig["D_MODEL"] 
        self.n_layer = ActorConfig["N_LAYER"]  
        self.n_head  = ActorConfig["N_HEAD"]  
        self.d_head  = ActorConfig["D_MODEL"] // ActorConfig["N_HEAD"]
        self.dropout = 0.1
        self.d_inner = 2048
        self.n_token = n_token   
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.emb_sizes = [128, 256, 64, 512, 128, 128]             

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
        
        # self.project_concat_type = nn.Linear(self.d_model+32, self.d_model)   # blend with type    
        self.value_funtion = nn.Sequential(nn.Linear(self.d_model, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 1),
                                            # nn.Sigmoid()
                                            )   
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
        y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = self.forward_output(h)
        
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
        emb_barbeat  = self.word_emb_barbeat(x[..., 2])        
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
    
    
    def forward_output(self, h):
        y_tempo    = self.proj_tempo(h)
        y_chord    = self.proj_chord(h)
        y_barbeat  = self.proj_barbeat(h)
        y_pitch    = self.proj_pitch(h)
        y_duration = self.proj_duration(h)
        y_velocity = self.proj_velocity(h)
        
        return  y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity


    # For CP-Word Inference
    def forward_output_sampling(self, h):
        # project other
        y_tempo    = self.proj_tempo(h)
        y_chord    = self.proj_chord(h)
        y_barbeat  = self.proj_barbeat(h)
        y_pitch    = self.proj_pitch(h)
        y_duration = self.proj_duration(h)
        y_velocity = self.proj_velocity(h)
        
        # sampling gen_cond
        cur_word_tempo   =  sampling(y_tempo, t=1.2, p=0.9)
        cur_word_barbeat =  sampling(y_barbeat, t=1.2)
        cur_word_chord   =  sampling(y_chord, p=0.99)
        cur_word_pitch   =  sampling(y_pitch, p=0.9)
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

################################################################################
#  Critic - LinearTransformer
################################################################################
class Critic_Transformer(nn.Module):
    def __init__(self, n_token):
        super(Critic_Transformer, self).__init__()
        # --- params --- #
        self.d_model = ActorConfig["D_MODEL"] 
        self.n_layer = ActorConfig["N_LAYER"]  
        self.n_head  = ActorConfig["N_HEAD"]  
        self.d_head  = ActorConfig["D_MODEL"] // ActorConfig["N_HEAD"]
        self.dropout = 0.1
        self.d_inner = 2048
        self.n_token = n_token   
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.emb_sizes = [128, 256, 64, 512, 128, 128]             

        # embeddings
        # print('Token_class >>>>>:', self.n_token)
        self.word_emb_tempo     = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.word_emb_chord     = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.word_emb_barbeat   = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.word_emb_pitch     = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.word_emb_duration  = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.word_emb_velocity  = Embeddings(self.n_token[5], self.emb_sizes[5])
        self.pos_emb            = PositionalEncoding(self.d_model, self.dropout)

        # linear 
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), self.d_model)

        # encoder (training)
        self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=self.n_layer,
            n_heads=self.n_head,
            query_dimensions=self.d_model//self.n_head,
            value_dimensions=self.d_model//self.n_head,
            feed_forward_dimensions=2048,
            activation='gelu',
            dropout=0.1,
            attention_type="causal-linear",).get()
        
        # self.value_funtion = nn.Sequential( nn.Linear(self.d_model, 128),
        #                                     nn.ReLU(),
        #                                     nn.Linear(128, 1),
        #                                     # nn.Sigmoid()
        #                                     )   
        
        # Features Linear
        self.proj_tempo    = nn.Linear(self.d_model, self.n_token[0])        
        self.proj_chord    = nn.Linear(self.d_model, self.n_token[1])
        self.proj_barbeat  = nn.Linear(self.d_model, self.n_token[2])
        self.proj_pitch    = nn.Linear(self.d_model, self.n_token[3])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[4])
        self.proj_velocity = nn.Linear(self.d_model, self.n_token[5])

        # Value calculation
        self.tempo_value    = nn.Linear(self.n_token[0], 1)        
        self.chord_value    = nn.Linear(self.n_token[1], 1)
        self.barbeat_value  = nn.Linear(self.n_token[2], 1)
        self.pitch_value    = nn.Linear(self.n_token[3], 1)
        self.duration_value = nn.Linear(self.n_token[4], 1)
        self.velocity_value = nn.Linear(self.n_token[5], 1)

    def value_produce(self, x):
        '''
        linear transformer: b x s x f
        x.shape=(bs, nf)
        '''
        # embeddings
        emb_tempo    = self.word_emb_tempo(x[..., 0])
        emb_chord    = self.word_emb_chord(x[..., 1])
        emb_barbeat  = self.word_emb_barbeat(x[..., 2])        
        emb_pitch    = self.word_emb_pitch(x[..., 3])
        emb_duration = self.word_emb_duration(x[..., 4])
        emb_velocity = self.word_emb_velocity(x[..., 5])
        
        embs = torch.cat([
                emb_tempo,
                emb_chord,
                emb_barbeat,
                emb_pitch,
                emb_duration,
                emb_velocity,
            ], dim=-1)

        emb_linear = self.in_linear(embs)
        pos_emb = self.pos_emb(emb_linear)
        
        # transformer
        attn_mask = TriangularCausalMask(pos_emb.size(1), device=x.device)
        hidden_seq = self.transformer_encoder(pos_emb, attn_mask)                # y: b x s x d_model

        # Each features network
        y_tempo    = self.proj_tempo(hidden_seq)
        y_chord    = self.proj_chord(hidden_seq)
        y_barbeat  = self.proj_barbeat(hidden_seq)
        y_pitch    = self.proj_pitch(hidden_seq)
        y_duration = self.proj_duration(hidden_seq)
        y_velocity = self.proj_velocity(hidden_seq)

        tempo_val    = self.tempo_value(y_tempo).mean(dim=1)
        chord_val    = self.chord_value(y_chord).mean(dim=1)
        barbeat_val  = self.barbeat_value(y_barbeat).mean(dim=1)
        pitch_val    = self.pitch_value(y_pitch).mean(dim=1)
        duration_val = self.duration_value(y_duration).mean(dim=1)
        velocity_val = self.velocity_value(y_velocity).mean(dim=1)
        
        value_list = [tempo_val,chord_val,barbeat_val,pitch_val,duration_val,velocity_val]
        sum_value=0
        for idx, context in enumerate(value_list):
            sum_value += context
        ans_val=sum_value/len(value_list)
        return ans_val


################################################################################
# Reward Model-LongFormer 
################################################################################
class LongFormer(nn.Module):
    def __init__(self, n_token):
        super(LongFormer, self).__init__()

        self.n_token = n_token
        self.MAX_SEQ = DiscriConfig['MAX_SEQ']
        self.D_MODEL = DiscriConfig['D_MODEL']
        self.N_layer = DiscriConfig['N_LAYER']
        self.N_head  = DiscriConfig['N_HEAD']
        self.CE_loss = nn.CrossEntropyLoss()

        self.emb_sizes =  [128, 256, 64, 512, 256, 256]  

        # print('Reward Model token >>>>> ', self.n_token)
        self.word_emb_tempo     = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.word_emb_chord     = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.word_emb_barbeat   = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.word_emb_pitch     = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.word_emb_duration  = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.word_emb_velocity  = Embeddings(self.n_token[5], self.emb_sizes[5])
    
        self.proj = nn.Linear(sum(self.emb_sizes), self.D_MODEL)

        # individual output
        self.proj_tempo    = nn.Linear(self.D_MODEL, self.n_token[0])        
        self.proj_chord    = nn.Linear(self.D_MODEL, self.n_token[1])
        self.proj_barbeat  = nn.Linear(self.D_MODEL, self.n_token[2])
        self.proj_pitch    = nn.Linear(self.D_MODEL, self.n_token[3])
        self.proj_duration = nn.Linear(self.D_MODEL, self.n_token[4])
        self.proj_velocity = nn.Linear(self.D_MODEL, self.n_token[5])

        # individual evaluation
        self.eval_tempo    = nn.Linear(self.n_token[0], 1)        
        self.eval_chord    = nn.Linear(self.n_token[1], 1)
        self.eval_barbeat  = nn.Linear(self.n_token[2], 1)
        self.eval_pitch    = nn.Linear(self.n_token[3], 1)
        self.eval_duration = nn.Linear(self.n_token[4], 1)
        self.eval_velocity = nn.Linear(self.n_token[5], 1)
        self.sigmoid = nn.Sigmoid()
        
        self.longformer_config = LongformerConfig(max_position_embeddings=self.MAX_SEQ,
                                                  hidden_size = self.D_MODEL,
                                                  num_hidden_layers = self.N_layer,
                                                  num_attention_heads = self.N_head,
                                                  hidden_act = "gelu",
                                                  hidden_dropout_prob = 0.1,
                                                  attention_probs_dropout_prob = 0.1,
                                                  position_embedding_type = "relative_key", # "absolute", "relative_key", "relative_key_query"
                                                  intermediate_size = 1024, 
                                                  attention_window = self.D_MODEL,    # D_MODEL 512
                                                  )
        self.longformer = LongformerModel(self.longformer_config)

    def compute_CEloss(self, predict, target, loss_mask):
        loss = self.CE_loss(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss
    
    def token_forward(self, data, target, loss_mask):
        tempo    = self.word_emb_tempo(data[..., 0])          # Take tempo in every note (取每個row的第1個col)
        chord    = self.word_emb_chord(data[..., 1])          # Take bar in every note   (取每個row的第2個col)
        barbeat  = self.word_emb_barbeat(data[..., 2])        # Take poistion in every note 
        pitch    = self.word_emb_pitch(data[..., 3])
        duration = self.word_emb_duration(data[..., 4])
        velocity = self.word_emb_velocity(data[..., 5])
        
        embedding = torch.cat([tempo, chord, barbeat, pitch, duration, velocity], dim=-1)
        cat_emb = self.proj(embedding)
        
        outputs = self.longformer(inputs_embeds=cat_emb, attention_mask=loss_mask)
        sequence_output = outputs.last_hidden_state         # (batch, seq_len, hidden_size)

        # individual prediction
        y_tempo    = self.proj_tempo(sequence_output)
        y_chord    = self.proj_chord(sequence_output)
        y_barbeat  = self.proj_barbeat(sequence_output)
        y_pitch    = self.proj_pitch(sequence_output)
        y_duration = self.proj_duration(sequence_output)
        y_velocity = self.proj_velocity(sequence_output)

        tempo_hid    = self.eval_tempo(y_tempo).mean(dim=1)
        chord_hid    = self.eval_chord(y_chord).mean(dim=1)
        barbeat_hid  = self.eval_barbeat(y_barbeat).mean(dim=1)
        pitch_hid    = self.eval_pitch(y_pitch).mean(dim=1)
        duration_hid = self.eval_duration(y_duration).mean(dim=1)
        velocity_hid = self.eval_velocity(y_velocity).mean(dim=1)

        sum_score = 0
        attr_list = [tempo_hid, chord_hid, barbeat_hid, pitch_hid, duration_hid, velocity_hid]
        for idx, attr in enumerate(attr_list):
            attr_list[idx] = self.sigmoid(attr)
            sum_score += attr_list[idx]
        sum_score = sum_score / len(attr_list)

        return sum_score