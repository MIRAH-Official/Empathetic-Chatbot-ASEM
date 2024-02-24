### TAKEN FROM https://github.com/kolloldas/torchnlp

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np
import math
from model.common_layer import EncLayer, EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm , _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask,  get_input_from_batch, get_output_from_batch, ExpertEncoder, top_filtering
from utils import config
import random
# from numpy import random
import os
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import os
import time
from copy import deepcopy
from sklearn.metrics import accuracy_score

# warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
import warnings
warnings.filterwarnings('ignore') 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# ASEM Main Contribution
class MoE_Encoder(nn.Module):
    def __init__(self, input_dim, expert_num,  embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 sent_maps, filter_size, max_length=900, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, dropout=0.0 ):
               
        super(MoE_Encoder, self).__init__()

        params = (hidden_size, 
                total_key_depth or hidden_size,  # Use total_key_depth if provided, else default to hidden_size
                total_value_depth or hidden_size,  # Same as above for total_value_depth
                filter_size, 
                num_heads, 
                _gen_bias_mask(max_length),  
                layer_dropout, 
                attention_dropout, 
                relu_dropout)

        # Initialize class attributes
        self.decoder_stn_num = expert_num
        self.input_dim = input_dim
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_length = max_length

        # Layer normalization applied to the output
        self.layer_norm = torch.nn.LayerNorm(hidden_size)

        # A list of expert encoders
        self.expert_list = nn.ModuleList([ExpertEncoder(self.input_dim, self.hidden_dim, self.num_heads, self.dropout,  max_length=900) for e in range(config.topk_stn)]) ####config.topk_stn
        
        # Sequential container of encoder layers
        self.enc = nn.Sequential(*[EncLayer(*params) for l in range(config.hop)])

    def forward(self,inputs, all_mask, encoder_outputs, attention_parameters_stn,  attention_mask): 

        # Get expert outputs
        exp_out = []

        for i, expert in enumerate(self.expert_list):
            fc_output = expert(inputs, encoder_outputs, all_mask)
            exp_out.append(fc_output)

        # Calculate the mean of each expert's output
        expert_outputs = []
        for i in range(config.topk_stn):  
            expert_output = exp_out[i].mean(dim=1)  # [batch_size, hidden_size]
            expert_outputs.append(expert_output)

        # Stack the expert outputs and perform operations to integrate attention parameters
        expert_outputs = torch.stack(expert_outputs, dim=1).cuda()  # [batch_size, num_experts, hidden_size] e.g. expert_outputs torch.Size([16, 2, 300])
        attention_parameters_stn = attention_parameters_stn.squeeze(-1).transpose(1, 2).cuda()
        attn_weights = torch.bmm(attention_parameters_stn, expert_outputs) # torch.Size([16, 1, 300])


        # Combine the attention weights with encoder outputs and normalize
        sum_output = torch.add(attn_weights, encoder_outputs) # torch.Size([16, 63, 300])
        sum_output_norm = self.layer_norm(sum_output)


       # Pass the normalized sum through the encoder layers and apply layer normalization again
        y, _, _ = self.enc((sum_output_norm, inputs, attention_mask))
        y = self.layer_norm(y)

        return sum_output, y





class Encoder(nn.Module):
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=900, input_dropout=0.0, layer_dropout=0.1, 
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """
        
        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length) if use_mask else None,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if(self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.Sequential(*[EncoderLayer(*params) for l in range(num_layers)])


        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        
        if(config.act):
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None
        
    def forward(self, inputs,last_turn,last_turn_mask, mask):
        #Add input dropout
        x = self.input_dropout(inputs)
        
        # Project to hidden size
        x = self.embedding_proj(x)
        
        if(self.universal):
            if(config.act):
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                    x2,x,_,_ = self.enc((x,last_turn,last_turn_mask,  mask))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
            
            for i in range(self.num_layers):
                x2,x,_,_ = self.enc((x,last_turn,last_turn_mask,  mask))
            
            y = self.layer_norm(x)
        return y

class Decoder(nn.Module):
    """
    A Transformer Decoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=900, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length), # mandatory
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        
        if(self.universal):
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])
        
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)


    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        #Add input dropout
        x = self.input_dropout(inputs)
        if (not config.project): x = self.embedding_proj(x)
            
        if(self.universal):
            if(config.act):
                x, attn_dist, (self.remainders,self.n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal, self.position_signal, self.num_layers, encoder_output, decoding=True)
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src,dec_mask)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
            
            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src,dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist

class MulDecoder(nn.Module):
    def __init__(self, expert_num,  embedding_size, hidden_size, vocab, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=900, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(MulDecoder, self).__init__()
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        self.mask = _get_attn_subsequent_mask(max_length)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length), # mandatory
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        if config.basic_learner: self.basic = DecoderLayer(*params)
        self.experts = nn.ModuleList([DecoderLayer(*params) for e in range(expert_num)])

        self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.embedding = nn.Embedding(self.vocab_size, hidden_size)


    def forward(self, inputs, encoder_output, mask, attention_epxert):
        
        mask_src, mask_trg = mask
        
        dec_mask = torch.gt(mask_trg.cuda() + self.mask[:, :mask_trg.size(-1), :mask_trg.cuda().size(-1)].cuda(), 0)
        #Add input dropout
        x = self.input_dropout(inputs)
        if (not config.project): x = self.embedding_proj(x)
        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
        expert_outputs = []
        if config.basic_learner:
            basic_out ,_, attn_dist, _ = self.basic((x, encoder_output, [], (mask_src,dec_mask)))

        #compute experts
        #TODO forward all experts in parrallel
        if (attention_epxert.shape[0]==1 and config.topk>0):
            for i, expert in enumerate(self.experts):
                if attention_epxert[0, i]>0.0001:         #speed up inference
                    expert_out , _,attn_dist, _ = expert((x, encoder_output, [], (mask_src,dec_mask)))
                    expert_outputs.append(attention_epxert[0, i].cpu()*expert_out.cpu())
            x = torch.stack(expert_outputs, dim=1)
            x = x.sum(dim=1)
                    
        else:
            for i, expert in enumerate(self.experts):
                expert_out , _, attn_dist, _ = expert((x, encoder_output, [], (mask_src,dec_mask)))
                expert_outputs.append(expert_out)
            x = torch.stack(expert_outputs, dim=1).cuda() #(batch_size, expert_number, len, hidden_size)
            x = attention_epxert.cuda() * x
            x = x.sum(dim=1).cpu() #(batch_size, len, hidden_size)
        if config.basic_learner:
            x+=basic_out.cpu()
        # Run decoder
        y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src,dec_mask)))
 
        # Final layer normalization
        y = self.layer_norm(y)

        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)
        temperature_dim=1
        self.temperature = nn.Parameter(torch.ones(temperature_dim))

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1, beam_search=False, attn_dist_db=None):
        temperature2 = 0.7
        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if(config.pointer_gen):
            vocab_dist = F.softmax(logit/temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist/temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist            
            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)]*x.size(1),1) ## extend for all seq
            if(beam_search):
                enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab_[0].unsqueeze(0)]*x.size(0),0) ## extend for all seq
            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_))
            return logit
        else:
            return F.log_softmax(logit,dim=-1)



class Transformer_experts(nn.Module):

    def __init__(self, vocab, decoder_number,decoder_stance_num, model_file_path=None, is_eval=False, load_optim=False):

        super(Transformer_experts, self).__init__()
        self.vocab = vocab

        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab,config.pretrain_emb)
        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth, total_value_depth=config.depth,
                                filter_size=config.filter,universal=config.universal)
        self.decoder_number = decoder_number
        self.decoder_stance_num = decoder_stance_num
        ## multiple encoders
        self.mulencoder = MoE_Encoder(input_dim=config.hidden_dim, expert_num= config.topk_stn, embedding_size= config.emb_dim, hidden_size= config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                    total_key_depth=config.depth,total_value_depth=config.depth, sent_maps=self.decoder_stance_num,
                                    filter_size=config.filter, max_length=900, input_dropout=0.0, layer_dropout=0.0, 
                                    attention_dropout=0.0, relu_dropout=0.0, dropout=0.0)



        ## multiple decoders
        self.decoder = MulDecoder(decoder_number, config.emb_dim, config.hidden_dim, self.vocab, num_layers=config.hop, num_heads=config.heads, 
                                    total_key_depth=config.depth,total_value_depth=config.depth,
                                    filter_size=config.filter)
 
        self.decoder_key = nn.Linear(config.hidden_dim ,decoder_number, bias=False)
        self.decoder_stance_key = nn.Linear(config.hidden_dim , decoder_stance_num, bias=False)

        self.generator = Generator(config.hidden_dim, self.vocab_size)


        self.proj = nn.Linear(config.hidden_dim, self.vocab_size)


        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if (config.label_smoothing):
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
            self.embedding_rob = nn.Embedding(self.vocab_size, config.hidden_dim)
        if config.softmax:
            self.attention_activation =  nn.Softmax(dim=1)
        else:
            self.attention_activation =  nn.Sigmoid() #nn.Softmax()

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config.lr)
        if(config.noam):
            self.optimizer = NoamOpt(config.hidden_dim, 1, 8000, torch.optim.AdamW(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)) #1e-9


        if model_file_path is not None:
            print("loading weights")
            

            # state = torch.load('chatbot_modelSEE4ED.pt', map_location= lambda storage, location: storage)
            # self.encoder.load_state_dict(state['encoder_state_dict'])
            # self.mulencoder.load_state_dict(state['mulencoder_state_dict'])

            # self.decoder.load_state_dict(state['decoder_state_dict'])
            # self.decoder_key.load_state_dict(state['decoder_key_state_dict']) 
            # self.decoder_stance_key.load_state_dict(state['decoder_stanceKey_state_dict']) 
            # self.generator.load_state_dict(state['generator_dict'])
            # self.embedding.load_state_dict(state['embedding_dict'])
            # if (load_optim):
            #     self.optimizer.load_state_dict(state['optimizer'])
            # state.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, f1_g,f1_b,ent_g,ent_b):
        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'mulencoder_state_dict': self.mulencoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'decoder_key_state_dict': self.decoder_key.state_dict(),
            'decoder_stanceKey_state_dict':  self.decoder_stance_key.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter,running_avg_ppl,f1_g,f1_b,ent_g,ent_b) )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, iter, ty, train=True):

        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)
        history_turn = batch['history_batch']
        current_turn = batch['current_turn_batch']
        emb_mask_current= batch['mask_input_current_tur']
        emb_mask_history = batch['mask_input_hist']
        
        emb_mask_full = torch.cat([emb_mask_history, emb_mask_current], dim=1).to(device)
        emb_mask_full_emb = self.embedding(emb_mask_full)

        mask_src_cuTu = current_turn.data.eq(config.PAD_idx).unsqueeze(1)


        if(config.noam):
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        # Create an embedding mask for the current turn
        current_turn_mask = self.embedding(batch["mask_input_current_tur"])
        # num_turns = batch["num_turns"]

        # Apply weighting factor to current turn tensor
        weighting_factor = 2

        # define weight vector for history and current turn tensors
        hist_emb = self.embedding(history_turn)
        current_emb = self.embedding(current_turn)
        bat_size, seq_len1, hid_dim = hist_emb.size()
        bat_size2, seq_len2, hid_dim2 = current_emb.size()

        current_turn_weight = torch.ones(bat_size2, seq_len2) * weighting_factor  # set weight_factor to a value greater than 1

        # multiply current turn tensor by its weight vector
        weighted_current_turn = current_emb * current_turn_weight.unsqueeze(-1).to(device)

        # concatenate history and weighted current turn tensors along sequence length dimension
        full_sequence = torch.cat([hist_emb, weighted_current_turn], dim=1).to(device)

        ful2 = torch.cat([history_turn, current_turn], dim=1).to(device)
        full_sequence_mask = (ful2.eq(config.PAD_idx).unsqueeze(1)).bool()

        #General Encoder
        encoder_outputs = self.encoder(full_sequence+emb_mask_full_emb,self.embedding(current_turn)+current_turn_mask,mask_src_cuTu, full_sequence_mask)

        q_h = torch.mean(encoder_outputs,dim=1) if config.mean_query else encoder_outputs[:,0]

        logit_prob_stn = self.decoder_stance_key(q_h) #(bsz, num_experts)

        
        if(config.topk_stn>0):
            k_max_value, k_max_index = torch.topk(logit_prob_stn, config.topk_stn)
            a = np.empty([logit_prob_stn.shape[0], self.decoder_stance_num])
            a.fill(float('-inf'))
            mask = torch.Tensor(a).cuda()
            logit_prob_stn2_ = mask.scatter_(1,k_max_index.cuda().long(),k_max_value)
            attention_parameters = self.attention_activation(logit_prob_stn2_)
        else:
            attention_parameters = self.attention_activation(logit_prob_stn)


        # if(config.oracle): attention_parameters = self.attention_activation(torch.FloatTensor(batch['stance_program'])*1000).cuda()
        if(config.oracle): attention_parameters = self.attention_activation(torch.FloatTensor(batch['stance_program'])*1000)

        attention_parameters_stn = attention_parameters.unsqueeze(-1).unsqueeze(-1) # (batch_size, expert_num, 1, 1)

        #stance Aware Encoders (Mixture of Experts)
        weighted_output, encoder_outputs2 = self.mulencoder(full_sequence+emb_mask_full_emb, full_sequence_mask,encoder_outputs,attention_parameters_stn,  full_sequence_mask)


        ## Attention over decoder
        q_h = torch.mean(weighted_output,dim=1) if config.mean_query else weighted_output[:,0]
        logit_prob = self.decoder_key(q_h) #(bsz, num_experts)

        if(config.topk>0):
            k_max_value, k_max_index = torch.topk(logit_prob, config.topk)
            a = np.empty([logit_prob.shape[0], self.decoder_number])
            a.fill(float('-inf'))
            mask = torch.Tensor(a).cuda()
            logit_prob_ = mask.scatter_(1,k_max_index.cuda().long(),k_max_value)
            attention_parameters = self.attention_activation(logit_prob_)
        else:
            attention_parameters = self.attention_activation(logit_prob)
        # print("===============================================================================")
        # print("listener attention weight:",attention_parameters.data.cpu().numpy())
        # print("===============================================================================")
        # if(config.oracle): attention_parameters = self.attention_activation(torch.FloatTensor(batch['target_program'])*1000).cuda()
        if(config.oracle): attention_parameters = self.attention_activation(torch.FloatTensor(batch['target_program'])*1000)

        attention_parameters = attention_parameters.unsqueeze(-1).unsqueeze(-1) # (batch_size, expert_num, 1, 1)
        
        # Decode 
        sos_token = torch.LongTensor([config.SOS_Rob_idx] * enc_batch.size(0)).unsqueeze(1)
        if config.USE_CUDA: sos_token = sos_token.cuda()
        dec_batch_shift = torch.cat((sos_token,dec_batch[:, :-1]),1)

        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

        #Emotion Aware Decoders (Mixture of Experts)
        pre_logit, attn_dist = self.decoder(self.embedding(dec_batch_shift),encoder_outputs2, (full_sequence_mask,mask_trg), attention_parameters)

        ## compute output dist
        logit = self.generator(pre_logit,attn_dist,enc_batch_extend_vocab if config.pointer_gen else None, extra_zeros, attn_dist_db=None)

        if(train and config.schedule>10):
            if(random.uniform(0, 1) <= (0.0001 + (1 - 0.0001) * math.exp(-1. * iter / config.schedule))):
                config.oracle=True
            else:
                config.oracle=False

      
        if config.softmax:

            loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))  + nn.CrossEntropyLoss()(logit_prob,torch.LongTensor(batch['program_label']).cuda()) + nn.CrossEntropyLoss()(logit_prob_stn,torch.LongTensor(batch['program_stance']).cuda()) #* 0.1

            loss_bce_program = nn.CrossEntropyLoss()(logit_prob,torch.LongTensor(batch['program_label']).cuda()).item() + nn.CrossEntropyLoss()(logit_prob_stn,torch.LongTensor(batch['program_stance']).cuda()).item()

        else:
            loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1)) + nn.BCEWithLogitsLoss()(logit_prob,torch.FloatTensor(batch['target_program']).cuda())
            loss_bce_program = nn.BCEWithLogitsLoss()(logit_prob,torch.FloatTensor(batch['target_program']).cuda()).item()
        pred_program = np.argmax(logit_prob.detach().cpu().numpy(), axis=1)
        act = batch["program_label"]
        program_acc = accuracy_score(batch["program_label"], pred_program)

        if(config.label_smoothing): 
            loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1)).item()
        
        if(train):
            loss.backward()
            self.optimizer.step()

        if(config.label_smoothing):
            return loss_ppl, math.exp(min(loss_ppl, 100)), loss_bce_program, program_acc, pred_program, act
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), loss_bce_program, program_acc, pred_program, act


    def decoder_topk(self, batch, max_dec_step=30):

        SPECIAL_TOKENS = ["EOS", "SOS", "USR", "SYS", "SPE", "PAD", "UNK"]
        special_tokens_ids = [self.vocab.word2index[ST] for ST in SPECIAL_TOKENS] #self.vocab.word2index(SPECIAL_TOKENS)
        

        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        history_turn = batch['history_batch']
        current_turn = batch['current_turn_batch']
        emb_mask_current= batch['mask_input_current_tur']
        emb_mask_history = batch['mask_input_hist']
        emb_mask_full = torch.cat([emb_mask_history, emb_mask_current], dim=1).to(device)
        emb_mask_full_emb = self.embedding(emb_mask_full)


        mask_src_cuTu = current_turn.data.eq(config.PAD_idx).unsqueeze(1)


        if(config.noam):
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        # Create an embedding mask for the current turn
        current_turn_mask = self.embedding(batch["mask_input_current_tur"])
        # num_turns = batch["num_turns"]

        # Apply weighting factor to current turn tensor
        weighting_factor = 2

        # define weight vector for history and current turn tensors
        hist_emb = self.embedding(history_turn)
        current_emb = self.embedding(current_turn)
        bat_size2, seq_len2, hid_dim2 = current_emb.size()
        # history_weight = torch.ones(bat_size, seq_len1)
        current_turn_weight = torch.ones(bat_size2, seq_len2) * weighting_factor  # set weight_factor to a value greater than 1

        # multiply current turn tensor by its weight vector
        weighted_current_turn = current_emb * current_turn_weight.unsqueeze(-1).to(device)

        # concatenate history and weighted current turn tensors along sequence length dimension
        full_sequence = torch.cat([hist_emb, weighted_current_turn], dim=1).to(device)

        ful2 = torch.cat([history_turn, current_turn], dim=1).to(device)
        full_sequence_mask = (ful2.eq(config.PAD_idx).unsqueeze(1)).bool()

        encoder_outputs = self.encoder(full_sequence+emb_mask_full_emb,self.embedding(current_turn)+current_turn_mask,mask_src_cuTu, full_sequence_mask)

        q_h = torch.mean(encoder_outputs,dim=1) if config.mean_query else encoder_outputs[:,0]

        logit_prob_stn = self.decoder_stance_key(q_h) #(bsz, num_experts)
        batch_size, num_sentences = enc_batch.size()

        
        if(config.topk_stn>0):
            k_max_value, k_max_index = torch.topk(logit_prob_stn, config.topk_stn)
            a = np.empty([logit_prob_stn.shape[0], self.decoder_stance_num])
            a.fill(float('-inf'))
            mask = torch.Tensor(a).cuda()
            logit_prob_stn2_ = mask.scatter_(1,k_max_index.cuda().long(),k_max_value)
            attention_parameters = self.attention_activation(logit_prob_stn2_)
        else:
            attention_parameters = self.attention_activation(logit_prob_stn)


        if(config.oracle): attention_parameters = self.attention_activation(torch.FloatTensor(batch['stance_program'])*1000)

        attention_parameters_stn = attention_parameters.unsqueeze(-1).unsqueeze(-1) # (batch_size, expert_num, 1, 1)

        weighted_output, encoder_outputs2 = self.mulencoder(full_sequence+emb_mask_full_emb, full_sequence_mask,encoder_outputs,attention_parameters_stn,  full_sequence_mask)

        ## Attention over decoder
        q_h = torch.mean(weighted_output,dim=1) if config.mean_query else weighted_output[:,0]
        #q_h = encoder_outputs[:,0]
        logit_prob = self.decoder_key(q_h) #(bsz, num_experts)

        if(config.topk>0):
            k_max_value, k_max_index = torch.topk(logit_prob, config.topk)
            a = np.empty([logit_prob.shape[0], self.decoder_number])
            a.fill(float('-inf'))
            mask = torch.Tensor(a).cuda()
            logit_prob = mask.scatter_(1,k_max_index.cuda().long(),k_max_value)

        attention_parameters = self.attention_activation(logit_prob)


        if(config.oracle): attention_parameters = self.attention_activation(torch.FloatTensor(batch['target_program'])*1000)

        attention_parameters = attention_parameters.unsqueeze(-1).unsqueeze(-1) # (batch_size, expert_num, 1, 1)

        ys = torch.ones(1, 1).fill_(config.SOS_Rob_idx).long()
        if config.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step+1):
            if(config.project):
                out, attn_dist = self.decoder(self.embedding_proj_in(self.embedding(ys)),self.embedding_proj_in(encoder_outputs2),encoder_outputs2, (full_sequence_mask,mask_trg), attention_parameters)

            else:

                out, attn_dist = self.decoder(self.embedding(ys),encoder_outputs2, (full_sequence_mask,mask_trg), attention_parameters)

            logit2= self.proj(out)
            logit2 = logit2[0, -1, :] / 0.7 # Temp is 0.7

            filtered_logit = top_filtering(logit2, top_k=0, top_p=0.9)
            probs = F.softmax(filtered_logit, dim=-1)


            prev = torch.multinomial(probs, 1)
            if i < 1 and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    if probs.max().item() == 1:
                        warnings.warn("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=1)

            if prev.item() in special_tokens_ids:
                break
            decoded_words.append(['EOS' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in prev.view(-1)])

            prev = prev.data.item()
            if config.USE_CUDA:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(prev).cuda()], dim=1)
                ys = ys.cuda()
            else:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(prev)], dim=1)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == 'EOS': break
                else: st+= e + ' '
            sent.append(st)


        return sent


### CONVERTED FROM https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py#L1062
class ACT_basic(nn.Module):
    def __init__(self,hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size,1)  
        self.p.bias.data.fill_(1) 
        self.threshold = 1 - 0.1

    def forward(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None, decoding=False):
        # init_hdd
        ## [B, S]
        halting_probability = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S
        remainders = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs).cuda()

        step = 0
        # for l in range(self.num_layers):
        while( ((halting_probability<self.threshold) & (n_updates < max_hop)).byte().any()):
            # Add timing signal
            state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)

            p = self.sigma(self.p(state)).squeeze(-1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if(decoding):
                state, _, attention_weight = fn((state,encoder_output,[]))
            else:
                # apply transformation on the state
                state = fn(state)

            # update running part in the weighted state and keep the rest
            previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            if(decoding):
                if(step==0):  previous_att_weight = torch.zeros_like(attention_weight).cuda()      ## [B, S, src_size]
                previous_att_weight = ((attention_weight * update_weights.unsqueeze(-1)) + (previous_att_weight * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop 
            ## to save a line I assigned to previous_state so in the next 
            ## iteration is correct. Notice that indeed we return previous_state
            step+=1

        if(decoding):
            return previous_state, previous_att_weight, (remainders,n_updates)
        else:
            return previous_state, (remainders,n_updates)
