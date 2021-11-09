import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import time
import torch.nn.functional as F

# SEED = 1234

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # since one hot vectors are given as input, they can be considered as an embedded representation of the input sequence
        # Hence, embedding layer is not required
        # self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.gru = nn.GRU(emb_dim, hid_dim, n_layers,
                          dropout=dropout, bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, hidden= None):
        
        # outputs, (hidden, cell) = self.rnn(src) # pass input directly to lstm 
        outputs, hidden = self.gru(src, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hid_dim] +
                   outputs[:, :, self.hid_dim:])
        
        # return hidden, cell
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.attention = Attention(hid_dim)
        self.gru = nn.GRU(hid_dim + emb_dim, hid_dim,
                          n_layers, dropout=dropout)
        
        # separate classifiers for intent and emotion
        self.fc_out_intent = nn.Linear(hid_dim * 2, 10)
        self.fc_out_emotion = nn.Linear(hid_dim * 2, 6)
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, input, last_hidden, encoder_outputs):
        
        
        input = input.unsqueeze(0)
        #input = [1, batch size]  --> dimension
        
        # output, (hidden, cell) = self.rnn(input, (hidden, cell)) # pass input directly to lstm
        
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([input, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        dec_output = torch.cat([output, context], 1)

        # Pass decoded outputs through both classifiers
        intent_prediction = self.fc_out_intent(dec_output)
        linearEmotion =self.fc_out_emotion(dec_output)
        
        # intent_prediction = self.fc_out_intent(output.squeeze(0))
        # linearEmotion =self.fc_out_emotion(output.squeeze(0))
        
        #prediction = [batch size, output dim]  --> dimension

        # return linear classifier outputs for intent and emotion along with decoder hidden state and memory cell
        # return intent_prediction, linearEmotion, hidden, cell
        return intent_prediction, linearEmotion, hidden, attn_weights

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.hidden_size = hid_dim
        self.attn = nn.Linear(self.hidden_size * 2, hid_dim)
        self.v = nn.Parameter(torch.rand(hid_dim))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.softmax = nn.LogSoftmax(dim=1)  # for emotion classifier output
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src dialogue len, batch size, ohv_size] --> dimension
        #trg = [trg dialogue len, batch size, ohv_size] --> dimension
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        input_length = src.shape[0]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim  #ohv_size
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device) # to store outputs for loss function
        outputs_predictions = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device) # to store processed predicted one hot vectors

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        # hidden, cell = self.encoder(src)
        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        
        #first input to the decoder is the farst target utterance label vector
        input = trg[0,:]
        
        # for t in length of sequence of dialogue:
        for t in range(trg_len):

            output_emotion = torch.zeros(batch_size, 6).to(self.device) # tensor for tracking emotion label
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (linear predictions) and new hidden and cell states
            output_intent, linearEmotion, hidden, attn_weights = self.decoder(input, hidden, encoder_output)
            
            
            # compute predicted intent label from linear intent output
            output_intent_sig = (torch.sigmoid(output_intent) >= 0.5).float()

            # compute log softmax activation for linear emotion output
            output_emotion_softmax = self.softmax(linearEmotion)
            # compute predicted emotion label
            output_emotion_prob = torch.exp(output_emotion_softmax)
            output_emotion_idx = torch.argmax(output_emotion_prob, dim=1)
            
            # for each emotion index for the sequence, initialize output_emotion tensor
            i=0
            for emo_idx in output_emotion_idx:
              output_emotion[i][emo_idx] = 1.0
              i+=1

            # combine intent and emotion outputs for loss computation
            output = torch.cat((output_intent, output_emotion_softmax),1)
            # combine intent and emotion output predictions for accuracy computation
            output_top1 = torch.cat((output_intent_sig, output_emotion),1)
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # initialize the predicted output for next possible input to decoder, depending on teacher forcing ratio
            top1 =  output_top1
            
            # append sequence-wise outputs for loss funciton
            outputs[t] = output
            # append sequence-wise outputs for accuracy computation
            outputs_predictions[t] = top1
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs, outputs_predictions

