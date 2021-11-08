import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import time

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
        
        # self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size, ohv_len]
        
        # embedded = self.dropout(self.embedding(src))
        # print('embedded size', embedded.size())
    
        #embedded = [src len, batch size, emb dim]
        
        # outputs, (hidden, cell) = self.rnn(embedded)
        # print(src.size())
        outputs, (hidden, cell) = self.rnn(src)
        
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out_intent = nn.Linear(hid_dim, 10)
        self.fc_out_emotion = nn.Linear(hid_dim, 6)
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size, ohv_len]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        # print("input decoder size", input.size())
        
        #input = [1, batch size]
        
        # embedded = self.dropout(self.embedding(input))
        # print("embedding decoder size", embedded.size())
        
        #embedded = [1, batch size, emb dim]
        # print('hidden state input to decoder size:', hidden.size())
        # print('cell state input to decoder size:', cell.size())
                
        # output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        # print('output size in decoder before linear layer',output.size())
        intent_prediction = self.fc_out_intent(output.squeeze(0))
        linearEmotion =self.fc_out_emotion(output.squeeze(0))
        # print('output size of liner layer in decoder for emotion:', linearEmotion)
        emotion_prediction = self.softmax(linearEmotion)
        
        #prediction = [batch size, output dim]
        
        return intent_prediction, emotion_prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src dialogue len, batch size, ohv_size]
        #trg = [trg dialogue len, batch size, ohv_size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        input_length = src.shape[0]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim  #ohv_size
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        print("output predictions stored array size", outputs.size())
        
        # for i in range(input_length):
        #    encoder_hidden, encoder_cell = self.encoder(src[i])

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # hidden, cell = encoder_hidden, encoder_cell
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        # for t in range(1,trg_len):
        for t in range(trg_len):

            output_emotion = torch.zeros(batch_size, 6).to(self.device)
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output_intent, output_emotion_softmax, hidden, cell = self.decoder(input, hidden, cell)
            
            # print("predicted decoder output size: ", output.size())
            #place predictions in a tensor holding predictions for each token
            output_intent = (torch.sigmoid(output_intent) >= 0.5).float()

            output_emotion_idx = torch.argmax(output_emotion_softmax, dim=1)
            print('output emotion index',output_emotion_idx)
            # print('output emotion softmax',output_emotion_softmax)
            i=0
            for emo_idx in output_emotion_idx:
              output_emotion[i][emo_idx] = 1.0
              i+=1

            output = torch.cat((output_intent, output_emotion),1)
            print('output intent size', output_intent.size())
            print('output emotion size', output_emotion.size())
            print('concatenated output size',output.size())
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # print('predicted output size:', output.size())
            #get the highest predicted token from our predictions
            top1 =  output
            print('decoder prediction:', top1)
            # print('post sigmoid activation and thresholding, prediction output size:', top1.size())
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        # print('predicted batch output size:', outputs.size())
        return outputs

