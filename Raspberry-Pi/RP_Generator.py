import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import keras
from keras.utils import to_categorical

import time

import tqdm
import os
import dill as pickle
from pathlib import Path
import random

import pandas as pd
from math import floor
from genmidi import Midi
from music import NoteSeq, Note
import music21
import random
import os, argparse
from midi2audio import FluidSynth

from mido import MidiFile

os.chdir('/home/pi/Raspberry_Piano/')

dtype = torch.float
device = torch.device("cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print('Available Device:', device)

#@title (OPTIONAL) Download ready-to-use Piano and Chamber Notewise DataSets
'''
sudo wget 'https://github.com/asigalov61/SuperPiano/raw/master/Super%20Chamber%20Piano%20Violin%20Notewise%20DataSet.zip'
sudo unzip 'Super Chamber Piano Violin Notewise DataSet.zip'
sudo rm 'Super Chamber Piano Violin Notewise DataSet.zip'
#@title Load and Encode TXT Notes DataSet'''
select_training_dataset_file = "/home/pi/Raspberry_Piano/notewise_chamber.txt" #@param {type:"string"}

# replace with any text file containing full set of data
MIDI_data = select_training_dataset_file

print('Loading Dataset. Please wait...')

with open(MIDI_data, 'r') as file:
    text = file.read()
print('Encoding Dataset. Please wait...')
# get vocabulary set
words = sorted(tuple(set(text.split())))
n = len(words)

# create word-integer encoder/decoder
word2int = dict(zip(words, list(range(n))))
int2word = dict(zip(list(range(n)), words))

# encode all words in dataset into integers
encoded = np.array([word2int[word] for word in text.split()])
print('Loading Model Functions. Please wait...')
"""# Main Model Setup"""

#@title Define all functions
# define model using the pytorch nn module
class WordLSTM(nn.ModuleList):
    
    def __init__(self, sequence_len, vocab_size, hidden_dim, batch_size):
        super(WordLSTM, self).__init__()
        
        # init the hyperparameters
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        
        # first layer lstm cell
        self.lstm_1 = nn.LSTMCell(input_size=vocab_size, hidden_size=hidden_dim)
        
        # second layer lstm cell
        self.lstm_2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # third layer lstm cell
        #self.lstm_3 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # dropout layer
        self.dropout = nn.Dropout(p=0.35)
        
        # fully connected layer
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)
        
    # forward pass in training   
    def forward(self, x, hc):
        """
            accepts 2 arguments: 
            1. x: input of each batch 
                - shape 128*149 (batch_size*vocab_size)
            2. hc: tuple of init hidden, cell states 
                - each of shape 128*512 (batch_size*hidden_dim)
        """
        
        # create empty output seq
        output_seq = torch.empty((self.sequence_len,
                                  self.batch_size,
                                  self.vocab_size))
        # if using gpu        
        output_seq = output_seq.to(device)
        
        # init hidden, cell states for lstm layers
        hc_1, hc_2, hc_3 = hc, hc, hc
        
        # for t-th word in every sequence 
        for t in range(self.sequence_len):
            
            # layer 1 lstm
            hc_1 = self.lstm_1(x[t], hc_1)
            h_1, c_1 = hc_1
            
            # layer 2 lstm
            hc_2 = self.lstm_2(h_1, hc_2)
            h_2, c_2 = hc_2

            # layer 3 lstm
            #hc_3 = self.lstm_3(h_2, hc_3)
            #h_3, c_3 = hc_3
            
            # dropout and fully connected layer
            output_seq[t] = self.fc(self.dropout(h_2))
            
        return output_seq.view((self.sequence_len * self.batch_size, -1))
          
    def init_hidden(self):
        
        # initialize hidden, cell states for training
        # if using gpu
        return (torch.zeros(self.batch_size, self.hidden_dim).to(device),
                torch.zeros(self.batch_size, self.hidden_dim).to(device))
    
    def init_hidden_generator(self):
        
        # initialize hidden, cell states for prediction of 1 sequence
        # if using gpu
        return (torch.zeros(1, self.hidden_dim).to('cpu'),
                torch.zeros(1, self.hidden_dim).to('cpu'))
    
    def predict(self, seed_seq, top_k=5, pred_len=128):
        """
            accepts 3 arguments: 
            1. seed_seq: seed string sequence for prediction (prompt)
            2. top_k: top k words to sample prediction from
            3. pred_len: number of words to generate after the seed seq
        """
        
        # set evaluation mode
        self.eval()
        
        # split string into list of words
        seed_seq = seed_seq.split()
        
        # get seed sequence length
        seed_len = len(seed_seq)
        
        # create output sequence
        out_seq = np.empty(seed_len+pred_len)
        
        # append input seq to output seq
        out_seq[:seed_len] = np.array([word2int[word] for word in seed_seq])
 
        # init hidden, cell states for generation
        hc = self.init_hidden_generator()
        hc_1, hc_2, hc_3 = hc, hc, hc
        
        # feed seed string into lstm
        # get the hidden state set up
        for word in seed_seq[:-1]:
            
            # encode starting word to one-hot encoding
            word = to_categorical(word2int[word], num_classes=self.vocab_size)

            # add batch dimension
            word = torch.from_numpy(word).unsqueeze(0)
            # if using gpu
            word = word.to('cpu') 
            
            # layer 1 lstm
            hc_1 = self.lstm_1(word, hc_1)
            h_1, c_1 = hc_1
            
            # layer 2 lstm
            hc_2 = self.lstm_2(h_1, hc_2)
            h_2, c_2 = hc_2

            # layer 3 lstm
            #hc_3 = self.lstm_3(h_2, hc_3)
            #h_3, c_3 = hc_3            

        word = seed_seq[-1]
        
        # encode starting word to one-hot encoding
        word = to_categorical(word2int[word], num_classes=self.vocab_size)

        # add batch dimension
        word = torch.from_numpy(word).unsqueeze(0)
        # if using gpu
        word = word.to('cpu') 

        # forward pass
        for t in range(pred_len):
            
            # layer 1 lstm
            hc_1 = self.lstm_1(word, hc_1)
            h_1, c_1 = hc_1
            
            # layer 2 lstm
            hc_2 = self.lstm_2(h_1, hc_2)
            h_2, c_2 = hc_2

            # layer 3 lstm
            #hc_3 = self.lstm_3(h_2, hc_3)
            #h_3, c_3 = hc_3
            
            # fully connected layer without dropout (no need)
            output = self.fc(h_2)
            
            # software to get probabilities of output options
            output = F.softmax(output, dim=1)
            
            # get top k words and corresponding probabilities
            p, top_word = output.topk(top_k)
            # if using gpu           
            p = p.cpu()
            
            # sample from top k words to get next word
            p = p.detach().squeeze().numpy()
            top_word = torch.squeeze(top_word)
            
            word = np.random.choice(top_word, p = p/p.sum())
            
            # add word to sequence
            out_seq[seed_len+t] = word
            
            # encode predicted word to one-hot encoding for next step
            word = to_categorical(word, num_classes=self.vocab_size)
            word = torch.from_numpy(word).unsqueeze(0)
            # if using gpu
            word = word.to('cpu')
            
        return out_seq


def get_batches(arr, n_seqs, n_words):
    """
        create generator object that returns batches of input (x) and target (y).
        x of each batch has shape 128*128*149 (batch_size*seq_len*vocab_size).
        
        accepts 3 arguments:
        1. arr: array of words from text data
        2. n_seq: number of sequence in each batch (aka batch_size)
        3. n_word: number of words in each sequence
    """
    
    # compute total elements / dimension of each batch
    batch_total = n_seqs * n_words
    
    # compute total number of complete batches
    n_batches = arr.size//batch_total
    
    # chop array at the last full batch
    arr = arr[: n_batches* batch_total]
    
    # reshape array to matrix with rows = no. of seq in one batch
    arr = arr.reshape((n_seqs, -1))
    
    # for each n_words in every row of the dataset
    for n in range(0, arr.shape[1], n_words):
        
        # chop it vertically, to get the input sequences
        x = arr[:, n:n+n_words]
        
        # init y - target with shape same as x
        y = np.zeros_like(x)
        
        # targets obtained by shifting by one
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, n+n_words]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        
        # yield function is like return, but creates a generator object
        yield x, y

#@title Compile the Model
training_batch_size = 256 #@param {type:"slider", min:0, max:1024, step:16}
attention_span_in_tokens = 512 #@param {type:"slider", min:0, max:512, step:64}
hidden_dimension_size = 512 #@param {type:"slider", min:0, max:512, step:64}
test_validation_ratio = 0.1 #@param {type:"slider", min:0, max:1, step:0.1}
learning_rate = 0.001 #@param {type:"number"}

print('Compiling NN. Please wait...')
# compile the network - sequence_len, vocab_size, hidden_dim, batch_size
net = WordLSTM(sequence_len=attention_span_in_tokens, vocab_size=len(word2int), hidden_dim=hidden_dimension_size, batch_size=training_batch_size)
# if using gpu
net.to(device)

# define the loss and the optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# split dataset into 90% train and 10% using index
val_idx = int(len(encoded) * (1 - test_validation_ratio))
train_data, val_data = encoded[:val_idx], encoded[val_idx:]

# empty list for the validation losses
val_losses = list()

# empty list for the samples
samples = list()
'''
#@title Download Raspberry Piano Pre-Trained Chamber Model
wget 'https://github.com/Tegridy-Code/Project-Hellraiser/raw/main/trained_model.zip'
unzip 'trained_model.zip'''

print('Loading Model. Please wait...')
#@title Load downloaded/pre-trained Model checkpoint
model = torch.load('/home/pi/Raspberry_Piano/trained_model.h5', map_location='cpu')
model.eval()



# Commented out IPython magic to ensure Python compatibility.
#@title Generate TXT and MIDI file
seed_prompt = "p24" #@param {type:"string"}
tokens_to_generate = 8192 #@param {type:"slider", min:0, max:8192, step:16}
time_coefficient = 3 #@param {type:"slider", min:1, max:16, step:1}
top_k_coefficient = 30 #@param {type:"slider", min:2, max:50, step:1}

print('Generating composition. Please wait...')
# Generating MIDI
with open("./output.txt", "w") as outfile:
    outfile.write(' '.join([int2word[int_] for int_ in model.predict(seed_seq=seed_prompt, pred_len=tokens_to_generate, top_k=top_k_coefficient)]))

#creating MIDI
sample_freq_variable = 12 #@param {type:"number"}
note_range_variable = 62 #@param {type:"number"}
note_offset_variable = 33 #@param {type:"number"}
number_of_instruments = 2 #@param {type:"number"}
chamber_option = True #@param {type:"boolean"}
# default settings: sample_freq=12, note_range=62

def decoder(filename):
    
    filedir = './'

    notetxt = filedir + filename

    with open(notetxt, 'r') as file:
        notestring=file.read()

    score_note = notestring.split(" ")

    # define some parameters (from encoding script)
    sample_freq=sample_freq_variable
    note_range=note_range_variable
    note_offset=note_offset_variable
    chamber=chamber_option
    numInstruments=number_of_instruments

    # define variables and lists needed for chord decoding
    speed=time_coefficient/sample_freq
    piano_notes=[]
    violin_notes=[]
    time_offset=0

    # start decoding here
    score = score_note

    i=0

    # for outlier cases, not seen in sonat-1.txt
    # not exactly sure what scores would have "p_octave_" or "eoc" (end of chord?)
    # it seems to insert new notes to the score whenever these conditions are met
    while i<len(score):
        if score[i][:9]=="p_octave_":
            add_wait=""
            if score[i][-3:]=="eoc":
                add_wait="eoc"
                score[i]=score[i][:-3]
            this_note=score[i][9:]
            score[i]="p"+this_note
            score.insert(i+1, "p"+str(int(this_note)+12)+add_wait)
            i+=1
        i+=1


    # loop through every event in the score
    for i in tqdm.auto.tqdm(range(len(score))):

        # if the event is a blank, space, "eos" or unknown, skip and go to next event
        if score[i] in ["", " ", "<eos>", "<unk>"]:
            continue

        # if the event starts with 'end' indicating an end of note
        elif score[i][:3]=="end":

            # if the event additionally ends with eoc, increare the time offset by 1
            if score[i][-3:]=="eoc":
                time_offset+=1
            continue

        # if the event is wait, increase the timestamp by the number after the "wait"
        elif score[i][:4]=="wait":
            time_offset+=int(score[i][4:])
            continue

        # in this block, we are looking for notes   
        else:
            # Look ahead to see if an end<noteid> was generated
            # soon after.  
            duration=1
            has_end=False
            note_string_len = len(score[i])
            for j in range(1,200):
                if i+j==len(score):
                    break
                if score[i+j][:4]=="wait":
                    duration+=int(score[i+j][4:])
                if score[i+j][:3+note_string_len]=="end"+score[i] or score[i+j][:note_string_len]==score[i]:
                    has_end=True
                    break
                if score[i+j][-3:]=="eoc":
                    duration+=1

            if not has_end:
                duration=12

            add_wait = 0
            if score[i][-3:]=="eoc":
                score[i]=score[i][:-3]
                add_wait = 1

            try: 
                new_note=music21.note.Note(int(score[i][1:])+note_offset)    
                new_note.duration = music21.duration.Duration(duration*speed)
                new_note.offset=time_offset*speed
                if score[i][0]=="v":
                    violin_notes.append(new_note)
                else:
                    piano_notes.append(new_note)                
            except:
                print("Unknown note: " + score[i])




            time_offset+=add_wait

    # list of all notes for each instrument should be ready at this stage

    # creating music21 instrument objects      
    
    piano=music21.instrument.fromString("Piano")
    violin=music21.instrument.fromString("Violin")

    # insert instrument object to start (0 index) of notes list
    
    piano_notes.insert(0, piano)
    violin_notes.insert(0, violin)
    # create music21 stream object for individual instruments
    
    piano_stream=music21.stream.Stream(piano_notes)
    violin_stream=music21.stream.Stream(violin_notes)
    # merge both stream objects into a single stream of 2 instruments
    note_stream = music21.stream.Stream([piano_stream, violin_stream])

    
    note_stream.write('midi', fp="./"+filename[:-4]+".mid")
    
    print("Done! Decoded midi file saved to './home/pi/Raspberry_Piano/'")

print('Creating MIDI from TXT output. Please wait...')    
decoder('output.txt')

print('Rendering WAV from output MIDI. Please wait...')
FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio('./output.mid', './output.wav')
