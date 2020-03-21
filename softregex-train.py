#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torchtext

import seq2seq

from seq2seq.trainer import SupervisedTrainer, SelfCriticalTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq, TopKDecoder
from seq2seq.loss import Perplexity, NLLLoss, PositiveLoss
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor, Evaluator
from seq2seq.util.checkpoint import Checkpoint
import torch.nn.functional as F
import sys


# In[13]:

dataset = 'kb13'

if len(sys.argv) < 1:
    sys.exit(-1)

dataset = sys.argv[1]

import warnings
warnings.filterwarnings('ignore')


# In[14]:


try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


# In[17]:


# Prepare dataset
src = SourceField()
tgt = TargetField()

# data/kb/train/data.txt
#data/NL-RX-Synth/train/data.txt
#data/NL-RX-Turk/train/data.txt

datasets = {
    'kb13': ('KB13', 35, 60),
    'NL-RX-Synth': ('NL-RX-Synth', 10, 40),
    'NL-RX-Turk': ('NL-RX-Turk', 10, 40)
}

data_tuple = datasets[dataset]

# max_len = 60
max_len = data_tuple[2]
def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len
train = torchtext.data.TabularDataset(
    path='data/' + data_tuple[0] + '/train/data.txt', format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)
dev = torchtext.data.TabularDataset(
    path='data/' + data_tuple[0] + '/val/data.txt', format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)
test = torchtext.data.TabularDataset(
    path='data/' + data_tuple[0] + '/test/data.txt', format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)
src.build_vocab(train, max_size=500)
tgt.build_vocab(train, max_size=500)
input_vocab = src.vocab
output_vocab = tgt.vocab


# In[18]:


# Prepare loss
weight = torch.ones(len(tgt.vocab))
pad = tgt.vocab.stoi[tgt.pad_token]

loss = NLLLoss(weight, pad)

if torch.cuda.is_available():
    loss.cuda()
    
seq2seq_model = None
optimizer = None


# In[19]:


hidden_size = 256
word_embedding_size = 128

bidirectional = True

encoder = EncoderRNN(len(src.vocab), max_len, hidden_size, dropout_p=0.1,rnn_cell='lstm',
                     bidirectional=bidirectional, n_layers=2, variable_lengths=True)
decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,rnn_cell='lstm',
                     dropout_p=0.25, use_attention=True, bidirectional=bidirectional, n_layers=2,
                     eos_id=tgt.eos_id, sos_id=tgt.sos_id)

seq2seq_model = Seq2seq(encoder, decoder)
if torch.cuda.is_available():
    seq2seq_model.cuda()

for param in seq2seq_model.parameters():
    param.data.uniform_(-0.1, 0.1)


optimizer = Optimizer(torch.optim.Adam(seq2seq_model.parameters()),  max_grad_norm=5)


# In[20]:


seq2seq_model = torch.nn.DataParallel(seq2seq_model)


# In[21]:


# train

t = SupervisedTrainer(loss=loss, batch_size=8,
                      checkpoint_every=200,
                      print_every=10000, expt_dir='./lstm_model/'+data_tuple[0]+'/Deepregex')


# In[22]:


seq2seq_model = t.train(seq2seq_model, train,
                  num_epochs=data_tuple[1], dev_data=dev,
                  optimizer=optimizer,
                  teacher_forcing_ratio=0.5,
                  resume=False)


# ### Self Critical Training

# In[23]:


class compare_regex(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, target_size):
        super(compare_regex, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embed = Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm1 = LSTM(embedding_dim ,hidden_dim, bidirectional=True, num_layers=1, batch_first=True)
        self.lstm2 = LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=1, batch_first=True)
        self.fc1 = Linear(hidden_dim*2*2, 60)
        self.fc2 = Linear(60, 20)
        self.fc3 = Linear(20, target_size)

        
    def init_hidden(self, bs):
        if torch.cuda.is_available():
            return (torch.zeros(2, bs, self.hidden_dim).cuda(),
                   torch.zeros(2, bs, self.hidden_dim).cuda())
        else:
            return (torch.zeros(2, bs, self.hidden_dim),
                   torch.zeros(2, bs, self.hidden_dim))
    
    def forward(self, bs, line1, line2, input1_lengths,input2_lengths):
        embeded1 = self.embed(line1)
        embeded2 = self.embed(line2)

        hidden1 = self.init_hidden(bs)
        lstm1_out, last_hidden1 = self.lstm1(embeded1,hidden1)
        hidden2 = self.init_hidden(bs)
        lstm2_out, last_hidden2 = self.lstm2(embeded2,hidden2)


        fc1_out = self.fc1(torch.cat((lstm1_out.mean(1), lstm2_out.mean(1)),1))  #encoder outputs 평균값 concat 97.8%

        
        fc1_out = F.tanh(fc1_out)
        fc2_out = self.fc2(fc1_out)
        fc2_out = F.tanh(fc2_out)
        fc3_out = self.fc3(fc2_out)
        score = F.log_softmax(fc3_out,dim=1)
        return score


# In[24]:


f = open('./regex_equal_model/compare_vocab.txt')
sc_loss_vocab = dict()
for line in f.read().splitlines():
    line = line.split('\t')
    sc_loss_vocab[line[0]] = int(line[1])
f.close()
compare_regex_model = torch.load('./regex_equal_model/compare_regex_model.pth')
compare_regex_model.eval()


# In[25]:


optimizer_new = Optimizer(torch.optim.Adadelta(seq2seq_model.parameters(), lr=0.05))

#if you want to train by oracle, put mode to None
sc_t = SelfCriticalTrainer(loss=PositiveLoss(mode='prob', prob_model=compare_regex_model, loss_vocab=sc_loss_vocab), batch_size=32,
                           checkpoint_every=100, print_every=100, expt_dir='./lstm_model/'+data_tuple[0]+'/SoftRegex', output_vocab=output_vocab)



seq2seq_model = sc_t.train(seq2seq_model, train,
                  num_epochs=30, dev_data=dev,
                  optimizer=optimizer_new, teacher_forcing_ratio=0.5,
                  resume=False)


# In[26]:


evaluator = Evaluator()


# In[27]:


evaluator.evaluate(seq2seq_model, dev) # (5.799417234628771, 0.6468332123976366)