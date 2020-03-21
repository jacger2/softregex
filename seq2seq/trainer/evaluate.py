from __future__ import print_function, division

import torch
import torchtext

import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torchtext

import seq2seq
#from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq, TopKDecoder
from seq2seq.loss import Perplexity, NLLLoss, PositiveLoss
#from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor, Evaluator
from seq2seq.util.checkpoint import Checkpoint

import subprocess

def decode_tensor(tensor, vocab):
    tensor = tensor.view(-1)
    words = []
    for i in tensor:
        word = vocab.itos[i.cpu().numpy()]
        if word == '<eos>':
            return ' '.join(words) 
        if word != '<sos>' and word != '<pad>' and word != '<eos>':
            words.append(word)
        #if word != '<sos>':
        #    words.append(word)
        #print('|' + word + '|')
    return ' '.join(words)

from regexDFAEquals import regex_equiv_from_raw, unprocess_regex, regex_equiv

def refine_outout(regex):
    par_list = []
    word_list = regex.split()
    
    for idx, word in enumerate(word_list):
        if word == '(' or word == '[' or word == '{':
            par_list.append(word)

        if word == ')' or word == ']' or word == '}':
            if len(par_list) == 0:
                word_list[idx] = ''
                continue

            par_in_list = par_list.pop()
            if par_in_list == '(':
                word_list[idx] = ')'
            elif par_in_list == '[':
                word_list[idx] = ']'
            elif par_in_list == '{':
                word_list[idx] = '}'
            
    while len(par_list) != 0:
        par_in_list = par_list.pop()
        if par_in_list == '(':
            word_list.append(')')
        elif par_in_list == '[':
            word_list.append(']')
        elif par_in_list == '{':
            word_list.append('}')
            
    word_list = [word for word in word_list if word != '']
    
    return ' '.join(word_list)

def eval_fa_equiv(model, data, input_vocab, output_vocab):
    loss = NLLLoss()
    batch_size = 1

    model.eval()

    loss.reset()
    match = 0
    total = 0

    device = None if torch.cuda.is_available() else -1
    batch_iterator = torchtext.data.BucketIterator(
        dataset=data, batch_size=batch_size,
        sort=False, sort_key=lambda x: len(x.src),
        device=device, train=False)
    tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
    pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]

    predictor = Predictor(model, input_vocab, output_vocab)

    num_samples = 0
    perfect_samples = 0
    dfa_perfect_samples = 0

    match = 0
    total = 0

    with torch.no_grad():
        for batch in batch_iterator:
            num_samples = num_samples + 1

            input_variables, input_lengths  = getattr(batch, seq2seq.src_field_name)

            target_variables = getattr(batch, seq2seq.tgt_field_name)



            target_string = decode_tensor(target_variables, output_vocab)

            #target_string = target_string + " <eos>"

            input_string = decode_tensor(input_variables, input_vocab)

            generated_string = ' '.join([x for x in predictor.predict(input_string.strip().split())[:-1] if x != '<pad>'])


            #str(pos_example)[2]

            generated_string = refine_outout(generated_string)

            #str(pos_example)[2]

            pos_example = subprocess.check_output(['python2', 'regexDFAEquals.py', '--gold', '{}'.format(target_string), '--predicted', '{}'.format(generated_string)])

            if target_string == generated_string:
                perfect_samples = perfect_samples + 1
                dfa_perfect_samples = dfa_perfect_samples + 1
            elif str(pos_example)[2] == '1':
                dfa_perfect_samples = dfa_perfect_samples + 1



            target_tokens = target_string.split()
            generated_tokens = generated_string.split()

            shorter_len = min(len(target_tokens), len(generated_tokens))

            for idx in range(len(generated_tokens)):
                total = total + 1

                if idx >= len(target_tokens):
                    total = total + 1
                elif target_tokens[idx] == generated_tokens[idx]:
                    match = match + 1


            if total == 0:
                accuracy = float('nan')
            else:
                accuracy = match / total

            string_accuracy = perfect_samples / num_samples
            dfa_accuracy = dfa_perfect_samples /num_samples

        f=open('./time_logs/log_score_time.txt','a')
        f.write('{}\n'.format(dfa_accuracy))
        f.close()