from __future__ import print_function, division

import torch
import torchtext

import seq2seq
from seq2seq.loss import NLLLoss, PositiveLoss

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64, output_vocab=None):
        self.loss = loss
        self.batch_size = batch_size
        self.output_vocab = output_vocab

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]

        with torch.no_grad():
            for batch in batch_iterator:
                input_variables, input_lengths  = getattr(batch, seq2seq.src_field_name)
                target_variables = getattr(batch, seq2seq.tgt_field_name)

                decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

                # Evaluation
                seqlist = other['sequence']
                for step, step_output in enumerate(decoder_outputs):
                    target = target_variables[:, step + 1]
                    loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                    non_padding = target.ne(pad)
                    correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy
    
    def evaluate_reward(self, model, data, vocab):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()             

        loss = PositiveLoss()
        loss.reset()

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]

        with torch.no_grad():
            for batch in batch_iterator:
                input_variable, input_lengths  = getattr(batch, seq2seq.src_field_name)
                target_variable = getattr(batch, seq2seq.tgt_field_name)

                decoder_output, decoder_hidden, other = model(input_variable, input_lengths.tolist(), target_variable)
        
                seqlist = []
                tensorlist = []

                for step, step_output in enumerate(decoder_output):
                    batch_size = target_variable.size(0)
                    #loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variable[:, step + 1])
                    #step_output.contiguous().view(batch_size, -1)
                    tensorlist.append(torch.max(step_output, dim=1)[0])
                    seqlist.append(torch.max(step_output, dim=1)[1])

                log_tensor = torch.stack(tensorlist, dim=1)        
                output_tensor = torch.stack(seqlist, dim=1)

                loss.eval_batch(log_tensor, output_tensor, target_variable, vocab)

        return loss.get_loss()
