from __future__ import print_function
import math
import torch.nn as nn
import numpy as np

import torch

class Loss(object):
    """ Base class for encapsulation of the loss functions.

    This class defines interfaces that are commonly used with loss functions
    in training and inferencing.  For information regarding individual loss
    functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.

    Attributes:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.  Implementation depends on individual
            sub-classes.
        acc_loss (int or torcn.nn.Tensor): variable that stores accumulated loss.
        norm_term (float): normalization term that can be used to calculate
            the loss of multiple batches.  Implementation depends on individual
            sub-classes.
    """

    def __init__(self, name, criterion):
        self.name = name
        self.criterion = criterion
        if not issubclass(type(self.criterion), nn.modules.loss._Loss):
            raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
        # accumulated loss
        self.acc_loss = 0
        # normalization term
        self.norm_term = 0

    def reset(self):
        """ Reset the accumulated loss. """
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        """ Get the loss.

        This method defines how to calculate the averaged loss given the
        accumulated loss and the normalization term.  Override to define your
        own logic.

        Returns:
            loss (float): value of the loss.
        """
        raise NotImplementedError

    def eval_batch(self, outputs, target):
        """ Evaluate and accumulate loss given outputs and expected results.

        This method is called after each batch with the batch outputs and
        the target (expected) results.  The loss and normalization term are
        accumulated in this method.  Override it to define your own accumulation
        method.

        Args:
            outputs (torch.Tensor): outputs of a batch.
            target (torch.Tensor): expected output of a batch.
        """
        raise NotImplementedError

    def cuda(self):
        self.criterion.cuda()

    def backward(self):
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate.")
        self.acc_loss.backward()
        

import subprocess

def score_by_example(gold, predicted, num_examples=1):
    #print(gold)
    # print(predicted)
    
    gold = unprocess_regex(gold)
    predicted = unprocess_regex(predicted)
    
    if gold == predicted:
        return 1
    try:
        count = 0
        for i in range(num_examples):
            example = subprocess.check_output(['java', '-jar', 'random_generate.jar', '{}'.format(gold)])
            result = subprocess.check_output(['java', '-jar', 'membership.jar', '{}'.format(predicted), '{}'.format(example[:-1].decode('utf-8'))])
            #print(gold, result, example[:-1], predicted)
            if result == b'true\n':
                count = count + 1
                
        return count / num_examples
        
    except Exception as e:
        return 0
    return 0

def score_by_oracle(gold, predicted):
    if gold == predicted:
        return 1
    try:
        score = int(subprocess.check_output(['python2', 'regexDFAEquals.py', '--gold', '{}'.format(gold), '--predicted', '{}'.format(predicted)], timeout=0.01).decode('utf-8')[0])
    except subprocess.TimeoutExpired:
        return 0
    return score

def to_pad(a,max_len=40):
    sh = a.shape
    sh = torch.Size([sh[0], 60-sh[1]])
    return torch.cat((a, torch.ones(sh, device='cuda').type(torch.cuda.LongTensor)), dim=1)[:,:max_len]
    


def score_by_probability_batch(golds,predicts, prob_model=None,vocab=None):
    max_len=40
    predicts=to_pad(predicts)
    golds=to_pad(golds)
    score = prob_model(len(golds), predicts.type(torch.LongTensor).cuda(), golds.type(torch.LongTensor).cuda(), (predicts!=1).sum(1).tolist(), (golds!=1).sum(1).tolist())
    score_list = (torch.max(torch.exp(score),1)[0].float()*torch.max(torch.exp(score),1)[1].float()).tolist()
    return score_list
#     except:
#         return [0 for i in range(len(golds))]

def score_by_probability(gold, predicted, prob_model = None, vocab = None):
    if gold == predicted:
        return 1
    gold = gold.split(' ')
    gold_len = len(gold)
    predicted = predicted.split(' ')
    predicted = [x for x in predicted if x]
    predicted_len = len(predicted)
    try:
        max_len = 40
        if len(gold) > max_len or len(predicted) > max_len:
            return 0
        gold_padded = gold+['<pad>']*(max_len-len(gold))
        gold_idx_input = [[vocab[i] for i in gold_padded]]

        predicted_padded = predicted+['<pad>']*(max_len-len(predicted))
        predicted_idx_input = [[vocab[i] for i in predicted_padded]]

        score = prob_model(1, torch.LongTensor(predicted_idx_input).cuda(), torch.LongTensor(gold_idx_input).cuda(), [predicted_len], [gold_len])
        score = math.exp(score[0][1])
    except KeyError:
        score = 0
    if score-0.5 > 0:
        score = score
    else:
        score=0
#     print("score_by oracle probability : {}, predicted: {}, score: {}".format(''.join(gold), ''.join(predicted), score))
    return score

def score_by_pos_and_neg_example(gold, predicted, num_examples=10):
    # print(gold)
    # print(predicted)
    if gold == predicted:
        return 1
    gold_tmp = unprocess_regex(gold)
    predicted = unprocess_regex(predicted)
    gold = gold_tmp
    neg_gold = unprocess_regex( '~(' + gold_tmp + ')'+'& (<LET>|<NUM>)')

    try:
        for i in range(num_examples):
            pos_example = subprocess.check_output(['java', '-jar', 'random_generate.jar', '{}'.format(gold)])
            if pos_example == b'\n':
                return 1
            pos_result = subprocess.check_output(['java', '-jar', 'membership.jar', '{}'.format(predicted), '{}'.format(pos_example[:-1].decode('utf-8'))])
            if pos_result == b'false\n':
                return 0
            neg_example = subprocess.check_output(['java', '-jar', 'random_generate.jar', '{}'.format(neg_gold)])
            if neg_example == b'\n':
                return 0
            neg_result = subprocess.check_output(['java', '-jar', 'membership.jar', '{}'.format(predicted), '{}'.format(neg_example[:-1].decode('utf-8'))])
#             print(result, example[:-1], predicted)
            if neg_result == b'true\n':
                return 0
        return 1
    except Exception as e:
        return 0
    return 0

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

def check_dfa_equality(gold, predicted):    
    try:
        result = subprocess.check_output(['python2', 'regexDFAEquals.py', '--gold', '{}'.format(gold), '--predicted', '{}'.format(predicted)], timeout=5)
    except subprocess.TimeoutExpired as exc:
        print("Command timeout: {}".format(exc))
        return 0
    else:
        if str(result)[2] == '1':
            return 0.5
        else:
            return -1.0

def unprocess_regex(regex):
    # regex = regex.replace("<VOW>", " ".join('[AEIOUaeiou]'))
    # regex = regex.replace("<NUM>", " ".join('[0-9]'))
    # regex = regex.replace("<LET>", " ".join('[A-Za-z]'))
    # regex = regex.replace("<CAP>", " ".join('[A-Z]'))
    # regex = regex.replace("<LOW>", " ".join('[a-z]'))

    regex = regex.replace("<VOW>", " ".join('AEIOUaeiou'))
    regex = regex.replace("<NUM>", " ".join('0-9'))
    regex = regex.replace("<LET>", " ".join('A-Za-z'))
    regex = regex.replace("<CAP>", " ".join('A-Z'))
    regex = regex.replace("<LOW>", " ".join('a-z'))

    regex = regex.replace("<M0>", " ".join('dog'))
    regex = regex.replace("<M1>", " ".join('truck'))
    regex = regex.replace("<M2>", " ".join('ring'))
    regex = regex.replace("<M3>", " ".join('lake'))

    regex = regex.replace(" ", "")

    return regex


class SelfCriticalLoss(object):
    """ Base class for encapsulation of the loss functions.

    This class defines interfaces that are commonly used with loss functions
    in training and inferencing.  For information regarding individual loss
    functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.

    Attributes:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.  Implementation depends on individual
            sub-classes.
        acc_loss (int or torcn.nn.Tensor): variable that stores accumulated loss.
        norm_term (float): normalization term that can be used to calculate
            the loss of multiple batches.  Implementation depends on individual
            sub-classes.
    """

    def __init__(self, name):
        self.name = name
        # accumulated loss
        self.acc_loss = 0
        # normalization term
        self.norm_term = 0

    def reset(self):
        """ Reset the accumulated loss. """
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        """ Get the loss.

        This method defines how to calculate the averaged loss given the
        accumulated loss and the normalization term.  Override to define your
        own logic.

        Returns:
            loss (float): value of the loss.
        """
        raise NotImplementedError

    def eval_batch(self, outputs, target):
        """ Evaluate and accumulate loss given outputs and expected results.

        This method is called after each batch with the batch outputs and
        the target (expected) results.  The loss and normalization term are
        accumulated in this method.  Override it to define your own accumulation
        method.

        Args:
            outputs (torch.Tensor): outputs of a batch.
            target (torch.Tensor): expected output of a batch.
        """
        raise NotImplementedError


    def backward(self):
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate.")
        self.acc_loss.backward()
        
class PositiveLoss(SelfCriticalLoss):
    """ Compute the reward of generated sequecnces by checking the acceptance of positive random strings

    Args:
    """

    _NAME = "Random Positive Acceptance Reward"

    def __init__(self,mode = None, prob_model=None,loss_vocab=None):
        self.prob_model = prob_model
        self.loss_vocab = loss_vocab
        self.mode = mode
        super(PositiveLoss, self).__init__(
            self._NAME)        

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss.data.item()
        return loss
    
    def decode_tensor(self, tensor):
        wordlist = [self.vocab.itos[tok] for tok in tensor if self.vocab.itos[tok] != '<eos>' and self.vocab.itos[tok] != '<sos>' and self.vocab.itos[tok] != '<pad>']
        return ' '.join(wordlist)

    def eval_batch(self, logs, outputs, targets, vocab):
        self.vocab = vocab

        reward = []

        if self.mode == 'prob':
            for i in range(logs.shape[0]):
                reward.append(score_by_probability(self.decode_tensor(targets[i]), self.decode_tensor(outputs[i]), prob_model = self.prob_model, vocab=self.loss_vocab))        
        elif self.mode == 'dis':
            for i in range(logs.shape[0]):
                reward.append(score_by_pos_and_neg_example(self.decode_tensor(targets[i]), self.decode_tensor(outputs[i])))
        else:
            for i in range(logs.shape[0]):
                reward.append(score_by_oracle(self.decode_tensor(targets[i]), self.decode_tensor(outputs[i])))
        
        reward_matrix = torch.from_numpy(np.repeat(np.array(reward)[:, np.newaxis], outputs.shape[1], 1)).float().cuda()
        mask = (outputs>1).float().cuda()
        logs = logs.cuda()

        self.acc_loss += torch.sum(- logs * reward_matrix * mask)
        self.norm_term += 1
        

class NLLLoss(Loss):
    """ Batch averaged negative log-likelihood loss.

    Args:
        weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
        size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
    """

    _NAME = "Avg NLLLoss"

    def __init__(self, weight=None, mask=None, size_average=True):
        self.mask = mask
        self.size_average = size_average
        if mask is not None:
            if weight is None:
                raise ValueError("Must provide weight with a mask.")
            weight[mask] = 0

        super(NLLLoss, self).__init__(
            self._NAME,
            nn.NLLLoss(weight=weight, size_average=size_average))

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss.data.item()
        if self.size_average:
            # average loss per batch
            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        target=target.to('cuda')
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1

class Perplexity(NLLLoss):
    """ Language model perplexity loss.

    Perplexity is the token averaged likelihood.  When the averaging options are the
    same, it is the exponential of negative log-likelihood.

    Args:
        weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
    """

    _NAME = "Perplexity"
    _MAX_EXP = 100

    def __init__(self, weight=None, mask=None):
        super(Perplexity, self).__init__(weight=weight, mask=mask, size_average=False)

    def eval_batch(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        if self.mask is None:
            self.norm_term += np.prod(target.size())
        else:
            self.norm_term += target.data.ne(self.mask).sum()

    def get_loss(self):
        nll = super(Perplexity, self).get_loss()
        nll /= self.norm_term.item()
        if nll > Perplexity._MAX_EXP:
            print("WARNING: Loss exceeded maximum value, capping to e^100")
            return math.exp(Perplexity._MAX_EXP)
        return math.exp(nll)