from __future__ import absolute_import, division, unicode_literals
import sys
import io
import numpy as np
import logging
import torch
import argparse
from nltk.tokenize import word_tokenize
from new_models import NLINet
import os

# import own files
# from main import *

# Set PATHs
PATH_TO_SENTEVAL = './SenteevalData'
PATH_TO_DATA = './.SenteevalData'
from new_dataset import get_glove, get_nli
# PATH_TO_VEC = './SentEval/pretrained/glove.840B.300d.txt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import SentEval.senteval.engine as senteval2


# added global variables for the model and embedding dimension
MODEL = None
EMBED_DIM = 300

# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        # tokens = word_tokenize(s)
        for word in s:
            word = word.lower()
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(word2id):
    word_vec = {}
    glove_emb = get_glove()
    for token in glove_emb.stoi:
        if token in word2id:
            word_vec[token] = glove_emb[token]
    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec

# SentEval prepare and batcher
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(params.word2id)
    params.wvec_dim = EMBED_DIM
    return

def batcher(params, batch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(torch.tensor(params.word_vec[word]))
        if not sentvec:
            vec = torch.zeros((1, 300))
            sentvec.append(vec)
        sentvec = torch.stack(sentvec, dim=0).squeeze(1)
        embeddings.append(sentvec)

    # pad into tensor
    sentence_lengths = torch.tensor([x.shape[0] for x in embeddings]).to(device)
    if all(length == sentence_lengths[0] for length in sentence_lengths):
        # stack the list of sequences [1,300] to [len, 1, 300]
        embeddings = torch.stack(embeddings, dim=0).to(device)
    else:
        embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, padding_value=0.0, batch_first=True).to(device)
    
    # pass through the model
    embeddings = MODEL.encoder( (embeddings.transpose(0,-2).float(), sentence_lengths) )

    # cast back to numpy
    embeddings = embeddings.cpu().detach().numpy()

    # return the embeddings
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# For command line activation
if __name__ == "__main__":
    # added parser for selecting the model
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', default='AWESentenceEncoder', type=str,
                        help='What model to use. Default is AWE',
                        choices=['AWESentenceEncoder', 'BLSTMEncoder_max', 'BLSTMEncoder_last', 'LSTMEncoder'])
    parser.add_argument('--results_path', default='results', type=str)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set the global variables
    if (args.model == 'AWESentenceEncoder'):
        EMBED_DIM = 300
        modelpath = 'savedir/AWESentenceEncoder_last_model.pickle'
    elif (args.model == 'LSTMEncoder'):
        EMBED_DIM = 2048
        modelpath = 'savedir/LSTMEncoder_last_model.pickle'
    elif (args.model == 'BLSTMEncoder_max'):
        EMBED_DIM = 2*2048
        modelpath = 'savedir/BLSTMEncoder_max_model.pickle'
    else:
        modelpath = 'savedir/BLSTMEncoder_last_model.pickle'
        EMBED_DIM = 2*2048
    MODEL = torch.load(modelpath, map_location=torch.device(device))

    # run the senteval
    se = senteval2.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS14','MR', 'CR', 'MPQA', 'SUBJ', 'SST5', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness','TREC']
    results = se.eval(transfer_tasks)

    # save the results
    os.makedirs(args.results_path, exist_ok=True)
    torch.save(results, os.path.join(args.results_path, args.model + "_SentEvalResults.pt"))

    # print the results
    print(results)