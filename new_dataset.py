from datasets import load_dataset
from torchtext import vocab
import nltk
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
import numpy as np
import torch
import pickle
nltk.download('punkt')

def save_vocab(vocab, path = 'vocab.pickle'):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(path = 'vocab.pickle'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_glove():
    # Get glove embeddings
    glove_embeddings = vocab.GloVe(name='840B', dim=300)
    return glove_embeddings

def get_nli():
    # Load SNLI dataset
    snli_dataset = load_dataset("snli")
    train_data = snli_dataset["train"]
    train_data = train_data.filter(lambda x: x["label"] != -1)
    validation_data = snli_dataset["validation"]
    validation_data = validation_data.filter(lambda x: x["label"] != -1)
    test_data = snli_dataset["test"]
    test_data = test_data.filter(lambda x: x["label"] != -1)
    return train_data, validation_data, test_data

def build_vocab(sentences):
    # Build vocabulary given then tokens in the dataset and glove embeddings
    word_dict = {}
    glove_emb = get_glove()
    token_set = set(glove_emb.stoi.keys())
    for sent in sentences:
        tokens = word_tokenize(sent)
        for token in tokens:
            token = token.lower()
            if token not in word_dict and token in token_set:
                word_dict[token] = glove_emb[token]
    # Add special tokens
    special_tokens = ['<s>', '</s>', '<p>']
    for token in special_tokens:
        if token in token_set:
            word_dict[token] = glove_emb[token]

    print('Found {0} words with glove vectors'.format(len(word_dict)))
    return word_dict

############################################

class SNLIDataset(Dataset):
    def __init__(self, s1, s2, target, word_vec):
        self.s1 = s1
        self.s2 = s2
        self.target = target
        self.word_vec = word_vec

        for split in ['s1', 's2']:
            setattr(self, split, self.tokenize_dataset(getattr(self, split)))

    def __len__(self):
        return len(self.s1)

    def __getitem__(self, idx):
        return {
            's1': self.s1[idx],
            's2': self.s2[idx],
            'target': self.target[idx],
            'word_vec': self.word_vec
        }

    # def __getitem__(self, idx):
    #     return {
    #         's1': ['<s>'] + [token.lower() for token in word_tokenize(self.s1[idx]) if token.lower() in self.word_vec] + ['</s>'],
    #         's2': ['<s>'] + [token.lower() for token in word_tokenize(self.s2[idx]) if token.lower() in self.word_vec] + ['</s>'],
    #         'target': self.target[idx],
    #         'word_vec': self.word_vec
    #     }
    
    # def tokenize_dataset(self, dataset):
    #     return [['<s>'] + [token for token in word_tokenize(sent) if token in self.word_vec] + ['</s>'] for sent in dataset]
    
    def tokenize_dataset(self, dataset):
        return [['<s>'] + [token.lower() for token in word_tokenize(sent) if token.lower() in self.word_vec] + ['</s>'] for sent in dataset]

def collate_fn(batch):
    batch_s1 = [item['s1'] for item in batch]
    batch_s2 = [item['s2'] for item in batch]
    batch_target = [item['target'] for item in batch]
    word_vec = batch[0]['word_vec']

    # Get batch embeddings and lengths using get_batch function
    s1_batch, s1_len = get_batch(batch_s1, word_vec)
    s2_batch, s2_len = get_batch(batch_s2, word_vec)

    return {
        's1_batch': s1_batch,
        's1_len': s1_len,
        's2_batch': s2_batch,
        's2_len': s2_len,
        'target': torch.tensor(batch_target, dtype=torch.long)
    }

def get_batch(batch, word_vec):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 300))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]

    return torch.from_numpy(embed).float(), torch.tensor(lengths)