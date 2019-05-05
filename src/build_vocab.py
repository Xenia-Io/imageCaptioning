import os
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import nltk

class vocab():
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.count = 0
        self.word2id["<pad>"] = self.count
        self.id2word[self.count] = "<pad>"
        self.count = self.count + 1
        self.word2id["<start>"] = self.count
        self.id2word[self.count] = "<start>"
        self.count = self.count + 1
        self.word2id["<end>"] = self.count
        self.id2word[self.count] = "<end>"
        self.count = self.count + 1
        self.word2id["<unk>"] = self.count
        self.id2word[self.count] = "<unk>"
        self.count = self.count + 1

    def add_word(self,word):
        self.word2id[word] = self.count
        self.id2word[self.count] = word
        self.count = self.count + 1

def build_vocab(csvfile, vocab_size,dataset_dir):
    train_df = pd.read_csv(csvfile)
    caption_list = train_df['caption']
    wordcount_dict = defaultdict(int)
    for caption in caption_list:
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        for token in tokens:
            wordcount_dict[token] += 1
    sorted_wordlist = []
    for k, _ in sorted(wordcount_dict.items(), key=lambda x: -x[1]):
        sorted_wordlist.append(k)
    vocabulary = vocab()
    for word in sorted_wordlist[:vocab_size]:
        vocabulary.add_word(word)
    vocab_path = dataset_dir + "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocabulary, f)

if __name__ == "__main__":
    dataset_dir = "/Users/shiprajain/ImageCaptioning/data"
    train_csv = "/Users/shiprajain/ImageCaptioning/data/train/image_captions_train.csv"
    vocab_size = 10000
    build_vocab(train_csv, vocab_size,dataset_dir)
    
    
