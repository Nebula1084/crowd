import gc
import os
import pickle

import gensim
import numpy as np


class Word2Vec(object):
    def __init__(self, path, embed_size):
        self.path = path
        self.word2vec = None
        self.embeddings = None
        self.vocabulary_word2index = None
        self.vocabulary_index2word = None
        self.embed_size = embed_size
        self.vocab_size = 0

    def load(self):
        print("Start to load model from %s" % self.path)
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.path, binary=True)
        self.vocabulary_word2index, self.vocabulary_index2word = self.create_vocabulary()
        self.vocab_size = len(self.vocabulary_index2word)
        self.embeddings = self.create_embeddings()

    def create_vocabulary(self, name_scope='word2vec'):
        cache_path = './data/' + name_scope + "_word_vocabulary.pik"
        print('Cache_path:', cache_path, 'file_exists:', os.path.exists(cache_path))
        if os.path.exists(cache_path):
            print('Use exist vocabulary cache')
            with open(cache_path, 'rb') as data_f:
                vocabulary_word2index, vocabulary_index2word = pickle.load(data_f)
                return vocabulary_word2index, vocabulary_index2word
        else:
            print('Create new vocabulary')
            vocabulary_word2index = {'PAD_ID': 0, 'EOS': 1}
            vocabulary_index2word = {0: 'PAD_ID', 1: 'EOS'}
            special_index = 1
            for i, vocab in enumerate(self.word2vec.vocab):
                vocabulary_word2index[vocab] = i + 1 + special_index
                vocabulary_index2word[i + 1 + special_index] = vocab

            with open(cache_path, 'wb') as data_f:
                pickle.dump((vocabulary_word2index, vocabulary_index2word), data_f)
        return vocabulary_word2index, vocabulary_index2word

    def create_indices(self, text):
        print('Create new indices')
        indices = []
        for i, sentence in enumerate(text):
            index = [self.vocabulary_word2index.get(word, 0) for word in sentence]
            indices.append(index)

        return np.array(indices)

    def create_embeddings(self):
        print('Start to create embeddings')
        count_exist = 0
        count_not_exist = 0
        word_embedding = [[]] * self.vocab_size  # create an empty word_embedding list.
        word_embedding[0] = np.zeros(self.embed_size)  # assign empty for first word:'PAD'
        bound = np.sqrt(6.0) / np.sqrt(self.vocab_size)  # bound for random variables.
        for i in range(1, self.vocab_size):  # loop each word
            word = self.vocabulary_index2word[i]  # get a word
            # noinspection PyBroadException
            try:
                embedding = self.word2vec[word]  # try to get vector:it is an array.
            except Exception:
                embedding = None
            if embedding is not None:  # the 'word' exist a embedding
                word_embedding[i] = embedding
                count_exist = count_exist + 1  # assign array to this word.
            else:  # no embedding for this word
                word_embedding[i] = np.random.uniform(-bound, bound, self.embed_size)
                count_not_exist = count_not_exist + 1  # init a random value for the word.
        del self.word2vec

        gc.collect()
        word_embedding_final = np.array(word_embedding)  # covert to 2d array.

        return word_embedding_final
