import pickle

import gensim
import numpy as np
import pandas as pd

import nlp.sanitize as st

word2vec_path = './data/GoogleNews-vectors-negative300.bin.gz'
embedding_path = './data/embeddings'
extended_embedding_path = './data/extended_embeddings'

word2vec_cache = None


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list) < 1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


def get_word2vec_embeddings(clean_questions, generate_missing=False):
    global word2vec_cache
    if word2vec_cache is None:
        print("Start to loading word2vec model....")
        word2vec_cache = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    else:
        print("Use cached model....")
    embeddings = clean_questions['tokens'].apply(lambda x: get_average_word2vec(x, word2vec_cache,
                                                                                generate_missing=generate_missing))
    return list(embeddings)


def dump_embeddings(path, embeddings):
    f = open(path, 'wb')
    pickle.dump(embeddings, f)
    f.close()


def load_embeddings(path):
    f = open(path, 'rb')
    embeddings = pickle.load(f)
    f.close()
    return embeddings


if __name__ == '__main__':
    tweet = pd.ExcelFile('data/tweet.xlsx')
    data = tweet.parse('Data')
    data = st.standardize(data, 'tweet_text')

    embedding = get_word2vec_embeddings(data)
    dump_embeddings(embedding_path, embedding)

    tweet_extend = pd.ExcelFile('./data/tweet_extended.xlsx')
    extended_data = tweet_extend.parse('Sheet1')
    extended_data = st.standardize(extended_data, 'tweet_text')

    extended_embedding = get_word2vec_embeddings(extended_data)
    dump_embeddings(extended_embedding_path, extended_embedding)
