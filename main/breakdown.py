import pickle
import sys

import numpy as np
import yaml

from classifier.bi_lstm import BiLstm
from classifier.predictor import Predictor
from classifier.trainer import Trainer
from loader.loader import Loader


def train(conf):
    loader = Loader(conf['embedding'], conf['text'])
    data, label_str, word2vec = loader.load()
    data = data[:700]
    labels = np.array(label_str[:700], dtype=np.int32)
    classifier = BiLstm(7, conf['embedding']['sequence_length'], word2vec.vocab_size, word2vec.embed_size)
    trainer = Trainer(classifier, word2vec.embeddings)
    trainer.train(data, labels)


def predict(conf):
    loader = Loader(conf['embedding'], conf['text'])
    data, label_str, word2vec = loader.load()
    classifier = BiLstm(7, conf['embedding']['sequence_length'], word2vec.vocab_size, word2vec.embed_size)
    predictor = Predictor(classifier)
    res = predictor.predict(data)
    f = open('./data/breakdown_predict.pik', 'wb')
    print(res[0])
    pickle.dump(res, f)
    f.close()


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        conf_file = open('./conf/breakdown_train.yaml')
        train(yaml.load(conf_file))
        conf_file.close()
    elif sys.argv[1] == 'predict':
        conf_file = open('./conf/breakdown_predict.yaml')
        predict(yaml.load(conf_file))
        conf_file.close()
