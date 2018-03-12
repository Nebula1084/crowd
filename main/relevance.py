import numpy as np
import yaml

from classifier.bi_lstm import BiLstm
from classifier.trainer import Trainer
from loader.loader import Loader


def train(conf):
    loader = Loader(conf['embedding'], conf['text'])
    data, label_str, word2vec = loader.load()
    data = data[:700]
    labels = np.array(label_str[:700], dtype=np.int32)
    classifier = BiLstm(2, conf['embedding']['sequence_length'], word2vec.vocab_size, word2vec.embed_size)
    trainer = Trainer(classifier, word2vec.embeddings)
    trainer.train(data, labels)


if __name__ == '__main__':
    conf_file = open('./conf/relevance_train.yaml')
    relevance_conf = yaml.load(conf_file)
    train(relevance_conf)
