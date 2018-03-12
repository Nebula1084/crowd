import numpy as np
import yaml

from classifier.bi_lstm import BiLstm
from classifier.trainer import Trainer
from loader.images import load_images
from loader.loader import Loader

gender_mapping = {
    'unknown': 0,
    'male': 1,
    'female': 2,
    'brand': 3,
}


def train(conf):
    loader = Loader(conf['embedding'], conf['text'])
    data, label_str, word2vec = loader.load()

    labels = np.zeros_like(label_str)
    for idx, val in enumerate(label_str):
        if val in gender_mapping:
            labels[idx] = gender_mapping[val]
        else:
            labels[idx] = 0

    classifier = BiLstm(4, conf['embedding']['sequence_length'], word2vec.vocab_size, word2vec.embed_size)
    trainer = Trainer(classifier, word2vec.embeddings)
    trainer.train(data, labels)


def image_train():
    ids, images = load_images('./data/profile_images.pik')
    print(ids.shape)
    print(images.shape)


if __name__ == '__main__':
    conf_file = open('./conf/gender_train.yaml')
    gender_conf = yaml.load(conf_file)
    # train(gender_conf)
    image_train()
