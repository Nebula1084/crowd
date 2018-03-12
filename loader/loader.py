from tflearn.data_utils import pad_sequences

import loader.text as text
from loader.word2vec import Word2Vec


class Loader:

    def __init__(self, embedding_info, text_info):

        """
        :param embedding_info: {path:string, embed_size:int, sequence_length:int}
        :param text_info:  {path:string, sheet:string, text_column:string, label_column:string}
        """
        self.word2vec_model_path = embedding_info['path']
        self.embed_size = embedding_info['embed_size']
        self.sequence_length = embedding_info['sequence_length']
        self.text_type = text_info['type']
        self.text_path = text_info['path']
        self.text_column = text_info['text_column']
        self.label_column = text_info['label_column']
        if self.text_type == 'excel':
            self.text_sheet = text_info['sheet']

    def load(self):
        label = None
        texts = None
        if self.text_type == 'excel':
            texts, label = text.load_excel(self.text_path, self.text_sheet, self.text_column,
                                           self.label_column)
        elif self.text_type == 'csv':
            texts, label = text.load_csv(self.text_path, self.text_column, self.label_column)
        word2vec = Word2Vec(self.word2vec_model_path, self.embed_size)
        word2vec.load()
        data = word2vec.create_indices(texts)
        data = pad_sequences(data, maxlen=self.sequence_length, value=0.)
        return data, label, word2vec
