# coding=utf-8
import keras
from crf_layer import CRF

class BiLstmCrfModel(object):
    def __init__(
            self, 
            max_len, 
            vocab_size, 
            embedding_dim, 
            lstm_units, 
            class_nums,
            embedding_matrix=None
        ):
        super(BiLstmCrfModel, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.class_nums = class_nums
        self.embedding_matrix = embedding_matrix
        if self.embedding_matrix is not None:
            self.vocab_size,self.embedding_dim = self.embedding_matrix.shape

    def build(self):
        inputs = keras.layers.Input(
                shape=(self.max_len,), 
                dtype='int32'
            )
        x = keras.layers.Masking(
                mask_value=0
            )(inputs)
        x = keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                trainable=False,
                weights=self.embedding_matrix,
                mask_zero=True
            )(x)
        x = keras.layers.Bidirectional(
                keras.layers.LSTM(
                    self.lstm_units, 
                    return_sequences=True
                )
            )(x)
        x = keras.layers.TimeDistributed(
                keras.layers.Dropout(
                    0.2
                )
            )(x)
        crf = CRF(self.class_nums)
        outputs = crf(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam', 
            loss=crf.loss_function, 
            metrics=[crf.accuracy]
            )
        print(model.summary())

        return model
        