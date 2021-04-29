#encoding=utf8
import keras
import keras.backend as K
from keras.models import Model
import tensorflow as tf 
import numpy as np 
import sys


class ESIM(object):
    def __init__( self, params):
        """Init."""
        self._params = params

    def make_embedding_layer(self,name='embedding',embed_type='char',**kwargs):

        def init_embedding(weights=None):
            if embed_type == "char":
                input_dim = self._params['max_features']
                output_dim = self._params['embed_size']
            else:
                input_dim = self._params['word_max_features']
                output_dim = self._params['word_embed_size']

            return keras.layers.Embedding(
                input_dim = input_dim,
                output_dim = output_dim,
                trainable = True,
                name = name,
                weights = weights,
                **kwargs)

        if embed_type == "char":
            embed_weights = self._params['embedding_matrix']
        else:
            embed_weights = self._params['word_embedding_matrix']

        if embed_weights == []:
            embedding = init_embedding()
        else:
            embedding = init_embedding(weights = [embed_weights])

        return embedding

    def _make_multi_layer_perceptron_layer(self) -> keras.layers.Layer:
        # TODO: do not create new layers for a second call
        def _wrapper(x):
            activation = self._params['mlp_activation_func']
            for _ in range(self._params['mlp_num_layers']):
                x = keras.layers.Dense(self._params['mlp_num_units'],
                                       activation=activation)(x)
            return keras.layers.Dense(self._params['mlp_num_fan_out'],
                                      activation=activation)(x)

        return _wrapper

    def _make_inputs(self) -> list:
        input_left = keras.layers.Input(
            name='text_left',
            shape=self._params['input_shapes'][0]
        )
        input_right = keras.layers.Input(
            name='text_right',
            shape=self._params['input_shapes'][1]
        )
        return [input_left, input_right]

    def _make_output_layer(self) -> keras.layers.Layer:
        """:return: a correctly shaped keras dense layer for model output."""
        task = self._params['task']
        if task == "Classification":
            return keras.layers.Dense(self._params['num_classes'], activation='softmax')
        elif task == "Ranking":
            return keras.layers.Dense(1, activation='linear')
        else:
            raise ValueError(f"{task} is not a valid task type."
                             f"Must be in `Ranking` and `Classification`.")

    def build(self):
        """
        Build the model.
        """
        a, b = self._make_inputs()

        # ---------- Embedding layer ---------- #
        embedding = self.make_embedding_layer()
        embedded_a = embedding(a)
        embedded_b = embedding(b)

        # ---------- Encoding layer ---------- #
        # encoded_a = keras.layers.Bidirectional(keras.layers.LSTM(
        #     self._params['lstm_units'],
        #     return_sequences=True,
        #     dropout=self._params['dropout_rate']
        # ))(embedded_a)
        # encoded_b = keras.layers.Bidirectional(keras.layers.LSTM(
        #     self._params['lstm_units'],
        #     return_sequences=True,
        #     dropout=self._params['dropout_rate']
        # ))(embedded_b)

        bilstm = keras.layers.Bidirectional(keras.layers.LSTM(
                    self._params['lstm_units'],
                    return_sequences=True,
                    dropout=self._params['dropout_rate']
                ))

        encoded_a = bilstm(embedded_a)
        encoded_b = bilstm(embedded_b)

        # ---------- Local inference layer ---------- #
        atten_a, atten_b = SoftAttention()([encoded_a, encoded_b])

        sub_a_atten = keras.layers.Lambda(lambda x: x[0]-x[1])([encoded_a, atten_a])
        sub_b_atten = keras.layers.Lambda(lambda x: x[0]-x[1])([encoded_b, atten_b])

        mul_a_atten = keras.layers.Lambda(lambda x: x[0]*x[1])([encoded_a, atten_a])
        mul_b_atten = keras.layers.Lambda(lambda x: x[0]*x[1])([encoded_b, atten_b])

        m_a = keras.layers.concatenate([encoded_a, atten_a, sub_a_atten, mul_a_atten])
        m_b = keras.layers.concatenate([encoded_b, atten_b, sub_b_atten, mul_b_atten])

        # ---------- Inference composition layer ---------- #
        composition_a = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['lstm_units'],
            return_sequences=True,
            dropout=self._params['dropout_rate']
        ))(m_a)

        avg_pool_a = keras.layers.GlobalAveragePooling1D()(composition_a)
        max_pool_a = keras.layers.GlobalMaxPooling1D()(composition_a)

        composition_b = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['lstm_units'],
            return_sequences=True,
            dropout=self._params['dropout_rate']
        ))(m_b)

        avg_pool_b = keras.layers.GlobalAveragePooling1D()(composition_b)
        max_pool_b = keras.layers.GlobalMaxPooling1D()(composition_b)
        print(K.int_shape(composition_b))
        print(K.int_shape(avg_pool_b))
        

        pooled = keras.layers.concatenate([avg_pool_a, max_pool_a, avg_pool_b, max_pool_b])
        pooled = keras.layers.Dropout(rate=self._params['dropout_rate'])(pooled)

        # ---------- Classification layer ---------- #
        mlp = self._make_multi_layer_perceptron_layer()(pooled)
        mlp = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(mlp)

        prediction = self._make_output_layer()(mlp)

        model = Model(inputs=[a, b], outputs=prediction)

        return model
        
class SoftAttention(object):
    """
    Layer to compute local inference between two encoded sentences a and b.
    """

    def __call__(self, inputs):
        a = inputs[0]
        b = inputs[1]

        attention = keras.layers.Lambda(self._attention,
                                        output_shape = self._attention_output_shape,
                                        arguments = None)(inputs)

        align_a = keras.layers.Lambda(self._soft_alignment,
                                     output_shape = self._soft_alignment_output_shape,
                                     arguments = None)([attention, b])
        align_b = keras.layers.Lambda(self._soft_alignment,
                                     output_shape = self._soft_alignment_output_shape,
                                     arguments = None)([attention, a])

        return align_a, align_b

    def _attention(self, inputs):
        """
        Compute the attention between elements of two sentences with the dot
        product.
        Args:
            inputs: A list containing two elements, one for the first sentence
                    and one for the second, both encoded by a BiLSTM.
        Returns:
            A tensor containing the dot product (attention weights between the
            elements of the two sentences).
        """
        attn_weights = K.batch_dot(x=inputs[0],
                                   y=K.permute_dimensions(inputs[1],
                                                          pattern=(0, 2, 1)))
        return K.permute_dimensions(attn_weights, (0, 2, 1))

    def _attention_output_shape(self, inputs):
        input_shape = inputs[0]
        embedding_size = input_shape[1]
        return (input_shape[0], embedding_size, embedding_size)

    def _soft_alignment(self, inputs):
        """
        Compute the soft alignment between the elements of two sentences.
        Args:
            inputs: A list of two elements, the first is a tensor of attention
                    weights, the second is the encoded sentence on which to
                    compute the alignments.
        Returns:
            A tensor containing the alignments.
        """
        attention = inputs[0]
        sentence = inputs[1]

        # Subtract the max. from the attention weights to avoid overflows.
        exp = K.exp(attention - K.max(attention, axis=-1, keepdims=True))
        exp_sum = K.sum(exp, axis=-1, keepdims=True)
        softmax = exp / exp_sum

        return K.batch_dot(softmax, sentence)

    def _soft_alignment_output_shape(self, inputs):
        attention_shape = inputs[0]
        sentence_shape = inputs[1]
        return (attention_shape[0], attention_shape[1], sentence_shape[2])
