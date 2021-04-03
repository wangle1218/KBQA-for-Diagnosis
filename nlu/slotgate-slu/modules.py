#! -*- coding: utf-8 -*-
import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
import tensorflow as tf

class IntentAttention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(IntentAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape =(input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name),
                                     shape=(input_shape[1],),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                            K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

    def get_config(self):
        config = {'step_dim':self.step_dim}
        base_config = super(IntentAttention,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SlotAttention(object):
    """https://www.aclweb.org/anthology/N18-2118/"""
    def __call__(self, inputs):
        attention = keras.layers.Lambda(self._attention,
                                        output_shape = self._attention_output_shape,
                                        arguments = None,
                                        name = 'slot_attn_weights')(inputs)

        align = keras.layers.Lambda(self._soft_alignment,
                                     output_shape = self._soft_alignment_output_shape,
                                     arguments = None,
                                     name = 'slot_attn')([attention, inputs])
        return align

    def _attention(self, inputs):
        """
        Compute the attention between elements of one sentences self with the dot
        product.
        Args:
            inputs: A sentence encoded by a BiLSTM.
        Returns:
            A tensor containing the dot product (attention weights between the
            elements of the sentences self).
        """
        attn_weights = K.batch_dot(x=inputs,
                                   y=K.permute_dimensions(inputs,
                                                          pattern=(0, 2, 1)))
        return K.permute_dimensions(attn_weights, (0, 2, 1))

    def _attention_output_shape(self, inputs):
        input_shape = inputs
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
        

class SlotGate(Layer):
    """
    g =\sum v·tanh(c^S_i + W ·c^I)，对c^I乘以权重W进行线性变换，将维度转换和slot_c的单个step一致
    slot_c 维度：[batch_size,maxlen,2*lstm_units]
    """
    def __init__(self, **kwargs):
        super(SlotGate, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W', 
                              shape=(input_shape[1][1], input_shape[0][2]),
                              initializer=initializers.get('glorot_uniform'),
                              trainable=True)
        self.v = self.add_weight(name='v', 
                              shape=(input_shape[0][2],),
                              initializer=initializers.get('glorot_uniform'),
                              trainable=True)
        super(SlotGate, self).build(input_shape) 

    def call(self, x):
        # 返回 g.c^S_i
        cS,cI = x[0],x[1]
        intent_gate = tf.raw_ops.MatMul(a=cI, b=self.W)
        intent_gate_shape = K.int_shape(intent_gate)
        intent_gate = K.reshape(intent_gate,(-1,1,intent_gate_shape[1]))
        slot_gate = self.v * K.tanh(cS + intent_gate)
        slot_gate = K.cast(K.sum(slot_gate, axis=2, keepdims=True) + K.epsilon(), K.floatx())
        # slot_gate = K.expand_dims(slot_gate,-1)
        return cS*slot_gate

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2])