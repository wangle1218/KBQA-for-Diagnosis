#! -*- coding: utf-8 -*-
import keras
from keras.models import Model
from keras import backend as K
import tensorflow as tf 

from modules import IntentAttention,SlotAttention,SlotGate

class SlotGatedSLU(object):
    """implementation SlotGated SLU model for keras
    https://www.aclweb.org/anthology/N18-2118/
    """
    def __init__(self, params):
        super(SlotGatedSLU, self).__init__()
        self._params = params

    def build(self):
        seq_input = keras.layers.Input(
            name='seq_input',
            shape=(self._params['maxlen'],)
            )
        x = self._make_embedding_layer(embed_type='char')(seq_input)
        x = keras.layers.SpatialDropout1D(
                0.1,
                name='embed_drop')(x)

        state_outputs,_fw,fw_final_c,_bw,bw_final_c = keras.layers.Bidirectional(
            keras.layers.LSTM(
                self._params['lstm_units'],
                dropout=self._params['lstm_dropout_rate'],
                return_sequences=True,
                return_state=True,
                ),
            name='bilstm_encoder'
            )(x)

        slot_inputs = state_outputs #[batch_size,maxlen,2*lstm_units]
        intent_input = keras.layers.concatenate(
            [fw_final_c,bw_final_c],
            name='final_state'
            ) #[batch_size,2*lstm_units]

        # 意图识别任务
        intent_attn = self._apply_intent_attn(state_outputs) #[batch_size,maxlen]
        intent_feats = keras.layers.concatenate(
            [intent_input,intent_attn],
            name='intent_feats'
            )
        intent_dense = keras.layers.Dense(
            self._params['intent_dense_size'], 
            activation="relu",
            name="intent_dense"
            )(intent_feats)
        intent_out = keras.layers.Dense(
            self._params['intent_nums'], 
            activation="softmax",
            name="intent_out"
            )(intent_dense)

        # 槽位填充任务
        if self._params['full_attention']:
            slot_attn_out = self._apply_slot_attn(slot_inputs)
            slot_feats = self._apply_slot_gate(
                state_outputs,slot_attn_out,intent_attn)
        else:
            slot_feats = self._apply_slot_gate(
                state_outputs,slot_inputs,intent_attn)
        slot_feats_drop = keras.layers.TimeDistributed(
            keras.layers.Dropout(0.2),
            name='slot_feats_drop'
            )(slot_feats)
        slot_dense = keras.layers.TimeDistributed(
            keras.layers.Dense(
                self._params['slot_dense_size'],
                activation='relu'
                ),
            name='slot_dense'
            )(slot_feats_drop)
        slot_out = keras.layers.TimeDistributed(
            keras.layers.Dense(
                self._params['slot_label_nums'],
                activation='softmax'
                ),
            name='slot_out'
            )(slot_dense)

        # 模型
        model = keras.models.Model(
            inputs=seq_input, 
            outputs=[intent_out,slot_out]
            )
        return model

    def _make_embedding_layer(self,name='embedding',embed_type='char',**kwargs):

        def init_embedding(weights=None):
            if embed_type == "char":
                input_dim = self._params['char_max_features']
                output_dim = self._params['char_embed_size']
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
            embed_weights = self._params['char_embedding_matrix']
        else:
            embed_weights = self._params['word_embedding_matrix']

        if embed_weights is None:
            embedding = init_embedding()
        else:
            embedding = init_embedding(weights = [embed_weights])
        return embedding

    def _apply_intent_attn(self,inputs):
        intent_attn = IntentAttention(self._params['maxlen'],name='intent_attn')(inputs)
        return intent_attn

    def _apply_slot_attn(self,inputs):
        # 将BILSTM编码输出先输入给一个前馈神经网络，不变换维度，然后计算attention
        slot_attn_ffn_size = K.int_shape(inputs)[2]
        slot_ffn = keras.layers.TimeDistributed(
            keras.layers.Dense(
                slot_attn_ffn_size,
                activation='relu'
                ),
            name='slot_ffn'
            )(inputs)
        slot_atten = SlotAttention()(slot_ffn)
        return slot_atten

    def _apply_slot_gate(self,hi,slot_c,intent_c):
        slot_gate = SlotGate(name='slot_gate')([slot_c,intent_c])
        slot_feats = keras.layers.concatenate(
            [hi,slot_gate],
            name='slot_feats'
            )
        return slot_feats



def test():
    import numpy as np

    param = {
        'maxlen':5,
        'char_max_features':50,
        'char_embed_size':10,
        'word_max_features':50,
        'word_embed_size':10,
        'char_embedding_matrix':None,
        'word_embedding_matrix':None,

        'lstm_units':8,
        'lstm_dropout_rate':0.1,
        'intent_dense_size':15,
        'intent_nums':5,
        'full_attention':True,
        'slot_dense_size':15,
        'slot_label_nums':7,
    }
    model = SlotGatedSLU(param).build()
    model.compile(
        optimizer='adam',
        loss={'slot_out':'categorical_crossentropy', 'intent_out':'categorical_crossentropy'},
        loss_weights={'slot_out': 1, 'intent_out': 1}
        )

    print(model.summary())

    trainX = np.array([[0,1,2,3,4],[3,2,5,1,6],[2,2,1,1,4]])
    intent_y = np.array([1,0,2])
    intent_y = keras.utils.to_categorical(intent_y,num_classes=param['intent_nums'])
    slot_y = np.array([[0,1,2,2,0],[3,4,0,1,2],[0,0,1,2,2]])
    slot_y = keras.utils.to_categorical(slot_y,num_classes=param['slot_label_nums'])

    H = model.fit(
            x=trainX,
            y={"slot_out": slot_y, "intent_out": intent_y},
            # validation_data=(
            #     testX,
            #     {"slot_out": testCategoryY, "color_output": testColorY}
            #     ),
            epochs=4,
            verbose=1
            )


if __name__ == '__main__':
    test()
