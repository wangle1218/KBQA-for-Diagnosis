#! -*- coding: utf-8 -*-
from __future__ import print_function
import json
import numpy as np
import pandas as pd
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam, extend_with_weight_decay
from bert4keras.snippets import DataGenerator
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import text_segmentate
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.snippets import uniout


# bert配置
config_path = '../bert_wwm/bert_config.json'
checkpoint_path = '../bert_wwm/bert_model.ckpt'
dict_path = '../bert_wwm/vocab.txt'

token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

class TotalLoss(Loss):
    """loss分两部分，一是seq2seq的交叉熵，二是相似度的交叉熵。
    """
    def compute_loss(self, inputs, mask=None):
        loss1 = self.compute_loss_of_seq2seq(inputs, mask)
        loss2 = self.compute_loss_of_similarity(inputs, mask)
        self.add_metric(loss1, name='seq2seq_loss')
        self.add_metric(loss2, name='similarity_loss')
        return loss1 + loss2

    def compute_loss_of_seq2seq(self, inputs, mask=None):
        y_true, y_mask, _, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

    def compute_loss_of_similarity(self, inputs, mask=None):
        _, _, y_pred, _ = inputs
        y_true = self.get_labels_of_similarity(y_pred)  # 构建标签
        y_pred = K.l2_normalize(y_pred, axis=1)  # 句向量归一化
        similarities = K.dot(y_pred, K.transpose(y_pred))  # 相似度矩阵
        similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线
        similarities = similarities * 30  # scale
        loss = K.categorical_crossentropy(
            y_true, similarities, from_logits=True
        )
        return loss

    def get_labels_of_similarity(self, y_pred):
        idxs = K.arange(0, K.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        labels = K.equal(idxs_1, idxs_2)
        labels = K.cast(labels, K.floatx())
        return labels

    def get_config(self):
        base_config = super(TotalLoss,self).get_config()
        return dict(list(base_config.items()) )


# 建立加载模型
def bulid_model():
    bert = build_transformer_model(
        config_path,
        checkpoint_path,
        with_pool='linear',
        application='unilm',
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
        return_keras_model=False,
    )

    encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
    seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])

    outputs = TotalLoss([2, 3])(bert.model.inputs + bert.model.outputs)
    model = keras.models.Model(bert.model.inputs, outputs)

    AdamW = extend_with_weight_decay(Adam, 'AdamW')
    optimizer = AdamW(learning_rate=2e-6, weight_decay_rate=0.01)
    model.compile(optimizer=optimizer)

    return model,encoder,seq2seq



