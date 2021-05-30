#! -*- coding: utf-8 -*-
# SimBERT训练代码
# 训练环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.7.7

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

from bert_sim_model import *

# 基本信息
maxlen = 48
batch_size = 16
steps_per_epoch = 2000
epochs = 20

def load_data(filename,train=True):
    df = pd.read_csv(filename,header=0,encoding='utf8')
    f = df.values
    D = []
    for i,l in enumerate(f):
        text1, text2, label = l
        if train:
            if int(label) == 1:
                D.append((text1, text2, int(label)))
        else:
            D.append((text1, text2, int(label)))
    return D

def truncate(text):
    """截断句子
    """
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    return text_segmentate(text, maxlen - 2, seps, strips)[0]

class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.some_samples = []

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (text1, text2, label) in self.sample(random):
            text, synonym = text1, text2
            text, synonym = truncate(text), truncate(synonym)
            self.some_samples.append(text)
            if len(self.some_samples) > 1000:
                self.some_samples.pop(0)
            token_ids, segment_ids = tokenizer.encode(
                text, synonym, max_length=maxlen * 2
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            token_ids, segment_ids = tokenizer.encode(
                synonym, text, max_length=maxlen * 2
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []

class Evaluate(keras.callbacks.Callback):
    """评估模型
    """
    def __init__(self):
        self.acc = 0

    def on_epoch_end(self, epoch, logs=None):
        # model.save_weights('../output/latest_model.weights')
        # 保存最优
        a_vecs = encoder.predict([a_token_ids, np.zeros_like(a_token_ids)],
                                 verbose=True)
        b_vecs = encoder.predict([b_token_ids, np.zeros_like(b_token_ids)],
                                 verbose=True)
        a_vecs = a_vecs / (a_vecs**2).sum(axis=1, keepdims=True)**0.5
        b_vecs = b_vecs / (b_vecs**2).sum(axis=1, keepdims=True)**0.5
        sims = (a_vecs * b_vecs).sum(axis=1)

        acc = ((sims > 0.93) == labels.astype('bool')).mean()

        if acc > self.acc:
            model.save_weights('../output/best_model.weights')
            print("Acc 从 %.5g 提升到 %.5g" % (self.acc,acc))
            self.acc = acc


def make_eval_data():
    test_data = load_data('../input/test.csv',train=False)
    a_token_ids, b_token_ids, labels = [], [], []

    for d in test_data:
        token_ids = tokenizer.encode(d[0], maxlen=maxlen)[0]
        a_token_ids.append(token_ids)
        token_ids = tokenizer.encode(d[1], maxlen=maxlen)[0]
        b_token_ids.append(token_ids)
        labels.append(d[2])

    labels = np.array(labels)
    a_token_ids = sequence_padding(a_token_ids)
    b_token_ids = sequence_padding(b_token_ids)
    return a_token_ids,b_token_ids,labels

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

a_token_ids,b_token_ids,labels = make_eval_data()
model,encoder,seq2seq = bulid_model()

if __name__ == '__main__':
    # 加载数据集
    train_data = load_data('../input/train.csv')
    valid_data = load_data('../input/dev.csv')
    
    train_generator = data_generator(train_data+valid_data, batch_size)
    evaluator = Evaluate()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )
