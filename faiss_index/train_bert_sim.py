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

# 基本信息
maxlen = 48
batch_size = 16
steps_per_epoch = 2000
epochs = 20

# bert配置
config_path = 'E:/bert_weight_files/bert_wwm/bert_config.json'
checkpoint_path = 'E:/bert_weight_files/bert_wwm/bert_model.ckpt'
dict_path = 'E:/bert_weight_files/bert_wwm/vocab.txt'

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


# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


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
                text, synonym, maxlen=maxlen * 2
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            token_ids, segment_ids = tokenizer.encode(
                synonym, text, maxlen=maxlen * 2
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


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
    model.summary()

    return model,encoder,seq2seq


def make_test_data():
    test_data = load_data('./input/test.csv',train=False)
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

a_token_ids,b_token_ids,labels = make_test_data()
model,encoder,seq2seq = bulid_model()

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
            model.save_weights('./output/best_model.weights')
            print("Acc 从 %.5g 提升到 %.5g" % (self.acc,acc))
            self.acc = acc


if __name__ == '__main__':
    # 加载数据集
    train_data = load_data('./input/train.csv')
    valid_data = load_data('./input/dev.csv')
    
    train_generator = data_generator(train_data+valid_data, batch_size)
    evaluator = Evaluate()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

    model_frame_path = "./output/bert_sim_model.json"
    model_json = model.to_json()
    with open(model_frame_path, "w") as json_file:
        json_file.write(model_json)

    model_frame_path = "./output/bert_sim_encoder.json"
    model_json = encoder.to_json()
    with open(model_frame_path, "w") as json_file:
        json_file.write(model_json)

else:

    model.load_weights('./output/best_model.weights')
