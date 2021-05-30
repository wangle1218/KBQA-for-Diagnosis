#! -*- coding: utf-8 -*-
import json
import pandas as pd 
import numpy as np 

from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from sklearn.metrics import classification_report
from bert4keras.optimizers import Adam

from bert_model import build_bert_model
from data_helper import load_data

#定义超参数和配置文件
class_nums = 13
maxlen = 60
batch_size = 16

config_path='E:/bert_weight_files/roberta/bert_config_rbt3.json'
checkpoint_path='E:/bert_weight_files/roberta/bert_model.ckpt'
dict_path = 'E:/bert_weight_files/roberta/vocab.txt'

tokenizer = Tokenizer(dict_path)
class data_generator(DataGenerator):
    """
    数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)#[1,3,2,5,9,12,243,0,0,0]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

if __name__ == '__main__':
    # 加载数据集
    train_data = load_data('./data/train.csv')
    test_data = load_data('./data/test.csv')

    # 转换数据集
    train_generator = data_generator(train_data, batch_size)
    test_generator = data_generator(test_data, batch_size)

    model = build_bert_model(config_path,checkpoint_path,class_nums)
    print(model.summary())
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(5e-6), 
        metrics=['accuracy'],
    )

    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        verbose=2, 
        mode='min'
        )
    bast_model_filepath = './checkpoint/best_model.weights'
    checkpoint = keras.callbacks.ModelCheckpoint(
        bast_model_filepath, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True,
        mode='min'
        )

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=test_generator.forfit(), 
        validation_steps=len(test_generator),
        shuffle=True, 
        callbacks=[earlystop,checkpoint]
    )

    model.load_weights(bast_model_filepath)
    test_pred = []
    test_true = []
    for x,y in test_generator:
        p = model.predict(x).argmax(axis=1)
        test_pred.extend(p)

    test_true = test_data[:,1].tolist()
    print(set(test_true))
    print(set(test_pred))

    target_names = [line.strip() for line in open('label','r',encoding='utf8')]
    print(classification_report(test_true, test_pred,target_names=target_names))