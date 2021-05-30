#encoding=utf8
import os
import sys
import jieba
import pickle
import keras 
import tensorflow as tf 
import numpy as np 
import pandas as pd

from esim import ESIM
from train import esim_params 
from data_helper import pad_sequences
from bm25_retrival import BM25Retrieval


class EntityMatch(object):
    def __init__(self,kb_path):
        super(EntityMatch, self).__init__()
        self.kb_path = kb_path
        self.bm25re = BM25Retrieval(self.kb_path)

        self.word2idx, _ = pickle.load(open("./checkpoint/word2id.pkl","rb"))
        self.model = ESIM(esim_params).build()
        self.model.load_weights('./checkpoint/best_esim_model.h5')

    def char_index(self, p_sentences, h_sentences):

        p_list, h_list = [], []
        for p_sentence, h_sentence in zip(p_sentences, h_sentences):
            p = [self.word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in self.word2idx.keys()]
            h = [self.word2idx[word.lower()] for word in h_sentence if len(word.strip()) > 0 and word.lower() in self.word2idx.keys()]

            p_list.append(p)
            h_list.append(h)

        p_list = pad_sequences(p_list, maxlen=esim_params['input_shapes'][0][0])
        h_list = pad_sequences(h_list, maxlen=esim_params['input_shapes'][0][0])

        return p_list, h_list

    def predict(self,query):
        cand_docs = self.bm25re.retrieval(query,20)
        querys = [query] * len(cand_docs)

        p,h = self.char_index(querys,cand_docs)

        scores = self.model.predict([p,h])
        scores = scores[:,1]
        match_score = {e:s for e,s in zip(cand_docs,scores)}
        match_score = sorted(match_score.items(),key=lambda x:x[1],reverse=True)
        return match_score[0],cand_docs

if __name__ == '__main__':
    emm = EntityMatch("../../yidu-n7k/code.txt")
    test = pd.read_csv("./data/test.csv")
    total = len(test)
    correct = 0
    for raw,norm in test[["sentence1","sentence2"]].values:
        pred,cand_docs = emm.predict(raw)
        if norm == pred[0]:
            correct += 1
        if pred[1] > 0.8 and norm != pred[0]:
            print(raw,norm,"可以做负样本")

    print("\n","acc:",correct/total)