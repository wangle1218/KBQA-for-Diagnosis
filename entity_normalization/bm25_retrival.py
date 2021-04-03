#encoding=utf8
import os
import sys
import jieba
import pickle
from gensim.summarization import bm25


class BM25Retrieval(object):
    """docstring for BM25Retrieval"""
    def __init__(self, path):
        super(BM25Retrieval, self).__init__()
        self.path = path
        self.kb_entitys = self.load_corpus(self.path)
        self.bm25Model = bm25.BM25([list(i) for i in self.kb_entitys])

    def load_corpus(self,path):
        kb_entitys = []
        with open(path,encoding='utf8') as f:
            for line in f.readlines():
                code,name = line.strip().split('\t')
                kb_entitys.append(name)

        return kb_entitys

    def retrieval(self,query,top_k):
        scores = self.bm25Model.get_scores(query)
        match_score = {e:s for e,s in zip(self.kb_entitys,scores)}
        match_score = sorted(match_score.items(),key=lambda x:x[1],reverse=True)
        return [i[0] for i in match_score[:top_k]]