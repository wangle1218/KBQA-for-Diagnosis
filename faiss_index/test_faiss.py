#! -*- coding: utf-8 -*-
import sys
import os
import pickle
import numpy as np
import pandas as pd
import faiss
import keras
from bert4keras.snippets import sequence_padding,text_segmentate
from bert_sim_model import bulid_model,tokenizer

maxlen = 48

model,encoder,_ = bulid_model()
model.load_weights('./output/best_model.weights')

def load_data(out_path):
    # train_df = pd.read_csv('./input/train.csv',header=0,encoding='utf8')
    valid_df = pd.read_csv('./input/dev.csv',header=0,encoding='utf8')
    # df = pd.concat([train_df,valid_df],axis=0)
    df = valid_df.loc[valid_df['label']==1,:]
    data = pd.DataFrame()
    data['sentence'] = df['sentence1'].tolist() + df['sentence2'].tolist()
    data.drop_duplicates(subset=['sentence'],inplace=True)
    idx = np.array(range(len(data))).astype(int)
    data['idx'] = idx
    data.to_csv(out_path,index=False)
    return data

def encode_sentense(data_df):
    # 测试相似度效果
    a_token_ids = []

    for d in data_df['sentence'].tolist():
        token_ids = tokenizer.encode(truncate(d), maxlen=maxlen)[0]
        a_token_ids.append(token_ids)

    a_token_ids = sequence_padding(a_token_ids)
    # 将句子进行向量化表达
    a_vecs = encoder.predict([a_token_ids, np.zeros_like(a_token_ids)],
                             verbose=True)
    pickle.dump(a_vecs,open('./output/sent_vec.pkl','wb'))

    return a_vecs


def build_faiss_index(faiss_index_path, database_path):
    
    if not os.path.exists('./output/sent_vec.pkl'):
        df = load_data(database_path)
        xb = encode_sentense(df)
    else:
        df = pd.read_csv(database_path,header=0)
        xb = pickle.load(open('./output/sent_vec.pkl','rb'))
    xb = xb.astype(np.float32)
    idx = df['idx'].values

    _index = faiss.IndexFlatL2(xb.shape[1])
    index = faiss.IndexIDMap(_index) 
    index.add_with_ids(xb,idx)

    print("training index")
    # index.train(xb)
    # index.add(xb)
    print(index.ntotal) 
    print("write ",faiss_index_path)
    faiss.write_index(index, faiss_index_path)
    

def l2_norm(vecs):
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

def truncate(text):
    """截断句子
    """
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    return text_segmentate(text, maxlen - 2, seps, strips)[0]

class Searcher(object):
    """docstring for Searcher"""
    def __init__(self, faiss_index_path, database_path):
        super(Searcher, self).__init__()
        self.faiss_index_path = faiss_index_path
        self.database_path = database_path

        if not os.path.exists(self.faiss_index_path) or not os.path.exists(self.database_path):
            build_faiss_index(self.faiss_index_path, self.database_path)

        self.database = pd.read_csv(self.database_path,header=0)
        self.index = faiss.read_index(self.faiss_index_path)
        # self.index.nprobe = 100

    def encode_sentense(self, sentense):
        sentense = truncate(sentense)
        token_ids = tokenizer.encode(sentense, maxlen=maxlen)[0]
        token_ids = sequence_padding([token_ids])
        sent_vector = encoder.predict([token_ids, np.zeros_like(token_ids)],verbose=0)
        return sent_vector#l2_norm(sent_vector)

    def retrieval_from_db(self,id_list):
        data = self.database.loc[self.database['idx'].isin(id_list),'sentence']
        return data.tolist()

    def search_similary_sentense(self, sentense, num=5):
        sent_vector = self.encode_sentense(sentense)
        D, I = self.index.search(sent_vector, num)
        return self.retrieval_from_db(I[0])

        

if __name__ == '__main__':
    searcher = Searcher("./output/trained.index","./output/database.csv")
    test_sent = [s.strip() for s in open('test_data.txt').readlines()]
    for i,sent in enumerate(test_sent):
        match_sent = searcher.search_similary_sentense(sent)
        print(i,"query:",sent)
        print("retrieval sentenses:",match_sent,"\n")

