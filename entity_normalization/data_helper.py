#! -*- coding:utf-8 -*-
import os
import jieba
import pickle
import numpy as np
import pandas as pd 
from bm25_retrival import BM25Retrieval

def gen_training_data(raw_data_dir):
    train_df = pd.read_excel(os.path.join(raw_data_dir,"train.xlsx"))
    answer_df = pd.read_excel(os.path.join(raw_data_dir,"answer.xlsx"))
    val_df = pd.read_excel(os.path.join(raw_data_dir,"val.xlsx"))

    bm25Model = BM25Retrieval(os.path.join(raw_data_dir,"code.txt"))

    total = 0
    error = 0
    def gen_negtive_sample(raw,norm):
        cand_entity = bm25Model.retrieval(raw,10)
        total += 1
        try:
            cand_entity.remove(norm)
        except:
            error += 1
        return cand_entity

    def match_positive_sample(list1,list2):
        ress = []
        for e1 in list1:
            score = 0
            pos_e = ""
            for e2 in list2:
                s = len(set(e1)&set(e2))
                if s > score:
                    score = s
                    pos_e = e2
            if pos_e != "":
                ress.append([e1,pos_e])

        return ress

    train = []
    data = np.concatenate([train_df.values , answer_df.values],axis=0)
    for raw_entity,norm_entity in data:
        #需要拆分具有多个实体的样本
        if '+' not in raw_entity and "##" not in norm_entity:
            train.append([raw_entity,norm_entity,1])
            for neg in gen_negtive_sample(raw_entity,norm_entity):
                train.append([raw_entity,neg,0])
        elif '+' not in raw_entity and "##" in norm_entity:
            for ne in norm_entity.split("##"):
                train.append([raw_entity,ne,1])
                for neg in gen_negtive_sample(raw_entity,ne):
                    train.append([raw_entity,neg,0])
        elif '+' in raw_entity and "##" in norm_entity:
            ne_list = norm_entity.split("##")
            re_list = raw_entity.split("+")
            for raw_ent,norm_ent in match_positive_sample(re_list,ne_list):
                train.append([raw_ent,norm_ent,1])
                for neg in gen_negtive_sample(raw_ent,norm_ent):
                    train.append([raw_ent,neg,0])
        else:
            re_list = raw_entity.split("+")
            for re in re_list:
                train.append([re,norm_entity,1])
                for neg in gen_negtive_sample(re,norm_entity):
                    train.append([re,neg,0])

    train = pd.DataFrame(train)
    train.columns = ["sentence1","sentence2","label"]
    train.drop_duplicates(subset=["sentence1","sentence2"],keep='first')

    test = []
    for raw_entity,norm_entity in val_df.values:
        #需要拆分具有多个实体的样本
        if '+' not in raw_entity and "##" not in norm_entity:
            test.append([raw_entity,norm_entity,1])

        elif '+' not in raw_entity and "##" in norm_entity:
            for ne in norm_entity.split("##"):
                test.append([raw_entity,ne,1])

        elif '+' in raw_entity and "##" in norm_entity:
            ne_list = norm_entity.split("##")
            re_list = raw_entity.split("##")
            for raw_ent,norm_ent in match_positive_sample(re_list,ne_list):
                test.append([raw_ent,norm_ent,1])
        else:
            re_list = raw_entity.split("##")
            for re in re_list:
                test.append([re,norm_entity,1])

    test = pd.DataFrame(test)
    test.columns = ["sentence1","sentence2","label"]

    train.to_csv("./data/train.csv",index=False,encoding="utf8")
    test.to_csv("./data/test.csv",index=False,encoding="utf8")
    print(train.shape)
    print("bm25 未召回正确规范实体的比例：",error / total)

    return train,test

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences

    把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
    最大的长度。填充和截取包括两种方法，post与pre，post指从尾部开始处理，pre指从头部
    开始处理，默认都是从尾部开始。

    Arguments:
        sequences: 序列
        maxlen: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值

    Returns:
        x: numpy array 填充后的序列维度为 (number_of_sequences, maxlen)

    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def shuffle(*arrs):
    """ shuffle

    Shuffle 数据

    Arguments:
        *arrs: 数组数据

    Returns:
        shuffle后的数据

    """
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)

def load_char_vocab():
    if os.path.exists("./checkpoint/word2id.pkl"):
        word2idx, idx2word = pickle.load(open("./checkpoint/word2id.pkl","rb"))
    else:
        df = pd.read_csv("./data/train.csv",encoding="utf8")
        vocab = []
        for ent in df["sentence1"].tolist()+df["sentence2"].tolist():
            vocab.extend(list(ent))

        with open(os.path.join("../../yidu-n7k/code.txt"),encoding='utf8') as f:
            for line in f.readlines():
                code,name = line.strip().split('\t')
                vocab.extend(list(name))

        vocab = sorted(set(vocab))
        print(len(vocab))
        word2idx = {word: index for index, word in enumerate(vocab,start=2)}
        word2idx["UNK"] = 1
        idx2word = {index: word for  word,index in word2idx.items()}
        pickle.dump((word2idx, idx2word),open("./checkpoint/word2id.pkl","wb"))

    return word2idx, idx2word


def char_index(p_sentences, h_sentences, maxlen=35):
    word2idx, idx2word = load_char_vocab()

    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word.lower()] for word in str(p_sentence) if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        h = [word2idx[word.lower()] for word in str(h_sentence) if len(word.strip()) > 0 and word.lower() in word2idx.keys()]

        p_list.append(p)
        h_list.append(h)

    p_list = pad_sequences(p_list, maxlen=maxlen)
    h_list = pad_sequences(h_list, maxlen=maxlen)

    return p_list, h_list


def load_char_data(path, data_size=None,maxlen=35):
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    p, h, label = shuffle(p, h, label)

    # [1,2,3,4,5] [4,1,5,2,0]
    p_c_index, h_c_index = char_index(p, h,maxlen=maxlen)

    return p_c_index, h_c_index, label


if __name__ == '__main__':
    gen_training_data("../../yidu-n7k")