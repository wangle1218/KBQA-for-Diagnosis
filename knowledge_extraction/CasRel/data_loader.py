#! -*- coding:utf-8 -*-
import numpy as np
import re, os, json
from random import choice

BERT_MAX_LEN = 256
RANDOM_SEED = 2019

def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

def to_tuple(sent):
    triple_list = []
    for triple in sent['triple_list']:
        triple_list.append(tuple(triple))
    sent['triple_list'] = triple_list

def seq_padding(batch, padding=0):
    length_batch = [len(seq) for seq in batch]
    max_length = max(length_batch)
    return np.array([
        np.concatenate([seq, [padding] * (max_length - len(seq))]) if len(seq) < max_length else seq for seq in batch
    ])

def load_data(train_path, dev_path, test_path, rel_dict_path):
    train_data = json.load(open(train_path,encoding='utf8'))
    dev_data = json.load(open(dev_path,encoding='utf8'))
    test_data = json.load(open(test_path,encoding='utf8'))
    id2rel, rel2id = json.load(open(rel_dict_path,encoding='utf8'))

    id2rel = {int(i): j for i, j in id2rel.items()}
    num_rels = len(id2rel)

    random_order = list(range(len(train_data)))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(random_order)
    train_data = [train_data[i] for i in random_order]

    for sent in train_data: 
        to_tuple(sent)
    for sent in dev_data:  
        to_tuple(sent)
    for sent in test_data: 
        to_tuple(sent)

    print("train_data len:", len(train_data))
    print("dev_data len:", len(dev_data))
    print("test_data len:", len(test_data))

    return train_data, dev_data, test_data, id2rel, rel2id, num_rels


class data_generator:
    def __init__(self, data, tokenizer, rel2id, num_rels, maxlen, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.rel2id = rel2id
        self.num_rels = num_rels
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(idxs)
            tokens_batch, segments_batch, sub_heads_batch, sub_tails_batch, sub_head_batch, sub_tail_batch, obj_heads_batch, obj_tails_batch = [], [], [], [], [], [], [], []
            for idx in idxs:
                line = self.data[idx]
                text = ' '.join(line['text'].split()[:self.maxlen])
                tokens = self.tokenizer.tokenize(text)
                if len(tokens) > BERT_MAX_LEN:
                    tokens = tokens[:BERT_MAX_LEN]
                text_len = len(tokens)

                s2ro_map = {}
                for triple in line['triple_list']:
                    triple = (self.tokenizer.tokenize(triple[0])[1:-1], triple[1], self.tokenizer.tokenize(triple[2])[1:-1])
                    sub_head_idx = find_head_idx(tokens, triple[0])
                    obj_head_idx = find_head_idx(tokens, triple[2])
                    if sub_head_idx != -1 and obj_head_idx != -1:
                        sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                        if sub not in s2ro_map:
                            s2ro_map[sub] = []
                        s2ro_map[sub].append((obj_head_idx,
                                           obj_head_idx + len(triple[2]) - 1,
                                           self.rel2id[triple[1]]))

                if s2ro_map:
                    token_ids, segment_ids = self.tokenizer.encode(text)
                    if len(token_ids) > text_len:
                        token_ids = token_ids[:text_len]
                        segment_ids = segment_ids[:text_len]
                    tokens_batch.append(token_ids)
                    segments_batch.append(segment_ids)
                    sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
                    for s in s2ro_map:
                        sub_heads[s[0]] = 1     
                        sub_tails[s[1]] = 1     
                    sub_head, sub_tail = choice(list(s2ro_map.keys()))
                    obj_heads, obj_tails = np.zeros((text_len, self.num_rels)), np.zeros((text_len, self.num_rels))
                    for ro in s2ro_map.get((sub_head, sub_tail), []): 
                        obj_heads[ro[0]][ro[2]] = 1
                        obj_tails[ro[1]][ro[2]] = 1
                    sub_heads_batch.append(sub_heads)
                    sub_tails_batch.append(sub_tails)
                    sub_head_batch.append([sub_head])
                    sub_tail_batch.append([sub_tail])
                    obj_heads_batch.append(obj_heads)
                    obj_tails_batch.append(obj_tails)
                    if len(tokens_batch) == self.batch_size or idx == idxs[-1]:
                        tokens_batch = seq_padding(tokens_batch)
                        segments_batch = seq_padding(segments_batch)
                        sub_heads_batch = seq_padding(sub_heads_batch)
                        sub_tails_batch = seq_padding(sub_tails_batch)
                        obj_heads_batch = seq_padding(obj_heads_batch, np.zeros(self.num_rels))
                        obj_tails_batch = seq_padding(obj_tails_batch, np.zeros(self.num_rels))
                        sub_head_batch, sub_tail_batch = np.array(sub_head_batch), np.array(sub_tail_batch)
                        yield [tokens_batch, segments_batch, sub_heads_batch, sub_tails_batch, sub_head_batch, sub_tail_batch, obj_heads_batch, obj_tails_batch], None
                        tokens_batch, segments_batch, sub_heads_batch, sub_tails_batch, sub_head_batch, sub_tail_batch, obj_heads_batch, obj_tails_batch, = [], [], [], [], [], [], [], []