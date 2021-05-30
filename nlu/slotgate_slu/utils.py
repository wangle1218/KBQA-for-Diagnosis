#! -*- coding: utf-8 -*-

import numpy as np

def createVocabulary(input_path, output_path, no_pad=False):
    if not isinstance(input_path, str):
        raise TypeError('input_path should be string')

    if not isinstance(output_path, str):
        raise TypeError('output_path should be string')

    vocab = {}
    with open(input_path, 'r', encoding='utf8') as fd, \
         open(output_path, 'w+', encoding='utf8') as out:
        for line in fd:
            line = line.rstrip('\r\n')
            words = line.split()

            for w in words:
                if w == '_UNK':
                    break
                if str.isdigit(w) == True:
                    w = '0'
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
        if no_pad == False:
            vocab = ['_PAD', '_UNK'] + sorted(vocab, key=vocab.get, reverse=True)
        else:
            vocab = ['_UNK'] + sorted(vocab, key=vocab.get, reverse=True)

        for v in vocab:
            out.write(v+'\n')

def loadVocabulary(path):
    if not isinstance(path, str):
        raise TypeError('path should be a string')

    vocab = []
    rev = []
    with open(path, encoding='utf8') as fd:
        for line in fd:
            line = line.rstrip('\r\n')
            rev.append(line)
        vocab = dict([(x,y) for (y,x) in enumerate(rev)])

    return {'vocab': vocab, 'rev': rev}

def sentenceToIds(data, vocab):
    if not isinstance(vocab, dict):
        raise TypeError('vocab should be a dict that contains vocab and rev')
    vocab = vocab['vocab']
    if isinstance(data, str):
        words = data.split()
    elif isinstance(data, list):
        words = data
    else:
        raise TypeError('data should be a string or a list contains words')

    ids = []
    for w in words:
        if str.isdigit(w) == True:
            w = '0'
        ids.append(vocab.get(w, vocab['_UNK']))

    return ids

def padSentence(s, max_length, vocab):
    if len(s) > max_length:
        return s[:max_length]
    else:
        while max_length - len(s) > 0:
            s.append(vocab['vocab']['_PAD'])
        return s

class DataProcessor(object):
    def __init__(self, in_path, slot_path, intent_path, in_vocab, slot_vocab, intent_vocab,max_len):
        self.__fd_in = open(in_path, 'r', encoding='utf8')
        self.__fd_slot = open(slot_path, 'r', encoding='utf8')
        self.__fd_intent = open(intent_path, 'r', encoding='utf8')
        self.__in_vocab = in_vocab
        self.__slot_vocab = slot_vocab
        self.__intent_vocab = intent_vocab
        self.max_len = max_len
        self.end = 0

    def close(self):
        self.__fd_in.close()
        self.__fd_slot.close()
        self.__fd_intent.close()

    def get_data(self):
        in_data = [] #输入序列 ，padding
        slot_data = [] # 输入序列对于的solt标签 ，padding
        slot_weight = []
        intents = [] #意图标签

        batch_in = [] #输入序列
        batch_slot = [] # 输入序列对于的solt标签
        max_len = 0

        #used to record word(not id)
        in_seq = []
        slot_seq = []
        intent_seq = []
        for i in range(100000):
            inp = self.__fd_in.readline()
            if inp == '':
                self.end = 1
                break
            slot = self.__fd_slot.readline()
            intent = self.__fd_intent.readline()
            inp = inp.rstrip()
            slot = slot.rstrip()
            intent = intent.rstrip()

            in_seq.append(inp)
            slot_seq.append(slot)
            intent_seq.append(intent)

            iii=inp
            sss=slot
            inp = sentenceToIds(inp, self.__in_vocab)
            slot = sentenceToIds(slot, self.__slot_vocab)
            intent = sentenceToIds(intent, self.__intent_vocab)
            batch_in.append(np.array(inp))
            batch_slot.append(np.array(slot))
            intents.append(intent[0])
            if len(inp) != len(slot):
                print(iii,sss)
                print(inp,slot)
                exit(0)
            if len(inp) > max_len:
                max_len = len(inp)

        intents = np.asarray(intents)
        for i, s in zip(batch_in, batch_slot):
            in_data.append(padSentence(list(i), self.max_len, self.__in_vocab))
            slot_data.append(padSentence(list(s), self.max_len, self.__slot_vocab))
            #print(s)
        in_data = np.asarray(in_data)
        slot_data = np.asarray(slot_data)

        self.close()
        return in_data, slot_data, intents
