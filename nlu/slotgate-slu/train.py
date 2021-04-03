#! -*- coding: utf-8 -*-
# https://github.com/MiuLab/SlotGated-SLU/blob/master/train.py
import os
import argparse
import keras
from keras.models import Model
from keras import backend as K
import tensorflow as tf 
import numpy as np

from utils import createVocabulary
from utils import loadVocabulary
from utils import DataProcessor

from model import SlotGatedSLU

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

parser = argparse.ArgumentParser(allow_abbrev=False)

#Training Environment
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--max_epochs", type=int, default=50, help="Max epochs to train.")
parser.add_argument("--maxlen", type=int, default=20, help="Max epochs to train.")
parser.add_argument("--max_features", type=int, default=750, help="Max epochs to train.")
parser.add_argument("--full_attention", type=bool, default=True, help="Max epochs to train.")

#Model and Vocab
parser.add_argument("--dataset", type=str, default='atis', help="""Type 'atis' or 'snips' to use dataset provided by us or enter what ever you named your own dataset.
                Note, if you don't want to use this part, enter --dataset=''. It can not be None""")
parser.add_argument("--model_path", type=str, default='./model_file', help="Path to save model.")
parser.add_argument("--vocab_path", type=str, default='./vocab', help="Path to vocabulary files.")

#Data
parser.add_argument("--train_data_path", type=str, default='train', help="Path to training data files.")
parser.add_argument("--test_data_path", type=str, default='test', help="Path to testing data files.")
parser.add_argument("--valid_data_path", type=str, default='valid', help="Path to validation data files.")
parser.add_argument("--input_file", type=str, default='seq.in', help="Input file name.")
parser.add_argument("--slot_file", type=str, default='seq.out', help="Slot file name.")
parser.add_argument("--intent_file", type=str, default='label', help="Intent file name.")

arg=parser.parse_args()

model_param = {
    'maxlen':arg.maxlen,
    'char_max_features':arg.max_features,
    'char_embed_size':200,
    'word_max_features':750,
    'word_embed_size':200,
    'char_embedding_matrix':None,
    'word_embedding_matrix':None,

    'lstm_units':128,
    'lstm_dropout_rate':0.1,
    'intent_dense_size':256,
    'intent_nums':23,
    'full_attention':arg.full_attention,
    'slot_dense_size':256,
    'slot_label_nums':122,
}

full_train_path = os.path.join('./data',arg.dataset,arg.train_data_path)
full_test_path = os.path.join('./data',arg.dataset,arg.test_data_path)
full_valid_path = os.path.join('./data',arg.dataset,arg.valid_data_path)

createVocabulary(os.path.join(full_train_path, arg.input_file), os.path.join(arg.vocab_path, 'in_vocab'))
createVocabulary(os.path.join(full_train_path, arg.slot_file), os.path.join(arg.vocab_path, 'slot_vocab'))
createVocabulary(os.path.join(full_train_path, arg.intent_file), os.path.join(arg.vocab_path, 'intent_vocab'))

in_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'in_vocab'))
slot_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'slot_vocab'))
intent_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'intent_vocab'))

train_processor = DataProcessor(
    os.path.join(full_train_path, arg.input_file), 
    os.path.join(full_train_path, arg.slot_file), 
    os.path.join(full_train_path, arg.intent_file), 
    in_vocab, slot_vocab, intent_vocab,
    arg.maxlen
    )

valid_processor = DataProcessor(
    os.path.join(full_valid_path, arg.input_file), 
    os.path.join(full_valid_path, arg.slot_file), 
    os.path.join(full_valid_path, arg.intent_file), 
    in_vocab, slot_vocab, intent_vocab,
    arg.maxlen
    )

test_processor = DataProcessor(
    os.path.join(full_test_path, arg.input_file), 
    os.path.join(full_test_path, arg.slot_file), 
    os.path.join(full_test_path, arg.intent_file), 
    in_vocab, slot_vocab, intent_vocab,
    arg.maxlen
    )


if __name__ == '__main__':
    train_X, train_slot_y, train_intent_y = train_processor.get_data()
    model_param['intent_nums'] = len(set(train_intent_y.flatten())) + 2
    model_param['intent_nums'] = len(set(train_slot_y.flatten())) + 2
    train_slot_y = keras.utils.to_categorical(train_slot_y,num_classes=model_param['slot_label_nums'])
    train_intent_y = keras.utils.to_categorical(train_intent_y,num_classes=model_param['intent_nums'])

    valid_X, valid_slot_y, valid_intent_y = valid_processor.get_data()
    valid_slot_y = keras.utils.to_categorical(valid_slot_y,num_classes=model_param['slot_label_nums'])
    valid_intent_y = keras.utils.to_categorical(valid_intent_y,num_classes=model_param['intent_nums'])

    model = SlotGatedSLU(model_param).build()
    model.compile(
        optimizer='adam',
        loss={'slot_out':'categorical_crossentropy', 'intent_out':'categorical_crossentropy'},
        loss_weights={'slot_out': 1.0, 'intent_out': 0.5},
        metrics={'intent_out':'accuracy'}
        )

    print(model.summary())

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_slot_out_loss', 
        factor=0.5, 
        patience=4, 
        verbose=1)

    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_slot_out_loss', 
        patience=8, 
        verbose=2, 
        mode='min'
        )
    bast_model_filepath = './model_file/slotgate_model.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(
        bast_model_filepath, 
        monitor='val_slot_out_loss', 
        verbose=1, 
        save_best_only=True,
        mode='min'
        )

    H = model.fit(
            x=train_X,
            y={"slot_out": train_slot_y, "intent_out": train_intent_y},
            validation_data=(
                valid_X,
                {"slot_out": valid_slot_y, "intent_out": valid_intent_y}
                ),
            batch_size=arg.batch_size,
            epochs=arg.max_epochs,
            callbacks=[reduce_lr,earlystop,checkpoint]
            )

    # model.load_weights(bast_model_filepath)
    test_X, test_slot_y, test_intent_y = test_processor.get_data()
    intent_pred,slot_pred = model.predict(test_X)

    # 意图准确率
    intent_pred = np.argmax(intent_pred,axis=1)
    intent_accuracy = (intent_pred==test_intent_y)
    intent_accuracy = np.mean(intent_accuracy)*100.0
    print("\n\n%s 数据集意图准确率：" % arg.dataset,intent_accuracy)

    # 槽位
    from metrics import *
    tag2id = slot_vocab['vocab']
    id2tag = {v:k for k,v in tag2id.items()}
    y_true, y_pred = [],[]

    for t_oh,p_oh in zip(test_slot_y,slot_pred):
        t_oh = [id2tag[i] for i in t_oh if i!=0]
        p_oh = np.argmax(p_oh,axis=1)
        p_oh = [id2tag[i] for i in p_oh if i!=0]

        y_true.append(t_oh)
        y_pred.append(p_oh)

    f1 = f1_score(y_true,y_pred,suffix=False)
    p = precision_score(y_true,y_pred,suffix=False)
    r = recall_score(y_true,y_pred,suffix=False)
    acc = accuracy_score(y_true,y_pred)
    print("\nf1_score: {:.4f}, precision_score: {:.4f}, recall_score: {:.4f}, accuracy_score: {:.4f}".format(f1,p,r,acc))
    print(classification_report(y_true, y_pred, digits=4, suffix=False))