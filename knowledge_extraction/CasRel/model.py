#! -*- coding:utf-8 -*-
import keras
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.regularizers import l2
from bert4keras.models import build_transformer_model
from utils import seq_gather, extract_items, metric
from tqdm import tqdm
import numpy as np

bert_layers = 12

def E2EModel(bert_config_path, bert_checkpoint_path, LR, num_rels):
    bert_model = build_transformer_model(
            config_path=bert_config_path,
            checkpoint_path=bert_checkpoint_path,
            return_keras_model=True,
        )

    gold_sub_heads_in = keras.layers.Input(shape=(None,))
    gold_sub_tails_in = keras.layers.Input(shape=(None,))
    sub_head_in = keras.layers.Input(shape=(1,))
    sub_tail_in = keras.layers.Input(shape=(1,))
    gold_obj_heads_in = keras.layers.Input(shape=(None, num_rels))
    gold_obj_tails_in = keras.layers.Input(shape=(None, num_rels))

    gold_sub_heads, gold_sub_tails, sub_head, sub_tail, gold_obj_heads, gold_obj_tails = gold_sub_heads_in, gold_sub_tails_in, sub_head_in, sub_tail_in, gold_obj_heads_in, gold_obj_tails_in
    tokens = bert_model.input[0]
    mask = keras.layers.Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(tokens)

    output_layer = 'Transformer-2-FeedForward-Norm'
    tokens_feature = bert_model.get_layer(output_layer).output
    pred_sub_heads = keras.layers.Dense(1, activation='sigmoid')(tokens_feature)   
    pred_sub_tails = keras.layers.Dense(1, activation='sigmoid')(tokens_feature)

    subject_model = Model(bert_model.input, [pred_sub_heads, pred_sub_tails]) 


    sub_head_feature = keras.layers.Lambda(seq_gather)([tokens_feature, sub_head])
    sub_tail_feature = keras.layers.Lambda(seq_gather)([tokens_feature, sub_tail])
    sub_feature = keras.layers.Average()([sub_head_feature, sub_tail_feature])

    tokens_feature = keras.layers.Add()([tokens_feature, sub_feature])
    pred_obj_heads = keras.layers.Dense(num_rels, activation='sigmoid')(tokens_feature) 
    pred_obj_tails = keras.layers.Dense(num_rels, activation='sigmoid')(tokens_feature)

    object_model = Model(bert_model.input + [sub_head_in, sub_tail_in], [pred_obj_heads, pred_obj_tails]) 


    hbt_model = Model(bert_model.input + [gold_sub_heads_in, gold_sub_tails_in, sub_head_in, sub_tail_in, gold_obj_heads_in, gold_obj_tails_in],
                        [pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails])

    gold_sub_heads = K.expand_dims(gold_sub_heads, 2) 
    gold_sub_tails = K.expand_dims(gold_sub_tails, 2) 

    sub_heads_loss = K.binary_crossentropy(gold_sub_heads, pred_sub_heads)
    sub_heads_loss = K.sum(sub_heads_loss * mask) / K.sum(mask)
    sub_tails_loss = K.binary_crossentropy(gold_sub_tails, pred_sub_tails)
    sub_tails_loss = K.sum(sub_tails_loss * mask) / K.sum(mask)

    obj_heads_loss = K.sum(K.binary_crossentropy(gold_obj_heads, pred_obj_heads), 2, keepdims=True)
    obj_heads_loss = K.sum(obj_heads_loss * mask) / K.sum(mask)
    obj_tails_loss = K.sum(K.binary_crossentropy(gold_obj_tails, pred_obj_tails), 2, keepdims=True)
    obj_tails_loss = K.sum(obj_tails_loss * mask) / K.sum(mask)

    loss = (sub_heads_loss + sub_tails_loss) + (obj_heads_loss + obj_tails_loss)

    hbt_model.add_loss(loss)
    hbt_model.compile(optimizer=Adam(LR))
    hbt_model.summary()

    return subject_model, object_model, hbt_model

class Evaluate(Callback):
    def __init__(self, subject_model, object_model, tokenizer, id2rel, eval_data, save_weights_path, min_delta=1e-4, patience=7):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_op = np.greater
        self.subject_model = subject_model
        self.object_model = object_model
        self.tokenizer = tokenizer
        self.id2rel = id2rel
        self.eval_data = eval_data
        self.save_weights_path = save_weights_path

    def on_train_begin(self, logs=None):
        self.step = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.warmup_epochs = 2
        self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        precision, recall, f1 = metric(self.subject_model, self.object_model, self.eval_data, self.id2rel, self.tokenizer)
        if self.monitor_op(f1 - self.min_delta, self.best) or self.monitor_op(self.min_delta, f1):
            self.best = f1
            self.wait = 0
            self.model.save_weights(self.save_weights_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))