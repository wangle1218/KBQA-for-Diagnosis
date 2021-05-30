# -*- coding:utf-8 -*-
import json
import flask
import pickle
import numpy as np
from gevent import pywsgi
import tensorflow as tf 
import keras
from keras.backend.tensorflow_backend import set_session
from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding

from bert_model import build_bert_model

global graph,model,sess 


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
graph = tf.get_default_graph()
set_session(sess)

class BertIntentModel(object):
    def __init__(self):
        super(BertIntentModel, self).__init__()
        self.dict_path = 'E:/bert_weight_files/roberta/vocab.txt'
        self.config_path='E:/bert_weight_files/roberta/bert_config_rbt3.json'
        self.checkpoint_path='E:/bert_weight_files/roberta/bert_model.ckpt'

        self.label_list = [line.strip() for line in open('label','r',encoding='utf8')]
        self.id2label = {idx:label for idx,label in enumerate(self.label_list)}

        self.tokenizer = Tokenizer(self.dict_path)
        self.model = build_bert_model(self.config_path,self.checkpoint_path,13)
        self.model.load_weights('./checkpoint/best_model.weights')

    def predict(self,text):
        token_ids, segment_ids = self.tokenizer.encode(text, maxlen=60)
        proba = self.model.predict([[token_ids], [segment_ids]])
        rst = {l:p for l,p in zip(self.label_list,proba[0])}
        rst = sorted(rst.items(), key = lambda kv:kv[1],reverse=True)
        name,confidence = rst[0]
        return {"name":name,"confidence":float(confidence)}


BIM = BertIntentModel()


if __name__ == '__main__':
    app = flask.Flask(__name__)

    @app.route("/service/api/bert_intent_recognize",methods=["GET","POST"])
    def bert_intent_recognize():
        data = {"sucess":0}
        result = None
        param = flask.request.get_json()
        print(param)
        text = param["text"]
        with graph.as_default():
            set_session(sess)
            result = BIM.predict(text)

        data["data"] = result
        data["sucess"] = 1

        return flask.jsonify(data)

    server = pywsgi.WSGIServer(("0.0.0.0",60062), app)
    server.serve_forever()


    # r = BIM.predict("淋球菌性尿道炎的症状")
    # print(r)