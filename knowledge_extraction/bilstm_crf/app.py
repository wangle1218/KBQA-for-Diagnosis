# -*- coding:utf-8 -*-
import json
import flask
import pickle
import ahocorasick
import numpy as np
from gevent import pywsgi
import tensorflow as tf 
import keras
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.sequence import pad_sequences

from crf_layer import CRF
from bilstm_crf_model import BiLstmCrfModel

class NerBaseDict(object):
    def __init__(self, dict_path):
        super(NerBaseDict, self).__init__()
        self.dict_path = dict_path
        self.region_words = self.load_dict(self.dict_path)
        self.region_tree = self.build_actree(self.region_words)

    def load_dict(self,path):
        with open(path,'r',encoding='utf8') as f:
            return json.load(f)

    def build_actree(self, wordlist):
        actree = ahocorasick.Automaton()
        for index, word in enumerate(wordlist):
            actree.add_word(word, (index, word))
        actree.make_automaton()
        return actree

    def recognize(self, text):
        item = {"string": text, "entities": []}

        region_wds = []
        for i in self.region_tree.iter(text):
            wd = i[1][1]
            region_wds.append(wd)
        stop_wds = []
        for wd1 in region_wds:
            for wd2 in region_wds:
                if wd1 in wd2 and wd1 != wd2:
                    stop_wds.append(wd1)
        final_wds = [i for i in region_wds if i not in stop_wds]
        item["entities"] = [{"word":i,"type":"disease","recog_label":"dict"} for i in final_wds]
        return item


class MedicalNerModel(object):
    """基于bilstm-crf的用于医疗领域的命名实体识别模型"""
    def __init__(self):
        super(MedicalNerModel, self).__init__()
        self.word2id,_,self.id2tag = pickle.load(
                open("./checkpoint/word_tag_id.pkl","rb")
            )
        self.model = BiLstmCrfModel(80,2410,200,128,24).build()
        self.model.load_weights('./checkpoint/best_bilstm_crf_model.h5')

        self.nbd = NerBaseDict('./checkpoint/diseases.json')

    def tag_parser(self,string,tags):
        item = {"string": string, "entities": [],"recog_label":"model"}
        entity_name = ""
        flag=[]
        visit=False
        for char, tag in zip(string, tags):
            if tag[0] == "B":
                if entity_name!="":
                    x=dict((a,flag.count(a)) for a in flag)
                    y=[k for k,v in x.items() if max(x.values())==v]
                    item["entities"].append({"word": entity_name,"type": y[0]})
                    flag.clear()
                    entity_name=""
                entity_name += char
                flag.append(tag[2:])
            elif tag[0]=="I":
                entity_name += char
                flag.append(tag[2:])
            else:
                if entity_name!="":
                    x=dict((a,flag.count(a)) for a in flag)
                    y=[k for k,v in x.items() if max(x.values())==v]
                    item["entities"].append({"word": entity_name,"type": y[0]})
                    flag.clear()
                flag.clear()
                entity_name=""
         
        if entity_name!="":
            x=dict((a,flag.count(a)) for a in flag)
            y=[k for k,v in x.items() if max(x.values())==v]
            item["entities"].append({"word": entity_name,"type": y[0]})

        return item

    def predict(self,texts):
        """
        texts 为一维列表，元素为字符串
        texts = ["淋球菌性尿道炎的症状","上消化道出血的常见病与鉴别"]
        """
        X = [[self.word2id.get(word,1) for word in list(x)] for x in texts ]
        X = pad_sequences(X,maxlen=80,value=0)
        pred_id = self.model.predict(X)
        res = []
        for text,pred in zip(texts,pred_id):
            tags = np.argmax(pred,axis=1)
            tags = [self.id2tag[i] for i in tags if i!=0]
            ents = self.tag_parser(text,tags)
            if ents["entities"]:
                res.append(ents)

        for text in texts:
            ents = self.nbd.recognize(text)
            if ents["entities"]:
                res.append(ents)

        return res


global graph,model,sess 

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
graph = tf.get_default_graph()
set_session(sess)

model = MedicalNerModel()


if __name__ == '__main__':
    app = flask.Flask(__name__)

    @app.route("/service/api/medical_ner",methods=["GET","POST"])
    def medical_ner():
        data = {"sucess":0}
        result = []
        text_list = flask.request.get_json()["text_list"]
        with graph.as_default():
            set_session(sess)
            result = model.predict(text_list)

        data["data"] = result
        data["sucess"] = 1

        return flask.jsonify(data)

    server = pywsgi.WSGIServer(("0.0.0.0",60061), app)
    server.serve_forever()


    # r = model.predict(["淋球菌性尿道炎的症状","上消化道出血的常见病与鉴别"])
    # print(r)

