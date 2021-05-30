#! -*- coding: utf-8 -*-
import pandas as pd
import json

def gen(out_path):
	train = pd.read_csv('../input/train.csv',encoding='utf8',header=0)
	test = pd.read_csv('../input/test.csv',encoding='utf8',header=0)
	dev = pd.read_csv('../input/dev.csv',encoding='utf8',header=0)

	data = pd.concat([train,test,dev],axis=0,ignore_index=True)
	print(data.head())

	data = data.loc[data['label']==1,:]
	sent1_set = test.loc[test['label']==1,'sentence1'].unique()
	with open(out_path,'w',encoding='utf8') as f:
		for sent in sent1_set:
			doc = {}
			doc[sent] = data.loc[data['sentence1']==sent,'sentence2'].tolist()
			if len(doc[sent]) >1:
				f.write(json.dumps(doc,ensure_ascii=False))
				f.write('\n')

gen('eval.json')