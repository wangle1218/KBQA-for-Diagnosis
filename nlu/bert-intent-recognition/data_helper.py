#! -*- coding: utf-8 -*-
import json
import pandas as pd 

def gen_training_data(raw_data_path):
    label_list = [line.strip() for line in open('label','r',encoding='utf8')]
    print(label_list)
    label2id = {label:idx for idx,label in enumerate(label_list)}

    data = []
    with open(raw_data_path,'r',encoding='utf8') as f:
        origin_data = f.read()
        origin_data = eval(origin_data)

    label_set = set()
    for item in origin_data:
        text = item["originalText"]

        label_class = item["label_4class"][0].strip("'")
        if label_class == "其他":
            data.append([text,label_class,label2id[label_class]])
            continue
        label_class = item["label_36class"][0].strip("'")
        label_set.add(label_class)
        if label_class not in label_list:
            # label_class = "其他"
            continue
        data.append([text,label_class,label2id[label_class]])

    print(label_set)

    data = pd.DataFrame(data,columns=['text','label_class','label'])

    print(data['label_class'].value_counts())

    data['text_len'] = data['text'].map(lambda x: len(x))
    print(data['text_len'].describe())
    import matplotlib.pyplot as plt
    plt.hist(data['text_len'], bins=30, rwidth=0.9, density=True,)
    plt.show()

    del data['text_len']

    data = data.sample(frac=1.0)
    train_num = int(0.9*len(data))
    train,test = data[:train_num],data[train_num:]
    train.to_csv("train.csv",index=False)
    test.to_csv("test.csv",index=False)


def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    df = pd.read_csv(filename,header=0)
    return df[['text','label']].values

if __name__ == '__main__':
    data_path = "E:/工作空间/CMID/CMID.json"
    gen_training_data(data_path)
