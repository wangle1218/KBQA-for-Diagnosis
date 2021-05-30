#! -*- coding: utf-8 -*-
import json
import random
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
        label_36class = item["label_36class"][0].strip("'")
        if len(text) > 60 and label_36class not in ["所属科室","传染性","治愈率","治疗时间"]:
            continue
        label_class = item["label_4class"][0].strip("'")
        if label_class == "其他":
            data.append([text,label_class,label2id[label_class]])
            continue
        
        label_set.add(label_class)
        if label_36class in label_list:
            data.append([text,label_36class,label2id[label_36class]])
        label_set.add(label_36class)

    print(label_set)

    return data



def gen_sample_base_template():
    explain_qwds = ['能否介绍一下','想了解','可以问一下','能否告知','如何理解','怎样解释','可以介绍','解释一下','解释解释']
    howtodo_qwds = ['怎样才能','怎么做可以', '咋样','咋', '如何','如何才可以', '如何做', '怎样','采取什么措施', '什么手段可以','采取什么方法',\
                    '什么方法能','怎么做','我该如何','我该咋样','我该怎样','']
    relevance_qwds = ['并发症', '并发', '一起发生', '一并发生', '一起出现', '一并出现', '一同发生', '一同出现', '伴随发生', '伴随', '共现','其他病并发']
    greet_qwds = ['你好，麻烦问一下','打扰问一下','','','您好','请问得了','请教一下','冒昧问一下','问一下','','如果得了','']
    define_qwds = ['有什么定义？','定义的意思','的定义','是什么意思？','是啥意思','是啥？','的意思是什么','的介绍','的释义','解释','介绍']
    department_qwds = ['属于什么科', '要看什么科', '可以挂什么科', '要挂什么科室','看什么医生','应该看啥医生','可以看啥科室','挂啥科','看啥科呢']
    infect_qwds = ['会传染给其他人吗？','有传染性吗','能感染到其他人吗','是否会传染？','传染性强吗','会传给孩子吗','会传给老人吗','会传给孕妇吗',\
                    '会人传人不？','易感人群是什么人','容易感染不','易发人群是哪些人','我会感染吗', '我会染上吗', '会让我得上吗','会传染给哪些人','什么人容易得']
    cureprob_qwds = ['多大概率能治好？', '多大几率能治好', '治好希望大么？', '痊愈几率', '治愈概率几成', '治愈比例', '治好的可能性', '能治吗', '可治率多高？',\
                     '可以治好吗', '能否治好','治愈率多高','能治好不']
    check_qwds = ['需要做什么检查', '要检查啥项目？', '如何体检', '检查什么', '做啥体检呢？', '如何体检？','要做什么化验治疗吗','化验什么啊','做啥检查']
    prevent_qwds = ['预防', '防范', '抵制', '抵御', '防止','躲避','逃避','避开','免得','逃开','避开','避掉','躲开','躲掉','绕开']
    lasttime_qwds = ['的治疗周期', '要治多久', '治疗多长时间', '治疗多少时间', '治疗几天', '治愈需要几年', '治好要多少天', '要治几多时间', '得治疗几个小时',\
                 '治好得多少年','要花多少时间治好','完全治愈要多久','治好要多久','治疗时间长吗','治疗周期？']

    symptom_qwds = ['症状', '的症状', '的症状是什么', '表征是啥', '现象', '症候', '临床表现','病症表现', '病症是啥', '病症是啥']
    # cause_qwds = ['原因','成因', '为什么', '怎么会', '怎样才', '咋样才', '怎样会', '如何会', '为啥', '为何', '如何才会', '怎么才会', '会导致', '会造成']
    # food_qwds = ['饮食', '饮用', '吃', '食', '伙食', '膳食', '喝', '菜' ,'忌口', '补品', '保健品', '食谱', '菜谱', '食用', '食物','补品']
    # drug_qwds = ['药', '药品', '用药', '胶囊', '口服液', '炎片']
    # 
    # cureway_qwds = ['怎么治疗', '如何医治', '怎么医治', '怎么治', '怎么医', '如何治', '医治方式', '疗法', '咋治', '怎么办', '咋办', '咋治']
    # cure_qwds = ['治疗什么', '治啥', '治疗啥', '医治啥', '治愈啥', '主治啥', '主治什么', '有什么用', '有何用', '用处', '用途',
                      # '有什么好处', '有什么益处', '有何益处', '用来', '用来做啥', '用来作甚', '需要', '要']

    label_list = [line.strip() for line in open('label','r',encoding='utf8')]
    label2id = {label:idx for idx,label in enumerate(label_list)}

    disease_list = json.load(open('../../knowledge_extraction/bilstm_crf/checkpoint/diseases.json','r',encoding='utf8'))
    n = len(disease_list)-1

    data = []

    #问定义
    template = "{greet}{explain}{disease}{define}"
    for i in range(60):
        greet = greet_qwds[random.randint(0,len(greet_qwds)-1)]
        explain = explain_qwds[random.randint(0,len(explain_qwds)-1)]
        disease = disease_list[random.randint(0,n)]
        if i % 15 == 0:
            disease == ""
        define = define_qwds[random.randint(0,len(define_qwds)-1)]
        text = template.format(greet=greet,explain=explain,disease=disease,define=define)
        data.append([text,'定义',label2id['定义']])

    #问临床表现(病症表现)
    template = "{greet}{disease}{symptom}"
    for i in range(60):
        greet = greet_qwds[random.randint(0,len(greet_qwds)-1)]
        explain = explain_qwds[random.randint(0,len(explain_qwds)-1)]
        disease = disease_list[random.randint(0,n)]
        if i % 15 == 0:
            disease == ""
        symptom = symptom_qwds[random.randint(0,len(symptom_qwds)-1)]
        text = template.format(greet=greet,disease=disease,symptom=symptom)
        data.append([text,'临床表现(病症表现)',label2id['临床表现(病症表现)']])

    #问预防措施
    template = "{howtodo}{prevent}{disease}"
    for i in range(100):
        howtodo = howtodo_qwds[random.randint(0,len(howtodo_qwds)-1)]
        disease = disease_list[random.randint(0,n)]
        if i % 15 == 0:
            disease == ""
        prevent = prevent_qwds[random.randint(0,len(prevent_qwds)-1)]
        text = template.format(howtodo=howtodo,disease=disease,prevent=prevent)
        data.append([text,'预防',label2id['预防']])

    #问相关病症
    #请问XX疾病有什么相关病吗
    template = "{greet}{disease}有什么{relevance}"
    for i in range(80):
        greet = greet_qwds[random.randint(0,len(greet_qwds)-1)]
        disease = disease_list[random.randint(0,n)]
        if i % 15 == 0:
            disease == ""
        relevance = relevance_qwds[random.randint(0,len(relevance_qwds)-1)]
        text = template.format(greet=greet,disease=disease,relevance=relevance)
        data.append([text,'相关病症',label2id['相关病症']])

    #问所属科室
    #请问XX疾病应该看什么科
    template = "{greet}{disease}{department}"
    for i in range(100):
        greet = greet_qwds[random.randint(0,len(greet_qwds)-1)]
        disease = disease_list[random.randint(0,n)]
        if i % 15 == 0:
            disease == ""
        department = department_qwds[random.randint(0,len(department_qwds)-1)]
        text = template.format(greet=greet,disease=disease,department=department)
        data.append([text,'所属科室',label2id['所属科室']])

    #问传染性
    #请问XX疾病会传染哪些人群？
    template = "{greet}{disease}{infect}"
    for i in range(120):
        greet = greet_qwds[random.randint(0,len(greet_qwds)-1)]
        disease = disease_list[random.randint(0,n)]
        if i % 15 == 0:
            disease == ""
        infect = infect_qwds[random.randint(0,len(infect_qwds)-1)]
        text = template.format(greet=greet,disease=disease,infect=infect)
        data.append([text,'传染性',label2id['传染性']])

    #问治愈率
    #请问得了XX疾病的治愈率多高？
    template = "{greet}{disease}{cureprob}"
    for i in range(120):
        greet = greet_qwds[random.randint(0,len(greet_qwds)-1)]
        disease = disease_list[random.randint(0,n)]
        if i % 15 == 0:
            disease == ""
        cureprob = cureprob_qwds[random.randint(0,len(cureprob_qwds)-1)]
        text = template.format(greet=greet,disease=disease,cureprob=cureprob)
        data.append([text,'治愈率',label2id['治愈率']])

    #问化验/体检方案
    #请问得了XX疾病要做什么检查？
    template = "{greet}{disease}{check}"
    for i in range(150):
        greet = greet_qwds[random.randint(0,len(greet_qwds)-1)]
        disease = disease_list[random.randint(0,n)]
        if i % 15 == 0:
            disease == ""
        check = check_qwds[random.randint(0,len(check_qwds)-1)]
        text = template.format(greet=greet,disease=disease,check=check)
        data.append([text,'化验/体检方案',label2id['化验/体检方案']])

    #问治疗时间
    #请问得了XX疾病要治疗多久？
    template = "{greet}{disease}{lasttime}"
    for i in range(150):
        greet = greet_qwds[random.randint(0,len(greet_qwds)-1)]
        disease = disease_list[random.randint(0,n)]
        if i % 15 == 0:
            disease == ""
        lasttime = lasttime_qwds[random.randint(0,len(lasttime_qwds)-1)]
        text = template.format(greet=greet,disease=disease,lasttime=lasttime)
        data.append([text,'治疗时间',label2id['治疗时间']])

    return data
    

def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    df = pd.read_csv(filename,header=0)
    return df[['text','label']].values

if __name__ == '__main__':
    data_path = "E:/工作空间/CMID/CMID.json"
    data1= gen_training_data(data_path)
    data2 = gen_sample_base_template()

    data = data1 + data2

    data = pd.DataFrame(data,columns=['text','label_class','label'])
    data = data.sample(frac=1.0)
    print(data['label_class'].value_counts())

    data['text_len'] = data['text'].map(lambda x: len(x))
    print(data['text_len'].describe())
    import matplotlib.pyplot as plt
    plt.hist(data['text_len'], bins=30, rwidth=0.9, density=True,)
    plt.show()

    del data['text_len']

    train_num = int(0.9*len(data))
    train,test = data[:train_num],data[train_num:]
    train.to_csv("./data/train.csv",index=False)
    test.to_csv("./data/test.csv",index=False)


    

