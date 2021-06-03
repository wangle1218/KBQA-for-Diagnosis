# -*- coding:utf-8 -*-
import os
import re
import json
import itchat
from itchat.content import *

from modules import gossip_robot,medical_robot,classifier
from utils.json_utils import dump_user_dialogue_context,load_user_dialogue_context

"""
问答流程：
1、用户输入文本
2、对文本进行解析得到语义结构信息
3、根据语义结构去查找知识，返回给用户

对文本进行解析的流程：
1、意图理解
    闲聊意图：问好、离开、肯定、拒绝
        问好、离开：需要有回复动作
        肯定、拒绝：需要执行动作
    诊断意图：
        当意图置信度达到一定阈值时(>=0.8)，可以查询该意图下的答案
        当意图置信度较低时(0.4~0.8)，按最高置信度的意图查找答案，询问用户是否问的这个问题
        当意图置信度更低时(<0.4)，拒绝回答
2、槽位填充
    如果输入是一个诊断意图，那么就需要语义槽的填充，得到结构化语义

"""



def delete_cache(file_name):
    """ 清除缓存数据，切换账号登入 """
    if os.path.exists(file_name):
        os.remove(file_name)


@itchat.msg_register(['Text'])
def text_replay(msg):
    user_intent = classifier(msg['Text'])
    print(user_intent)
    if user_intent in ["greet","goodbye","deny","isbot"]:
        reply = gossip_robot(user_intent)
    elif user_intent == "accept":
        reply = load_user_dialogue_context(msg.User['NickName'])
        reply = reply.get("choice_answer")
    else:
        reply = medical_robot(msg['Text'],msg.User['NickName'])
        if reply["slot_values"]:
            dump_user_dialogue_context(msg.User['NickName'],reply)
        reply = reply.get("replay_answer")

    msg.user.send(reply)


if __name__ == '__main__':
    # delete_cache(file_name='./logs/loginInfo.pkl')
    itchat.auto_login(hotReload=True, enableCmdQR=2, statusStorageDir='./logs/loginInfo.pkl')
    itchat.run()
