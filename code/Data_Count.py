import json
cnt = 0
cnt1 = 0
import jieba

def get_word_num(sentence):
    # Tokenize the sentence using jieba
    words = jieba.lcut(sentence)
    # Count the number of words
    word_num = len(words)
    return word_num
#动作数量
cnt  = []
with open('D:\pycode\MGTC\data\data_correct_formated.json', encoding='utf-8') as f:
    datas = json.load(f)
cnt_all = 0
for data in datas:
    cnt_all = cnt_all + len(data['labels'])
    cnt.append(data["labels"])
#句子平均词数
with open('D:\pycode\MGTC\data\data_correct_formated.json', encoding='utf-8') as f:
    datas = json.load(f)
word_num = []
for data in datas:
    t = get_word_num(data["sentence"])
    word_num.append(t)        
#命题含有语义角色的种类
sr = []
t = {"A0":0,"A1":0,"A2":0,"ADV":0,"TMP":0,"CND":0,"PRP":0,"MNR":0}
for sentence in datas:
    for pro in sentence["labels"]:
        for arg in pro["ARG"]:
            t[arg]+=1
        for argm in pro["ARGM"]:
            t[argm]+=1
    sr.append(t)

print(cnt)
print(len(datas))