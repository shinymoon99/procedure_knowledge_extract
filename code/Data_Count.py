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
with open('D:\pycode\MGTC\data\data_correct_formated.json', encoding='utf-8') as f:
    datas = json.load(f)
    for data in datas:
        cnt = cnt + len(data['labels'])

#句子平均词数
with open('D:\pycode\MGTC\data\data_correct_formated.json', encoding='utf-8') as f:
    datas = json.load(f)
#句子含有动作的数量

#句子含有语义角色的种类

print(cnt)
print(len(datas))