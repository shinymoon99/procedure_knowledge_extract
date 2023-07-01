#get
#output format
import re
import json
#output
from transformers import BertTokenizer
#input format

def is_chinese(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
#
def pos_str2seq(pos_input):
    tokens = []
    pos = []
    tag_str =re.split("[ ]+",pos_input)
    for x in tag_str:
        #print(x)
        #print(re.split("\|",x))
        print(tag_str)
        t = re.split("\|",x)
        token =t[0]
        tag = t[1]
        #中文
        if len(token)!=0 and is_chinese(token[0]):
            for i,t in enumerate(token):
                tokens.append(t)
                if i==0:
                    pos.append("B-"+tag)
                else:
                    pos.append("I-" + tag)
        elif token!="":
            tokens.append(token)
            pos.append("B-"+tag)
        # ' '空格不加入

    return tokens,pos
def pos_convert2bertformat(tokenizer,dim,token_seq,label_seq,l2i):
    temp_token_seq = ['CLS']+token_seq
    temp_label_seq = ['B-x']+label_seq

    #convert to id
    temp_token_seq = tokenizer.convert_tokens_to_ids(temp_token_seq)
    temp_label_seq = [l2i[x] for x in temp_label_seq]
    # add pad
    temp_token_seq = temp_token_seq + [0]*(dim-len(temp_token_seq))
    temp_label_seq = temp_label_seq + [-1]*(dim-len(temp_label_seq))
    return temp_token_seq,temp_label_seq

if  __name__=='__main__':
    with open("/data/data.json", encoding='utf-8') as f:
        data = json.load(f)
    tokens_list = []
    pos_list = []
    for sentence in data:
        tokens,pos = pos_str2seq(sentence["pos"])

        print(tokens)
        print(pos)
        assert  len(tokens)==len(pos)
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        label_set = ('B-n', 'I-n', 'B-np', 'I-np', 'B-ns', 'I-ns', 'B-ni', 'I-ni', 'B-nz', 'I-nz', 'B-m', 'I-m', 'B-q', 'I-q', 'B-mq', 'I-mq', 'B-t', 'I-t', 'B-f', 'I-f', 'B-s', 'I-s', 'B-v', 'I-v', 'B-a', 'I-a', 'B-d', 'I-d', 'B-h', 'I-h', 'B-k', 'I-k', 'B-i', 'I-i', 'B-j', 'I-j', 'B-r', 'I-r', 'B-c', 'I-c', 'B-p', 'I-p', 'B-u', 'I-u', 'B-y', 'I-y', 'B-e', 'I-e', 'B-o', 'I-o', 'B-g', 'I-g', 'B-w', 'I-w', 'B-x', 'I-x')
        l2i = {x: i for i, x in enumerate(label_set)}
        s, i = convert2bertformat(tokenizer, 512, tokens, pos, l2i)
        tokens_list.append(s)
        pos_list.append(i)