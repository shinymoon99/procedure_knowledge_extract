import sys
sys.path.append('./')
from input_gen.pr_input_gen import split_sentence,pr_combine_tokens_by_pattern
# TODO: the converting process should be the same. we should do it on another
import re
import unicodedata
import json
def is_chinese(char):
    return "CJK" in unicodedata.name(char)

def convert_str_to_tokens(s):
    pattern = r"^([-_a-zA-Z()]+)"
    tokens = []
    i = 0
    while i < len(s):
        if not is_chinese(s[i]):
            match = re.match(pattern, s[i:])
            if match:
                token = match.group(1)
                tokens.append(token)
                i += len(token)
            else:
                tokens.append(s[i])
                i += 1
        else:
            tokens.append(s[i])
            i += 1
    return tokens
def getIndex(sentence,word,word_span):
    matches =re.finditer(word,sentence)
    i = -1
    index = -1
    for match in matches:
        i = i+1
        if match.start() == word_span[0]:
            index = i
    return index
def getWordPos(word,tokens,index):
    """
    arg1:
    """
    span = []
    word_tokens = [i for i in word]
    ind = -1
    for i in range(0,len(tokens)-len(word_tokens)):
        if tokens[i:i+len(word)]== word_tokens:
            ind = ind + 1
        if ind == index:
            span = [i,i+len(word)-1]
            return span
    return span
ratio = 0.8
data = []
# read data
pattern = "([-_a-zA-Z()]*\(?([-_a-zA-Z]*)\)?[-_a-zA-Z()]*)"
with open("data\data_correct_formated.json","r",encoding="utf-8") as f:
    data = json.load(f)
for sentence_info in data:
    for prop in sentence_info["labels"]:
        tokens = split_sentence(sentence_info["sentence"],pattern)
        #tokens = convert_str_to_tokens(sentence_info["sentence"])
        ind = getIndex(sentence_info["sentence"],prop["REL"],prop["span"])
        new_span = getWordPos(prop["REL"],tokens,ind)
        prop["span"] = new_span
        # sentence_info["tokens"] = tokens
with open("data\data_output_eval.json","w",encoding="utf-8") as f1:
    json.dump(data[int(ratio*len(data)):],f1,ensure_ascii=False)