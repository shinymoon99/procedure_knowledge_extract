import sys
sys.path.append('./')
import os,json
from util.utils import  read_2dintlist_from_file
# TODO：wordlist与str的等价互转的完成
def contains_english_alpha(s):
    for c in s:
        if c.isalpha() and c.isascii():
            return True
    return False
def combineTokens2Sen(tokens):
    result = ""
    for i in range(len(tokens)):
        if i > 0 and contains_english_alpha(tokens[i-1]) and contains_english_alpha(tokens[i]):
            result += " "  # Add a space between tokens with English letters
        result += tokens[i]
    return result
def getSentenceTokens(tokens_list):
    sentenceTokens = []
    for tokens in tokens_list:
        pos = tokens.index("[SEP]")
        sentenceTokens.append(tokens[1:pos])
    return sentenceTokens
def getPTokens(tokens_list):
    PTokens = []
    for tokens in tokens_list:
        pos = tokens.index("[SEP]")
        PTokens.append(tokens[pos+1:-1])
    return PTokens
def reformatInfo2Prop(p_tokens,semantic_roles,predicate_span):
    prop = {}
    predicate = "".join(p_tokens)
    prop["REL"] = predicate
    prop["span"] = predicate_span
    prop["ARG"] = {}
    prop["ARGM"] = {}
    for role in semantic_roles:
        if role in ["A0","A1","A2"]:
            prop["ARG"][role] = semantic_roles[role]
        elif role in ["ADV","TMP","CND","PRP","MNR"]:
            prop["ARGM"][role] = semantic_roles[role]
    return prop

# # Rest of your code goes here
from util.utils import get_positions, print_2dlist_to_file, read_2dintlist_from_file,get_different_num_positions,read_2dstrlist_from_file,get_token_labels
nums = read_2dintlist_from_file('./out/SRL/eval_result_pattern_PRoutput.txt')
tokens = read_2dstrlist_from_file('./out/SRL/eval_tokens_PRoutput.txt')
#    srl_label_set = ("O", "A0", "A1", "A2", "A3", "A4", "ADV", "CND", "PRP", "TMP", "MNR") 0-10

positions = []
for num in nums:
    t = get_different_num_positions(num)
    positions.append(t)
result = []
for i in range(len(positions)):
    t1 = get_token_labels(positions[i],tokens[i])
    result.append(t1)

print(positions)
"""
    convert to normal form
"""
result1 = result.copy()
#only save the first one and delete '|'
for pro in result1:
    for key,value in pro.items():
        if isinstance(value,list):
            pro[key] = value[0].replace('|','')
        else:
            pro[key] = value.replace('|','')
#edit the keys from index to actual label
srl_label_set = ("O","A0","A1","A2","A3","A4","ADV","CND","PRP","TMP","MNR")
#srl_label_set = ("O","A0","A1","A2")
num_labels = len(srl_label_set)  # Number of labels: B-PRED, I-PRED, B-CON
i2l = { i:label for i, label in enumerate(srl_label_set)}
prediction_SRL_list = []
for pro in result1:
    new_dict = {i2l.get(k,k):v for k,v in pro.items()}
    prediction_SRL_list.append(new_dict)
print(prediction_SRL_list)

#get result and form a list,
gold_srl = []
ratio = 0.8
# get 20% of sentences of data, which correspond to the input of PR
with open('./data/data_correct_formated.json', encoding='utf-8') as f:
    data = json.load(f)
eval_data = data[int(ratio*(len(data))):]            


sen_tokens = getSentenceTokens(tokens)
p_tokens =  getPTokens(tokens)
p_spans = read_2dintlist_from_file("./out/SRL/eval_PredicateSpan_PRoutput.txt")

#make span to ignore [cls], to get true span
for span in p_spans:
    span[0] = span[0]-1
    span[1] = span[1]-1
# store predicted result into the format of json and compare
"""
    Args:predicate_list ,tokens,p_spans
    Returns: data in json
"""
sentences_info = []
sentence_info = {}
sentence_info["sentence"] = combineTokens2Sen(sen_tokens[0])
sentence_info["labels"] = []
reformat_proposition = reformatInfo2Prop(p_tokens[0],prediction_SRL_list[0],p_spans[0]) 
sentence_info["labels"].append(reformat_proposition)
for i in range(1,len(tokens)):
    if sen_tokens[i] == sen_tokens[i-1]:
        reformat_proposition = reformatInfo2Prop(p_tokens[i],prediction_SRL_list[i],p_spans[i]) 
        sentence_info["labels"].append(reformat_proposition)
    else:
        sentences_info.append(sentence_info)        
        sentence_info = {}
        reformat_proposition = reformatInfo2Prop(p_tokens[i],prediction_SRL_list[i],p_spans[i])
        sentence_info["sentence"] = combineTokens2Sen(sen_tokens[i])
        sentence_info["labels"] = []
        sentence_info["labels"].append(reformat_proposition)
sentences_info.append(sentence_info)
with open('./out/SRL/SRL_output.json', 'w',encoding='utf-8') as file:
    # Write the dictionary to the file in JSON format
    json.dump(sentences_info, file,ensure_ascii=False)

