import re
import sys
import json
sys.path.append('./')
from util.utils import getPRPosFromPattern,read_2dintlist_from_file,read_2dstrlist_from_file,convert_negatives,get_token_labels,getPRTokenLabels,extractPredicate,filterPredicate
pattern = read_2dintlist_from_file('./out/PR/eval_result_pattern.txt')
tokens = read_2dstrlist_from_file('./out/PR/eval_tokens.txt')
patterns = convert_negatives(pattern)
positions = []
for p in patterns:
    t = getPRPosFromPattern(p)
    positions.append(t)
result = []
for i in range(len(positions)):
    t1 = getPRTokenLabels(positions[i],tokens[i])
    result.append(t1)
print(positions)
#using template to filter
pset = extractPredicate('./data/data_correct_formated.json')
predicate_list = filterPredicate(result,pset)
print(predicate_list)

json_data = []
#generate the input for SRL
with open('./out/PR/eval_tokens.txt', 'r',encoding='utf-8') as file:
    sentence_tokens = []
    for line in file:
        line = line.strip()
        t = line.split()
        sentence_tokens.append(t)
assert len(sentence_tokens)==len(predicate_list)
#get SRL input
SRL_input = []
for i in range(len(sentence_tokens)):

    sentenceSRL = {} 
    words = sentence_tokens[i]
    sentenceSRL["words"]=words
    sentenceSRL["predicates"]=[]
    for predicate in predicate_list[i]:
        span = predicate[1]
        ptext = predicate[0].replace('|','')
        sentenceSRL["predicates"].append({"span":span,"ptext":ptext})
    SRL_input.append(sentenceSRL)
    
# Open a file for writing
with open('./out\PR\SRL_input.json', 'w',encoding='utf-8') as file:
    # Write the dictionary to the file in JSON format
    json.dump(SRL_input, file,ensure_ascii=False)

# SRL_input = []

# for i in range(len(sentence_tokens)):
#     for predicate in predicate_list[i]:
#         p = [t.replace('|','') for t in predicate[0]]
#         t = ['[SEP]']+p+['[SEP]']
#         SRL_input.append(sentence_tokens[i]+t)

