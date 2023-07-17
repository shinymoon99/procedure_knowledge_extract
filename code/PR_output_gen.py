import re
import sys
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

#generate the input for SRL
with open('./out/PR/eval_tokens.txt', 'r') as file:
    sentence_tokens = []
    for line in file:
        line = line.strip()
        t = line.split()
        sentence_tokens.append(t)
assert len(sentence_tokens)==len(predicate_list)
#get SRL input
SRL_input = []
for i in range(len(sentence_tokens)):
    for predicate in predicate_list[i]:
        t = ['[SEP]']+[predicate]+['[SEP]']
        SRL_input.append(sentence_tokens[i]+t)
print(SRL_input)
with open('./out/PR/SRL_input.txt','w') as f:
    for i in SRL_input:
        SRL_input_text= " ".join(i)
        f.write(SRL_input_text+'\n')
