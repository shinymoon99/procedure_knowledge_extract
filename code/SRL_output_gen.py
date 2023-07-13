import os

print(os.environ.get('PYTHONPATH'))
import sys
sys.path.append('./')

# # Rest of your code goes here
from util.utils import get_positions, print_2dlist_to_file, read_2dintlist_from_file,get_different_num_positions,read_2dstrlist_from_file,get_token_labels
nums = read_2dintlist_from_file('./out/SRL/eval_result_pattern.txt')
tokens = read_2dstrlist_from_file('./out/SRL/eval_tokens.txt')
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


        

