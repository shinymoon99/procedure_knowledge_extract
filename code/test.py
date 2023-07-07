
import os

print(os.environ.get('PYTHONPATH'))
import sys
sys.path.append('D:\pycode\procedure_knowledge_extract')

# # Rest of your code goes here
from util.utils import get_positions, print_2dlist_to_file, read_2dintlist_from_file,get_different_num_positions,read_2dstrlist_from_file,get_token_labels
nums = read_2dintlist_from_file('D:\pycode\procedure_knowledge_extract\out\eval_result_pattern.txt')
tokens = read_2dstrlist_from_file('D:\pycode\procedure_knowledge_extract\out\eval_tokens.txt')
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

