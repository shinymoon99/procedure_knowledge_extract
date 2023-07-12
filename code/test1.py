import re
import sys
sys.path.append('./')
from util.utils import getPRPosFromPattern,read_2dintlist_from_file,read_2dstrlist_from_file,convert_negatives,get_token_labels,getPRTokenLabels
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
