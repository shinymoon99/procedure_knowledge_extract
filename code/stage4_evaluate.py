from tuple_construct.tuple_constructor import tuple_construct
def find_common_first_level(list1, list2):
    return [element for element in list1 if element in list2]
def calculate_recall_precision(ground_truth,model_result):
    #
    #
    tp = 0
    mn = 0
    gn = 0

    for key,value in ground_truth.items():
        list1 = model_result[key]
        list2 = value
        #common = list(set([tuple(sublist) for sublist in list1]).intersection(set([tuple(sublist) for sublist in list2])))
        common = find_common_first_level(list1,list2)
        tp = tp + len(common)
        gn = gn + len(list2)
        mn = mn + len(list1)

    recall = tp/gn
    precision = tp/mn

    return recall,precision

"""get st4 result using st3 result"""
#model result
tc = tuple_construct("D:\pycode\MGTC\\result1.txt","D:\pycode\MGTC\out\st3_result.txt")
tc.get_BlockAndBlockType()
tc.get_BlockWord()
tc.get_BlockTag()
tc.get_BlockTuple()
tc.autocomplete_BlockTuple()
tc.autocombine_BlockTuple()
model_result = tc.block_tuple
"""get ground truth from file"""
#ground truth
import json

with open('D:\pycode\MGTC\data\S-TC.json') as f:
    ground_truth = json.load(f)




calculate_recall_precision(ground_truth,model_result)