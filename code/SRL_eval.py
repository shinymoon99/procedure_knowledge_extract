import json
from util.eval import getAccuracy
ratio = 0.8
with open('./out\SRL\SRL_output.json',encoding='utf-8') as file:
    # Write the dictionary to the file in JSON format
    out_data = json.load(file)

with open('./data/data_correct_formated.json', encoding='utf-8') as f:
    data = json.load(f)
eval_data = data[int(ratio*(len(data))):]            
def findMatchedProp(prop,prop_list):
    span = prop["span"]
    p = prop["REL"]
    for ep in prop_list:
        espan = ep["span"]
        ep = ep["REL"]
        if span==espan & ep==p:
            return ep
    return []
def getSRL(proposition):
    sr = {}
    arg = proposition["ARG"]
    argm = proposition["ARGM"]
    sr.update(arg)
    sr.update(argm)
    return sr
    
outProp = []
evalProp = []
for i in range(len(eval_data)):
    assert out_data[i]["sentence"]==eval_data[i]["sentence"]
    for prop in out_data[i]:
        mp =findMatchedProp(prop,eval_data[i])    
        outProp.append(getSRL(prop))
        evalProp.append(getSRL(mp))
        getAccuracy(outProp,evalProp)

