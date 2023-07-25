import sys
sys.path.append('./')
import json
from util.eval import getAccuracy
ratio = 0.8
with open('./out/SRL/SRL_output.json',encoding='utf-8') as file:
    # Write the dictionary to the file in JSON format
    out_data = json.load(file)

with open('./data/data_output_eval.json', encoding='utf-8') as f:
    eval_data = json.load(f)
#eval_data = data[int(ratio*(len(data))):]            
def findMatchedProp(prop,prop_list):
    
    span = prop["span"]
    p = prop["REL"]
    for ep in prop_list:
        espan = ep["span"]
        ep_predicate = ep["REL"]
        if span==espan and ep_predicate==p:
            return ep
    return []
def getSRL(proposition):
    sr = {}
    if proposition!=[]:
        if "ARG" in proposition:
            arg = proposition["ARG"]
            sr.update(arg)
        if "ARGM" in proposition:
            argm = proposition["ARGM"]
            sr.update(argm)
    else:
        sr = []
    return sr
    
outProp = []
evalProp = []
for i in range(len(eval_data)):
    t = i
    print(t)
    print("outdata: {}\nevaldata:{}".format(out_data[i]["sentence"],eval_data[i]["sentence"]))
    assert out_data[i]["sentence"].replace(" ","")==eval_data[i]["sentence"].replace(" ","")
    for prop in out_data[i]["labels"]:
        mp =findMatchedProp(prop,eval_data[i]["labels"])    
        outProp.append(getSRL(prop))
        
        evalProp.append(getSRL(mp))

accuracy = getAccuracy(outProp,evalProp)
print("Accuracy",accuracy)
