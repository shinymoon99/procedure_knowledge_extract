import json
import re
import copy
with open("D:\pycode\MGTC\data\data.json",encoding="utf-8") as f:
    data = json.load(f)
data_copy = copy.deepcopy(data)
def is_same_list(list1, list2):
    if len(list1) != len(list2):
        return False
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            return False
    return True

for i, sentence_info in enumerate(data):
    # used to identify each proposition_info
    proposition_info = data[i]["labels"]
    text = data[i]["sentence"]


    for x, proposition_temp in enumerate(proposition_info):
        proposition = proposition_info[x]
        rel = proposition["REL"]
        span = proposition["span"]

        if text[span[0]:span[1] + 1] != rel:
            match = re.search(rel, text)
            if match != None:
                start, end = match.span()
                span[0] = start
                span[1] = end - 1
        print(proposition)
        for key,value in proposition["ARG"].items():
            print(key)
            print(value)
            if proposition["ARG"][key]==None or proposition["ARG"][key]["text"]==None:
                continue
            arg_str = proposition["ARG"][key]["text"]
            span1 = proposition["ARG"][key]["span"]

            if text[span1[0]:span1[1] + 1] != arg_str:
                #print("c1:" + arg_str)
                arg_pattern = r''+arg_str
                match = re.search(arg_pattern, text)
                if match != None:
                    #print("c2:"+arg_str)
                    start, end = match.span()
                    span1[0] = start
                    span1[1] = end - 1
        if "ARGM" in proposition and proposition["ARGM"]!=None:
            #print(proposition["ARGM"])
            for key,value in proposition["ARGM"].items():
                arg_str = proposition["ARGM"][key]["text"]
                span1 = proposition["ARGM"][key]["span"]
                if text[span1[0]:span1[1] + 1] != arg_str:
                    arg_pattern = r'' + arg_str
                    match = re.search(arg_pattern, text)
                    if match != None:
                        start, end = match.span()
                        span1[0] = start
                        span1[1] = end - 1

print(is_same_list(data,data_copy))
with open("D:\pycode\MGTC\data\data_correctspan.json", "w") as f:
    json.dump(data, f,ensure_ascii=False)
