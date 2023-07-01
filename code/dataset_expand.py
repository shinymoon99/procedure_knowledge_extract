import re

with open("D:\pycode\MGTC\\result3_copy.txt",'r',encoding="gbk") as f:
    data = f.readlines()

#
labels = []
cnt = {"role":0,"action":0,"object":0,"adv":0,"none":0}
for d in data:
    text,idx,label=re.split(", ",d[:-1])
    cnt[label] = cnt[label]+1

for i,x in cnt.items():
    print(i+":"+str(x))

tran_word = ["发送","转发","返回","回复"]
#通知=告知
import jieba.posseg as pseg
'''用于扩大数据集中的action
datalist
'''
datalist = []
for d in data:
    text,idx,label = re.split(", ",d[:-1])
    idx=int(idx)
    words = re.split(" ",text)
    if words[idx] in tran_word:
        # if words[idx]=='发送':
        #     datalist.append([" ".join(words[:idx]+["转发"]+[words[idx+1:]]),idx,label])
        #     datalist.append([" ".join(words[:idx] + ["返回"] + [words[idx + 1:]]), idx, label])
        #     datalist.append([" ".join(words[:idx] + ["返回"] + [words[idx + 1:]]), idx, label])
        for w in tran_word:
            if words[idx]!=w:
                datalist.append([" ".join(words[:idx] + [w] + words[idx + 1:]), str(idx), label])
data2 = []
for d in datalist:
    data2.append(", ".join(d)+"\n")
data = data2+data
# with open("D:\pycode\MGTC\\result3_copy.txt","w") as f1:
#      f1.writelines(data)
role_word = ["AMF","UE","UDM","NRF","AN","(R)AN"]