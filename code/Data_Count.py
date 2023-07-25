import json

import jieba
import matplotlib.pyplot as plt
import csv
def save_dict_to_csv(d, filename):
    # Sort the dictionary by its values
    sorted_dict = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}

    # Write the sorted dictionary to a CSV file
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Key", "Value"])
        for key, value in sorted_dict.items():
            writer.writerow([key, value])
def plot_dict(d,filename):
    # Get the keys and values from the dictionary
    keys = list(d.keys())
    values = list(d.values())

    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Plot the bar chart
    ax.bar(keys, values)

    # Set the title and axis labels
    ax.set_title("Dictionary Plot")
    ax.set_xlabel("Keys")
    ax.set_ylabel("Values")

    # Show the plot
    plt.show()
    # Save the plot as a PNG file
    fig.savefig(filename)    
def get_word_num(sentence):
    # Tokenize the sentence using jieba
    words = jieba.lcut(sentence)
    # Count the number of words
    word_num = len(words)
    return word_num
#动作数量（不考虑动作词性） cnt
# 不同的动作总共出现了多少次 p_cnt
def sum_dicts(dicts):
    result = {}
    for d in dicts:
        for key, value in d.items():
            result[key] = result.get(key, 0) + value
    return result
cnt = 0
cnt1 = 0
cnt  = []
with open('D:\pycode\MGTC\data\data_correct_formated.json', encoding='utf-8') as f:
    datas = json.load(f)
cnt_all = 0
p_cnt = {}
for data in datas:
    cnt_all = cnt_all + len(data['labels'])
    cnt.append(len(data["labels"]))
    for pro in data["labels"]:
        p = pro["REL"]
        if p in p_cnt:
            p_cnt[p]  += 1       
        else:
            p_cnt[p] = 1

#句子平均词数
with open('D:\pycode\MGTC\data\data_correct_formated.json', encoding='utf-8') as f:
    datas = json.load(f)
word_num = []
for data in datas:
    t = get_word_num(data["sentence"])
    word_num.append(t)        
#命题中含有语义角色的种类
sr = []
t = {"A0":0,"A1":0,"A2":0,"ADV":0,"TMP":0,"CND":0,"PRP":0,"MNR":0}
for sentence in datas:
    t = {"A0":0,"A1":0,"A2":0,"ADV":0,"TMP":0,"CND":0,"PRP":0,"MNR":0}
    for pro in sentence["labels"]:
        for arg in pro["ARG"]:
            t[arg]+=1
        if "ARGM" in pro:
            for argm in pro["ARGM"]:
                t[argm]+=1
    sr.append(t)
total_sr =sum_dicts(sr)

# get top 20 
top_items = sorted(p_cnt.items(), key=lambda x: x[1], reverse=True)[:20]
top_p_cnt = {k:v for k,v in top_items}
plot_dict(top_p_cnt,"./out/datacount/pcnt")
save_dict_to_csv(p_cnt,"./out/datacount/pcnt")
plot_dict(total_sr,"./out/datacount/total_sr")
save_dict_to_csv(total_sr,"./out/datacount/total_sr")
#动作数量（不考虑动作类型） cnt
# 不同的动作总共出现了多少次 p_cnt
#不同语义角色 sr
#句子数 datas
print(cnt)
print(p_cnt)
print(len(datas))
print(sr)
print(total_sr)