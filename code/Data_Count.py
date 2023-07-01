import json
cnt = 0
cnt1 = 0
with open('D:\pycode\MGTC\data\data_correct_formated.json', encoding='utf-8') as f:
    datas = json.load(f)
    for data in datas:
        cnt = cnt + len(data['labels'])
print(cnt)
print(len(datas))