with open('D:\pycode\MGTC\\result.txt', 'r') as f:
    lines = f.readlines()
lines = [x.replace(",",'') for x in lines]
trigger_word = ["若","当","由于","不能","时","而","了","不到","之前","了","包含","有","未","是","没有","未能","不在","不再","存在","如果"]
state_word =["携带"]
lines1 = []
lines2 = []

tran_word = ["发送","转发","返回","回复","发起","回复","通知","告知","指示"]
delete_word = ["取消","释放","删除","丢弃","终止","移除","断开"]
import re
for line in lines:
    if set(trigger_word) & set(re.split(" ",line)):
        lines1.append(line[:-1] + (", scene\n"))
    elif set(state_word) & set(re.split(" ",line)):
        lines1.append(line[:-1] + (", state\n"))
    else:
        lines1.append(line[:-1] + (", action\n"))
        if set(tran_word) & set(re.split(" ",line)):
            lines2.append(line[:-1] + (", transmit, y\n"))
            lines2.append(line[:-1] + (", release, n\n"))
            lines2.append(line[:-1] + (", process, n\n"))
        elif set(delete_word) & set(re.split(" ",line)):
            lines2.append(line[:-1] + (", transmit, n\n"))
            lines2.append(line[:-1] + (", release, y\n"))
            lines2.append(line[:-1] + (", process, n\n"))
        else:
            lines2.append(line[:-1] + (", transmit, n\n"))
            lines2.append(line[:-1] + (", release, n\n"))
            lines2.append(line[:-1] + (", process, y\n"))

with open('D:\pycode\MGTC\\result1.txt','w') as f1:
    for line in lines1:
        f1.write(line)

with open('D:\pycode\MGTC\\result2.txt','w') as f2:
    for line in lines2:
        f2.write(line)


from ddparser import DDParser
ddp = DDParser(prob=True, use_pos=True)
with open("D:\pycode\MGTC\\result3.txt","w") as fa:
    for line in lines:
        tokenized_sen = re.split(" ",line)
        result =  ddp.parse_seg([tokenized_sen])
        #print(result)
        #save
        if result != None:
            anno_results = ['none']*len(tokenized_sen)
            head = result[0]['head'].index(0)+1
            # for i,x in enumerate(result[0]['head']):
            #     if  x == head or x == 0:
            #         if result[0]['deprel'][i] == 'SBV':
            #             anno_results[i] = 'role'
            #         elif result[0]['deprel'][i] == 'VOB':
            #             anno_results[i] = 'object'
            #         elif result[0]['deprel'][i] == 'HED':
            #             anno_results[i] = 'action'
            point = result[0]['head']
            for i,x in enumerate(result[0]['deprel']):
                if x == 'SBV':
                    anno_results[i] = 'role'
                    anno_results[point[i]-1] = 'action'
                elif x == 'VOB':
                    anno_results[point[i]-1] = 'action'
                elif result[0]['deprel'][i] == 'HED':
                    anno_results[i] = 'action'
                elif result[0]['deprel'][i] == 'ADV' and result[0]['word'][i] in ['根据','向','通过','给']:
                    anno_results[i] = 'adv'
            temp = []
            for i,x in enumerate(anno_results):
                if anno_results[i]  in ['adv','action']:
                    temp.append(i)
            for i in range(temp[0],len(anno_results)):
                if anno_results[i] not in ['adv','action']:
                    anno_results[i]='object'

            #规则1，对于所有adv和action后面的内容
            for i,x in enumerate(tokenized_sen):
                sentence = " ".join(tokenized_sen)
                sentence = sentence.replace('\n','')
                fa.write(sentence+", "+str(i)+", "+anno_results[i]+'\n')
