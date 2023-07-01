import thulac
import unicodedata
from sklearn.metrics import precision_recall_fscore_support
import json
import jieba.posseg as pseg
import numpy as np
label_set = (
            'B-n', 'I-n', 'B-np', 'I-np', 'B-ns', 'I-ns', 'B-ni', 'I-ni', 'B-nz', 'I-nz', 'B-m', 'I-m', 'B-q', 'I-q',
            'B-mq',
            'I-mq', 'B-t', 'I-t', 'B-f', 'I-f', 'B-s', 'I-s', 'B-v', 'I-v', 'B-a', 'I-a', 'B-d', 'I-d', 'B-h', 'I-h',
            'B-k',
            'I-k', 'B-i', 'I-i', 'B-j', 'I-j', 'B-r', 'I-r', 'B-c', 'I-c', 'B-p', 'I-p', 'B-u', 'I-u', 'B-y', 'I-y',
            'B-e',
            'I-e', 'B-o', 'I-o', 'B-g', 'I-g', 'B-w', 'I-w', 'B-x', 'I-x')
pos_string ="n/名词 np/人名 ns/地名 ni/机构名 nz/其它专名 m/数词 q/量词 mq/数量词 t/时间词 f/方位词 s/处所词 v/动词 a/形容词 d/副词 h/前接成分 k/后接成分 i/习语 j/简称 r/代词 c/连词 p/介词 u/助词 y/语气助词 e/叹词 o/拟声词 g/语素 w/标点 x/其它"
pos_set = set()

# Split the string into individual segments
segments = pos_string.split()

# Iterate over the segments and extract the POS label
for segment in segments:
    pos_label = segment.split("/")[0]
    pos_set.add(pos_label)
def is_chinese(character):
    return 'CJK' in unicodedata.name(character) and character!='_'

def convert_to_bio_format(sentence):
    words_tags = sentence.split()

    bio_tags = []
    for word_tag in words_tags:
        word, tag = word_tag.split('|')
        word_length = len(word)
        if tag not in pos_set:
            tag = 'x'
        # try:
        #     if tag not in pos_set:
        #         raise Exception(f"Unexpected tag: {tag} of {word}")
        # except Exception as e:
        #     print(str(e))
        if word_length == 0:
            continue
        if word_length == 1 and is_chinese(word):
            bio_tags.append((word[0],"B-"+tag))
        else:
            if is_chinese(word[0]):
                bio_tags.append((word[0],"B-"+tag))
            for i in range(1, word_length):
                if is_chinese(word[i]):
                    bio_tags.append((word[i],"I-"+tag))

    return bio_tags




def get_score(ground_truth_str,prediction_str,sentence):
    #convert to bio format
    #return ground_truth_bio:a list of list[tuple(word,tag)]
    ground_truth_bio = []
    prediction_bio = []
    for i,true_labels in enumerate(ground_truth_str):
        temp1 = convert_to_bio_format(ground_truth_str[i])
        temp2 = convert_to_bio_format(prediction_str[i])
        # print(sentence[i])
        # print(temp1)
        # print(temp2)
        assert len(temp1) == len(temp2)
        ground_truth_bio.append(temp1)
        prediction_bio.append(temp2)

    '''
    convert to label index
    '''

    l2i = {x: i for i, x in enumerate(label_set)}
    for i,true_labels in enumerate(ground_truth_bio):
        for j,true_label in enumerate(true_labels):
            #print(true_label[1])
            ground_truth_bio[i][j] = (true_label[0],l2i[true_label[1]])
    for i,predict_labels in enumerate(prediction_bio):
        for j,predict_label in enumerate(predict_labels):
            #print(f"{predict_label[0]} {predict_label[1]}")
            prediction_bio[i][j] = (predict_label[0],l2i[predict_label[1]])

    #get all tags when confirm all labels is correct
    true_labels_flat = []
    predictions_flat = []
    for i,true_labels in enumerate(ground_truth_bio):
        for j,true_label in enumerate(true_labels):
            if ground_truth_bio[i][j][0]==prediction_bio[i][j][0]:
                true_labels_flat.append(ground_truth_bio[i][j][1])
                predictions_flat.append(prediction_bio[i][j][1])

    num_labels = len(label_set)  # Number of labels: B-PRED, I-PRED, B-CON
    l2i = {label: i for i, label in enumerate(label_set)}
    label = list(label_set)
    # calculate precision, recall, and f1 score for each label
    p, r, f1, support = precision_recall_fscore_support(true_labels_flat, predictions_flat, average=None)
    class_tags = sorted(set(true_labels_flat) | set(predictions_flat))
    class_tags = [x for x in class_tags if x != -1]
    # print the results
    # for i in range(len(p)):
    #     print(f"Label {label[class_tags[i]]}: precision={p[i]:.4f}, recall={r[i]:.4f}, f1={f1[i]:.4f},support={support[i]}")
    labels = [label[class_tags[x]] for x in range(len(p))]
    return labels,p,r,f1,support

'''
convert ground_truth and predict to bio format
'''

sentence = []
ground_truth = []
prediction = []

#get sentence and ground truth
with open("D:\pycode\MGTC\data\data_correct_formated.json",encoding="utf-8") as f:
    a = json.load(f)
for sentence_info in a:
    sentence.append(sentence_info["sentence"])
    ground_truth.append(sentence_info["pos"])
'''
get prediction
'''
#thulac
thu1 = thulac.thulac(deli='|')
for i,x in enumerate(sentence):
    predict = thu1.cut(sentence[i],text=True)
    prediction.append(predict)
labels,p,r,f,s= get_score(ground_truth,prediction,sentence)
overall_precision = np.mean(p)
overall_recall = np.mean(r)
overall_fscore = np.mean(f[i] for i in range(len(f)) if f[i]>0.2)

print("Overall Precision:", overall_precision)
print("Overall Recall:", overall_recall)
print("Overall F1-Score:", overall_fscore)
# #jieba

for i,x in enumerate(sentence):
    word_flag = []
    words = pseg.cut(sentence[i],use_paddle=False)
    for word, flag in words:
        #print('%s %s' % (word, flag))
        word_flag.append(fr"{word}|{flag}")
    result = " ".join(word_flag)
    prediction.append(result)
labels,p,r,f,s= get_score(ground_truth,prediction,sentence)
overall_precision = np.mean(p)
overall_recall = np.mean(r)
overall_fscore = np.mean(f[i] for i in range(len(f)) if f[i]>0.2)

print("Overall Precision:", overall_precision)
print("Overall Recall:", overall_recall)
print("Overall F1-Score:", overall_fscore)
#paddle
pseg.enable_paddle()

for i,x in enumerate(sentence):
    word_flag = []
    words = pseg.cut(sentence[i],use_paddle=True)
    for word, flag in words:
        #print('%s %s' % (word, flag))
        word_flag.append(fr"{word}|{flag}")
    result = " ".join(word_flag)
    prediction.append(result)
labels,p,r,f,s= get_score(ground_truth,prediction,sentence)
overall_precision = np.mean(p)
overall_recall = np.mean(r)
overall_fscore = np.mean(f[i] for i in range(len(f)) if f[i]>0.2)

print("Overall Precision:", overall_precision)
print("Overall Recall:", overall_recall)
print("Overall F1-Score:", overall_fscore)
import csv

# def write_scores_to_csv(labels, p3, r3, f3, s3, filename):
#     data = zip(labels, p3, r3, f3, s3)
#     with open(filename, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Label', 'P3', 'R3', 'F3', 'S3', ])  # Write the header row
#         writer.writerows(data)  # Write the scores data



# write_scores_to_csv(labels, p2, r2, f2, s2, 'D:\pycode\MGTC\\fine-tuned_model\pos_model\scores_jieba.csv')
