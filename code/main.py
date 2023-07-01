import numpy as np
from gensim.models.word2vec import Word2Vec
from coarse2fine import C2F

from gensim.models import KeyedVectors
from gensim.test.utils import datapath

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForTokenClassification, AdamW
import sklearn.metrics as metrics
from transformers import BertForMaskedLM as WoBertForMaskedLM
from wobert import WoBertTokenizer


tokenizer = WoBertTokenizer.from_pretrained('junnyu/wobert_chinese_plus_base')
def pad_or_truncate_labels(labels, max_length):
    if len(labels) >= max_length:
        return labels[:max_length]
    else:
        return labels + [0] * (max_length - len(labels))

def tokenize_and_padding(text, labels, tokenizer):
    # 对文本进行分词
    tokenized_text = tokenizer.tokenize(text)
    # 对文本进行编码
    input_ids = tokenizer.encode(text,max_length=512,padding='max_length')

    # 将文本编码为张量
    input_ids = torch.tensor(input_ids)
    return input_ids


all_SC, all_SSR, all_SRL = [], [], []
label_SC, label_SSR, label_SRL = set(), set(), set()

for line in open("/root/autodl-tmp/MGTC/result1.txt",encoding="gbk").read().split("\n"):
    objs = line.split(", ")
    if len(objs)==2:
        all_SC.append(objs)
        label_SC.add(objs[-1])

for line in open("/root/autodl-tmp/MGTC/result2.txt",encoding="gbk").read().split("\n"):
	objs = line.split(", ")
	if line.endswith(", y") and len(objs)==3:
		objs = objs[:-1]
		all_SSR.append(objs)
		label_SSR.add(objs[-1])

for line in open("/root/autodl-tmp/MGTC/result3_copy.txt",encoding="gbk").read().split("\n"):
	objs = line.split(", ")
	if len(objs)==3:
		all_SRL.append(objs)
		label_SRL.add(objs[-1])

print(len(all_SC))
print(all_SC[0:10])
print(label_SC)

print(len(all_SSR))
print(all_SSR[0:10])
print(label_SSR)

print(len(all_SRL))
print(all_SRL[0:10])
print(label_SRL)



ratio = 0.80
train_SC,  test_SC  = all_SC[:int(len(all_SC)*ratio)],   all_SC[int(len(all_SC)*ratio):]
train_SSR, test_SSR = all_SSR[:int(len(all_SSR)*ratio)], all_SSR[int(len(all_SSR)*ratio):]
train_SRL, test_SRL = all_SRL[:int(len(all_SRL)*ratio)], all_SRL[int(len(all_SRL)*ratio):]
print(len(train_SC), len(test_SC))
print(len(train_SSR), len(test_SSR))
print(len(train_SRL), len(test_SRL))


#bert model
# from transformers import AutoTokenizer, AutoModelForMaskedLM
#
# w2v_embedding_size = 128
# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-lert-small")
#
# model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-lert-small")


#from wikipedia2vec import Wikipedia2Vec

w2v_embdding_size = 100

w2v = KeyedVectors.load_word2vec_format(datapath('/root/autodl-tmp/MGTC/zhwiki_20180420_100d.txt'), binary=False)  # C text format


# vocabulary = set(open("./w2v/text8_vocabulary.txt").read().split("\n"))
#w2v = Wikipedia2Vec.load('/root/autodl-tmp/MGTC/zhwiki_20180420_100d.pkl')

vocabulary = list(w2v.index_to_key)
label_SC  = list(label_SC)
label_SSR = list(label_SSR)
label_SRL = list(label_SRL)


def Encode_Sentence_Data(array, label_map):
	embeddings, labels = [], []
	for line in array:
		words = line[0].split(" ")
		label = line[1]

		mat = []
		for word in words:
			if(word in vocabulary):
				mat.append(w2v[word])
			else:
				mat.append(w2v["a"])
		while len(mat)<10:
			mat.append(w2v["a"])
		mat = mat[:10]

		embeddings.append(mat)
		labels.append(label_map.index(label))

		# print(line)


	return embeddings, labels

def Encode_Word_Data(array, label_map):
	embeddings, wembeddings, labels = [], [], []
	for line in array:
		words = line[0].split(" ")
		label = line[-1]

		mat = []
		for word in words:
			if(word in vocabulary):
				mat.append(w2v[word])
			else:
				mat.append(w2v["a"])
		while len(mat)<10:
			mat.append(w2v["a"])
		mat = mat[:10]

		embeddings.append(mat)

		index = int(line[1])
		center_word = line[0].split(" ")[index]
		if (center_word in vocabulary):
			rep = list(np.array(w2v[center_word]))
			rep.extend([index*1.0])
			rep = [float(obj) for obj in rep]
			wembeddings.append(rep)
		else:
			rep = list(np.array(w2v["a"]))
			rep.extend([index * 1.0])
			rep = [float(obj) for obj in rep]
			wembeddings.append(rep)

		labels.append(label_map.index(label))

		# print(line)

	return embeddings, wembeddings, labels

train_x1, train_y1 = Encode_Sentence_Data(train_SC, label_SC)
test_x1,  test_y1  = Encode_Sentence_Data(test_SC, label_SC)

train_x2, train_y2 = Encode_Sentence_Data(train_SSR, label_SSR)
test_x2,  test_y2  = Encode_Sentence_Data(test_SSR, label_SSR)

train_x3s, train_x3w, train_y3 = Encode_Word_Data(train_SRL, label_SRL)
test_x3s,  test_x3w,  test_y3  = Encode_Word_Data(test_SRL, label_SRL)


c2f = C2F(len(label_SC), len(label_SSR), len(label_SRL))
test_SC_labels,test_SSR_labels,test_SRL_labels,loss1,loss2,loss3 = c2f.train(train_x1, train_y1, test_x1,  test_y1, train_x2, train_y2, test_x2,  test_y2, train_x3s, train_x3w, train_y3, test_x3s,  test_x3w,  test_y3)
with open('/root/autodl-tmp/MGTC/out/st1_result.txt',"w") as fa:
	#test_SRL  list=[str]
	#test_SRL_labels  list=[str]
	label_SC_list =list(label_SC)
	for i,x in enumerate(test_SC):
		fa.write(test_SC[i][0]+", "+label_SC_list[test_SC_labels[i]]+'\n')
with open('/root/autodl-tmp/MGTC/out/st2_result.txt',"w") as fb:
	#test_SRL  list=[str]
	#test_SRL_labels  list=[str]
	label_SSR_list =list(label_SSR)
	for i,x in enumerate(test_SSR):
		fb.write(test_SRL[i][0]+", "+label_SSR_list[test_SSR_labels[i]]+'\n')
with open('/root/autodl-tmp/MGTC/out/st3_result.txt',"w") as fc:
	#test_SRL  list=[str]
	#test_SRL_labels  list=[str]
	label_SRL_list =list(label_SRL)
	for i,x in enumerate(test_SRL):
		fc.write(test_SRL[i][0]+", "+test_SRL[i][1]+", "+label_SRL_list[test_SRL_labels[i]]+'\n')

import matplotlib.pyplot as plt

with open('/root/autodl-tmp/MGTC/out/loss1.txt', 'w') as fx:
	for item in loss1:
		fx.write("%s\n" % item)
with open('/root/autodl-tmp/MGTC/out/loss2.txt', 'w') as fy:
	for item in loss1:
		fy.write("%s\n" % item)
with open('/root/autodl-tmp/MGTC/out/loss3.txt', 'w') as fz:
	for item in loss1:
		fz.write("%s\n" % item)
epoch1 = [i for i in range(0,len(loss1))]
epoch2 = [i for i in range(0,len(loss2))]
epoch3 = [i for i in range(0,len(loss3))]
plt.plot(epoch1, loss1)
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('/root/autodl-tmp/MGTC/out/loss_curve1.png')
plt.plot(epoch2, loss2)
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('/root/autodl-tmp/MGTC/out/loss_curve2.png')
plt.plot(epoch3, loss3)
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('/root/autodl-tmp/MGTC/out/loss_curve3.png')
