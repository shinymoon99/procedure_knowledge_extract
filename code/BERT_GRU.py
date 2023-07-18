import sys
sys.path.append('./')
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForTokenClassification
from models.model import  Bert_GRU,Bert_SRL
from input_gen.data_load import POS_data_load,PR_data_load,SRL_data_load,pos_label_set
from models.model_train import pos_train,pr_train,srl_train
from models.model_eval import pos_eval,pr_eval,srl_eval
from models.model_run import srl_run
import transformers
import json
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from util.utils import extract_arguments
from util.eval import calculate_f1_score,getPredictedSRL
import os
"""
load model
"""

#bert_model = BertModel.from_pretrained('bert-base-chinese',ignore_mismatched_sizes=True)


# Define the SRL model with a BiGRU layer
hidden_size = 768
#srl_label_set = ("O","REL","A0","A1","A2","A3","A4","ADV","CND","PRP","TMP","MNR")
srl_label_set = ("O","A0","A1","A2","A3","A4","ADV","CND","PRP","TMP","MNR")
#srl_label_set = ("O","A0","A1","A2")
num_labels = len(srl_label_set)  # Number of labels: B-PRED, I-PRED, B-CON
l2i = {label: i for i, label in enumerate(srl_label_set)}
bert_model = BertModel.from_pretrained('bert-base-chinese',ignore_mismatched_sizes=True)
SRL_model = Bert_SRL(bert_model, hidden_size, num_labels)

if os.path.isfile('./fine-tuned_model/SRL/BERT_SRL_weight.pth'):
    t1 = torch.load('./fine-tuned_model/SRL/BERT_SRL_weight.pth')
else:
    pass
#SRL_model.load_state_dict(torch.load('/root/autodl-tmp/MGTC/fine-tuned_model/srl_model_default/BERT_GRU_weight.pth'))
#SRL_model.load_state_dict(torch.load('./fine-tuned_model/BERT/SRL/BERT_SRL_weight.pth'))

"""
load data from standard file
"""
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
#get data
with open('./data/data_correct_formated.json', encoding='utf-8') as f:
    data = json.load(f)

#Define SRL data
srl_train_dataloader,srl_eval_dataloader = SRL_data_load(data,l2i,8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# SRL_model = nn.DataParallel(SRL_model,device_ids = [0,1])
# SRL_model.to(device)
"""
load data from pr output
"""

"""
train model
"""
# # Fine-tune the models on the respective tasks


# SRL_model.train()

# LEARNING_RATE = 2e-5

# srl_optimizer = torch.optim.Adam(SRL_model.parameters(), lr=LEARNING_RATE)

# srl_total_steps = len(srl_train_dataloader) * 10  # 10 epochs

# srl_scheduler = transformers.get_linear_schedule_with_warmup(srl_optimizer, num_warmup_steps=0, num_training_steps=srl_total_steps)
# NUM_EPOCHS = 200

# class_weight = torch.tensor([1.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0])
# for epoch in range(NUM_EPOCHS):
#     start_time = time.time()

#     # Train the SRL model
#     srl_train(SRL_model,srl_train_dataloader,device,srl_optimizer,srl_scheduler,class_weight)
#     end_time = time.time()
#     print(f"epoch:{epoch} time:{end_time-start_time}")
"""
eval model
"""
# Evaluate the models on the respective tasks

SRL_model.eval()


#SRL eval 
srl_eval(SRL_model,srl_eval_dataloader,device,srl_label_set)

gold_arguments_list = extract_arguments('./data/data_correct_formated.json')
arguments_list = getPredictedSRL('out\SRL\eval_result_pattern.txt','out\SRL\eval_tokens.txt')
p,r,f = calculate_f1_score(arguments_list,gold_arguments_list)
"""
save model
"""

#torch.save(SRL_model.state_dict(), '/root/autodl-tmp/MGTC/fine-tuned_model/srl_model_default/BERT_GRU_weight.pth')
torch.save(SRL_model.state_dict(), './fine-tuned_model/BERT/BERT_SRL_weight.pth')