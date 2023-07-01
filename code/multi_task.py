import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForTokenClassification
from models.model import  Bert_GRU,WeightedBertForTokenClassification
from input_gen.data_load import POS_data_load,PR_data_load,SRL_data_load,pos_label_set
from models.model_train import pos_train,pr_train,srl_train
from models.model_eval import pos_eval,pr_eval,srl_eval
import transformers
import json
import time
import torch.distributed as dist
import os
# Initialize the process group
# Set the world size
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
os.environ['WORLD_SIZE'] = '1'  # Set the desired world size value
os.environ['RANK'] = '0'
dist.init_process_group(backend='nccl')

"""
load model
"""
bert_model = BertModel.from_pretrained('bert-base-chinese',ignore_mismatched_sizes=True)
class SharedBertModel(torch.nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        return bert_output
shared_bert_model = SharedBertModel(bert_model)
# Define the POS tagging model
model_name = 'bert-base-chinese'
POS_model = transformers.BertForTokenClassification.from_pretrained(model_name, num_labels=len(pos_label_set),ignore_mismatched_sizes=True)
POS_model.bert = shared_bert_model.bert
# Define the PR model
model_name = 'bert-base-chinese'
class_weight = torch.tensor([ 100.0,100.0,1.0]) # assign higher weight to predicate class
PR_model = WeightedBertForTokenClassification.from_pretrained(model_name, num_labels=3,ignore_mismatched_sizes=True)
PR_model.bert = shared_bert_model.bert


# Define the SRL model with a BiGRU layer
hidden_size = 768
srl_label_set = ("O","A0","A1","A2","A3","A4","ADV","CND","PRP","TMP","MNR")
num_labels = len(srl_label_set)  # Number of labels: B-PRED, I-PRED, B-CON
l2i = {label: i for i, label in enumerate(srl_label_set)}
bert_model = BertModel.from_pretrained('bert-base-chinese',ignore_mismatched_sizes=True)
SRL_model = Bert_GRU(shared_bert_model.bert, hidden_size, num_labels)


POS_model.load_state_dict(torch.load('/root/autodl-tmp/MGTC/fine-tuned_model/multi_model/POS_model_weight.pth'))
PR_model.load_state_dict(torch.load('/root/autodl-tmp/MGTC/fine-tuned_model/multi_model/PR_model_weight.pth'))
# PR_state_dict = torch.load('/root/autodl-tmp/MGTC/fine-tuned_model/multi_model/PR_model_weight.pth')
# # Remove unexpected keys from the state_dict
# filtered_state_dict = {k: v for k, v in PR_state_dict.items() if k in PR_model.state_dict()}
# # Load the filtered state_dict
# PR_model.load_state_dict(filtered_state_dict)
SRL_model.load_state_dict(torch.load('/root/autodl-tmp/MGTC/fine-tuned_model/multi_model/SRL_model_weight.pth'))
"""
load data
"""
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
#get data
with open('/root/autodl-tmp/MGTC/data/data_correct_formated.json', encoding='utf-8') as f:
    data = json.load(f)
#Define POS data
pos_train_dataloader,pos_eval_dataloader = POS_data_load(data,tokenizer,batch_size=4)
#Define NER data
pr_train_dataloader,pr_eval_dataloader = PR_data_load(data,tokenizer,batch_size=4)
#Define SRL data
srl_train_dataloader,srl_eval_dataloader = SRL_data_load(data,l2i,batch_size=4)


"""
train model
"""
# Fine-tune the models on the respective tasks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]  # IDs of the GPUs you want to use

print(device)
#model = nn.DataParallel(model, device_ids=device_ids).to(device)

POS_model= POS_model.to(device)
PR_model=  PR_model.to(device)
SRL_model= SRL_model.to(device)
POS_model.train()
PR_model.train()
SRL_model.train()

LEARNING_RATE = 2e-5
pos_optimizer = torch.optim.Adam(POS_model.parameters(), lr=LEARNING_RATE)
pr_optimizer = torch.optim.Adam(PR_model.parameters(), lr=LEARNING_RATE)
srl_optimizer = torch.optim.Adam(SRL_model.parameters(), lr=LEARNING_RATE)
pos_total_steps = len(pos_train_dataloader) * 10  # 10 epochs
pr_total_steps = len(pr_train_dataloader) * 10  # 10 epochs
srl_total_steps = len(srl_train_dataloader) * 10  # 10 epochs
pos_scheduler = transformers.get_linear_schedule_with_warmup(pos_optimizer, num_warmup_steps=0, num_training_steps=pos_total_steps)
pr_scheduler = transformers.get_linear_schedule_with_warmup(pr_optimizer, num_warmup_steps=0, num_training_steps=pr_total_steps)
srl_scheduler = transformers.get_linear_schedule_with_warmup(srl_optimizer, num_warmup_steps=0, num_training_steps=srl_total_steps)
NUM_EPOCHS = 1

srl_class_weight = torch.tensor([1.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0])
for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    # Train the POS model
    pos_train(POS_model,pos_train_dataloader,device,pos_optimizer,pos_scheduler)
    # Train the PR model
    pr_train(PR_model,pr_train_dataloader,device,pr_optimizer,pr_scheduler,class_weight)
    # Train the SRL model
    srl_train(SRL_model,srl_train_dataloader,device,srl_optimizer,srl_scheduler,srl_class_weight)
    end_time = time.time()
    print(f"epoch:{epoch} time:{end_time-start_time}")
"""
eval model
"""
# Evaluate the models on the respective tasks
POS_model.eval()
PR_model.eval()
SRL_model.eval()

#POS eval
pos_eval(POS_model,pos_eval_dataloader,device)
#PR eval
pr_eval(PR_model,pr_eval_dataloader,device)
#SRL eval
srl_eval(SRL_model,srl_eval_dataloader,device,srl_label_set)

"""
save model
"""
torch.save(POS_model.state_dict(), '/root/autodl-tmp/MGTC/fine-tuned_model/multi_model/POS_model_weight.pth')
torch.save(PR_model.state_dict(), '/root/autodl-tmp/MGTC/fine-tuned_model/multi_model/PR_model_weight.pth')
torch.save(SRL_model.state_dict(), '/root/autodl-tmp/MGTC/fine-tuned_model/multi_model/SRL_model_weight.pth')