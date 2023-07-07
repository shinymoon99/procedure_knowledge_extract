import sys
sys.path.append('./')
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForTokenClassification
from models.model import  Bert_GRU,WeightedBertForTokenClassification
from input_gen.data_load import POS_data_load,PR_data_load,SRL_data_load,pos_label_set
from models.model_train import pos_train,pr_train,srl_train
from models.model_eval import pos_eval,pr_eval,srl_eval
from util.utils import print_2dlist_to_file
import transformers
import json
import time
import torch.distributed as dist
import os
# Initialize the process group
# Set the world size
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12345'
# os.environ['WORLD_SIZE'] = '1'  # Set the desired world size value
# os.environ['RANK'] = '0'
# dist.init_process_group(backend='nccl')

"""
load model
"""
bert_model = BertModel.from_pretrained('bert-base-chinese',ignore_mismatched_sizes=True)



# Define the PR model
model_name = 'bert-base-chinese'
class_weight = torch.tensor([ 100.0,100.0,1.0]) # assign higher weight to predicate class
PR_model = WeightedBertForTokenClassification.from_pretrained(model_name, num_labels=3,ignore_mismatched_sizes=True)


#PR_model.load_state_dict(torch.load('/root/autodl-tmp/MGTC/fine-tuned_model/multi_model/PR_model_weight.pth'))

"""
load data
"""
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
#get data
with open('./data/data_correct_formated.json', encoding='utf-8') as f:
    data = json.load(f)
#Define PR data
pr_train_dataloader,pr_eval_dataloader,eval_tokens= PR_data_load(data,tokenizer,batch_size=4)
print_2dlist_to_file(eval_tokens, './out/eval_tokens.txt')


"""
train model
"""
# Fine-tune the models on the respective tasks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]  # IDs of the GPUs you want to use

print(device)
#model = nn.DataParallel(model, device_ids=device_ids).to(device)

PR_model=  PR_model.to(device)
PR_model.train()


LEARNING_RATE = 2e-5

pr_optimizer = torch.optim.Adam(PR_model.parameters(), lr=LEARNING_RATE)
pr_total_steps = len(pr_train_dataloader) * 10  # 10 epochs
pr_scheduler = transformers.get_linear_schedule_with_warmup(pr_optimizer, num_warmup_steps=0, num_training_steps=pr_total_steps)

NUM_EPOCHS = 1


for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    # Train the PR model
    pr_train(PR_model,pr_train_dataloader,device,pr_optimizer,pr_scheduler,class_weight)
    end_time = time.time()
    print(f"epoch:{epoch} time:{end_time-start_time}")
"""
eval model
"""
# Evaluate the models on the respective tasks
PR_model.eval()
#PR eval
pr_eval(PR_model,pr_eval_dataloader,device)

"""
save model
"""
torch.save(PR_model.state_dict(), './fine-tuned_model/PR/PR_model_weight.pth')
