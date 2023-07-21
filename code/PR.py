import sys
sys.path.append('./')
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForTokenClassification
from models.model import  Bert_GRU,WeightedBertForTokenClassification
from input_gen.data_load import POS_data_load,PR_data_load,SRL_data_load,pos_label_set,PR_eval_labels_load
from models.model_train import pos_train,pr_train,srl_train
from models.model_eval import pos_eval,pr_eval,srl_eval
from util.utils import print_2dlist_to_file,append_loss_values_to_csv,draw_and_save_loss_curve,read_list_from_csv
from util.eval import recall,precision,calculate_f1_score,getPredictedSRL,getAccuracy
from sklearn.metrics import precision_recall_fscore_support
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

if os.path.isfile('./fine-tuned_model/PR/BERT/PR_model_weight.pth'):
    PR_model.load_state_dict(torch.load('./fine-tuned_model/PR/BERT/PR_model_weight.pth'))
else:
    pass
"""
load data
"""
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
#get data
with open('./data/data_correct_formated.json', encoding='utf-8') as f:
    data = json.load(f)
#Define PR data
ratio = 0.8
pattern = "([-_a-zA-Z()]*\(?([-_a-zA-Z]*)\)?[-_a-zA-Z()]*)"
pr_train_dataloader,pr_eval_dataloader,eval_tokens= PR_data_load(data,tokenizer,batch_size=4,ratio=ratio,pattern=pattern)
eval_labels_list = PR_eval_labels_load(data,ratio)
print_2dlist_to_file(eval_tokens, './out/PR/eval_tokens.txt')


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

losses = []
for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    # Train the PR model
    loss = pr_train(PR_model,pr_train_dataloader,device,pr_optimizer,pr_scheduler,class_weight)
    end_time = time.time()
    losses.append(loss.item())
    print(f"epoch:{epoch} time:{end_time-start_time}")
append_loss_values_to_csv(losses,"./out/PR/loss_log.csv")
whole_loss = read_list_from_csv('./out/PR/loss_log.csv')
draw_and_save_loss_curve(whole_loss,'./out/PR/loss_curve.png')
"""
eval model using gold input
"""
# Evaluate the models on the respective tasks
PR_model.eval()
#PR span-based eval
pr_eval(PR_model,pr_eval_dataloader,device)
#PR semantic-based eval
from util.utils import getPRPosFromPattern,read_2dintlist_from_file,read_2dstrlist_from_file,convert_negatives,getPRTokenLabels,extractPredicate,filterPredicate
pattern = read_2dintlist_from_file('./out/PR/eval_result_pattern.txt')
tokens = read_2dstrlist_from_file('./out/PR/eval_tokens.txt')
patterns = convert_negatives(pattern)
positions = []
for p in patterns:
    t = getPRPosFromPattern(p)
    positions.append(t)
result = []
for i in range(len(positions)):
    t1 = getPRTokenLabels(positions[i],tokens[i])
    result.append(t1)
print(positions)
#using template to filter
pset = extractPredicate('./data/data_correct_formated.json')
predicate_list = filterPredicate(result,pset)
p,r,f = calculate_f1_score(predicate_list,eval_labels_list)
accuracy = getAccuracy(predicate_list,eval_labels_list)
print("p:{:.2f} r:{:.2f} f:{:.2f}".format(p,r,f))
print("accuracy:{:.2f}".format(accuracy))
"""
save model
"""
torch.save(PR_model.state_dict(), './fine-tuned_model/PR/BERT/PR_model_weight.pth')
