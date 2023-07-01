import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForTokenClassification
from models.model import  Bert_GRU,Bert_SRL
from input_gen.data_load import POS_data_load,PR_data_load,SRL_data_load,SRL_eval_data_load,pos_label_set
from models.model_train import pos_train,pr_train,srl_train
from models.model_eval import pos_eval,pr_eval,srl_eval
import transformers
import json
import time
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
import os


class SharedBertModel(torch.nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        return bert_output

def main(rank,world_size,SRL_model,data,l2i,NUM_EPOCHS):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # Set the CUDA device for this process
    torch.cuda.set_device(rank)
    # Set up distributed training
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device('cuda')

    # Define SRL data
    srl_train_dataloader, srl_eval_dataloader = SRL_data_load(data, l2i, 8, world_size, rank)
    """
    train model
    """
    # Fine-tune the models on the respective tasks

    SRL_model.to(device)
    SRL_model = torch.nn.parallel.DistributedDataParallel(SRL_model,device_ids=[rank],find_unused_parameters=True)

    SRL_model.train()

    LEARNING_RATE = 2e-5

    srl_optimizer = torch.optim.Adam(SRL_model.parameters(), lr=LEARNING_RATE)

    srl_total_steps = len(srl_train_dataloader) * 10  # 10 epochs

    srl_scheduler = transformers.get_linear_schedule_with_warmup(srl_optimizer, num_warmup_steps=0,
                                                                 num_training_steps=srl_total_steps)

    class_weight = torch.tensor([1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # Train the SRL model
        srl_train(SRL_model, srl_train_dataloader, rank, srl_optimizer, srl_scheduler, class_weight)
        end_time = time.time()
        print(f"epoch:{epoch} time:{end_time - start_time}")
    # save the model from the process with rank 0
    if rank == 0:
        torch.save(SRL_model.state_dict(), '/root/autodl-tmp/MGTC/fine-tuned_model/srl_model_default/BERT_SRL_weight.pth')
    # Clean up
    dist.destroy_process_group()


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    """
    load model
    """


    bert_model = BertModel.from_pretrained('bert-base-chinese', ignore_mismatched_sizes=True)
    shared_bert_model = SharedBertModel(bert_model)

    # Define the SRL model with a BiGRU layer
    hidden_size = 768
    # srl_label_set = ("O","REL","A0","A1","A2","A3","A4","ADV","CND","PRP","TMP","MNR")
    srl_label_set = ("O", "A0", "A1", "A2", "A3", "A4", "ADV", "CND", "PRP", "TMP", "MNR")
    # srl_label_set = ("O","A0","A1","A2")
    num_labels = len(srl_label_set)  # Number of labels: B-PRED, I-PRED, B-CON
    l2i = {label: i for i, label in enumerate(srl_label_set)}
    bert_model = BertModel.from_pretrained('bert-base-chinese', ignore_mismatched_sizes=True)
    SRL_model = Bert_SRL(shared_bert_model.bert, hidden_size, num_labels)


    #SRL_model.load_state_dict(torch.load('/root/autodl-tmp/MGTC/fine-tuned_model/srl_model_default/BERT_SRL_weight.pth'))


    """
    load data
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # get data
    with open('/root/autodl-tmp/MGTC/data/data_correct_formated.json', encoding='utf-8') as f:
        data = json.load(f)


    NUM_EPOCHS = 10
    mp.spawn(main, args=(torch.cuda.device_count(),SRL_model,data,l2i,NUM_EPOCHS), nprocs=torch.cuda.device_count())


    """
    eval model
    """
    state_dict = torch.load('/root/autodl-tmp/MGTC/fine-tuned_model/srl_model_default/BERT_SRL_weight.pth')
    new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    SRL_model.load_state_dict(new_state_dict)

    # Evaluate the models on the respective tasks

    device = torch.device('cuda')
    SRL_model.to(device)
    SRL_model.eval()


    # Define SRL data
    srl_eval_dataloader = SRL_eval_data_load(data, l2i, 8)
    #SRL eval
    srl_eval(SRL_model,srl_eval_dataloader,device,srl_label_set)
    torch.save(SRL_model.state_dict(), '/root/autodl-tmp/MGTC/fine-tuned_model/srl_model_default/BERT_SRL_weight.pth')

