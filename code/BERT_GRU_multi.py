import sys
sys.path.append('./')
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForTokenClassification
from models.model import  Bert_GRU,Bert_SRL
from input_gen.data_load import POS_data_load,PR_data_load,SRL_data_load,SRL_eval_data_load,pos_label_set,SRL_evaldata_loadFromPR
from models.model_train import pos_train,pr_train,srl_train
from models.model_eval import pos_eval,pr_eval,srl_eval,srl_eval_FromPRoutput
import transformers
import json
import time
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
import os
from util.utils import print_2dlist_to_file,extract_arguments,append_loss_values_to_csv,draw_and_save_loss_curve,read_list_from_csv
from util.eval import getPredictedSRL,calculate_f1_score,getAccuracy

# TODO:由于没有考虑相同输入的情况，输入必须保证每个句子只能一个，功能等待完善
class SharedBertModel(torch.nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        return bert_output
def correct_state_dict(state_dict):
    # Create a new state_dict with modified keys
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict
def main(rank,world_size,SRL_model,data,l2i,NUM_EPOCHS):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # Set the CUDA device for this process
    torch.cuda.set_device(rank)
    # Set up distributed training
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device('cuda')
    pattern = "([-_a-zA-Z()]*\(?([-_a-zA-Z]*)\)?[-_a-zA-Z()]*)"
    # Define SRL data
    srl_train_dataloader, srl_eval_dataloader = SRL_data_load(data, l2i, 8, pattern,world_size, rank)
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
    losses = []
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # Train the SRL model
        loss = srl_train(SRL_model, srl_train_dataloader, rank, srl_optimizer, srl_scheduler, class_weight)
        end_time = time.time()
        print(f"epoch:{epoch} time:{end_time - start_time}")
        losses.append(loss)
    
    append_loss_values_to_csv(losses,"./out/SRL/loss_log.csv")
    whole_loss = read_list_from_csv('./out/SRL/loss_log.csv')
    draw_and_save_loss_curve(whole_loss,'./out/SRL/loss_curve.png')
    # save the model from the process with rank 0
    if rank == 0:
        torch.save(SRL_model.state_dict(), './fine-tuned_model/SRL/BERT_SRL_weight.pth')
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


    # SRL_model.load_state_dict(torch.load('/root/autodl-tmp/MGTC/fine-tuned_model/srl_model_default/BERT_GRU_weight.pth'))
    # t = torch.load('/root/autodl-tmp/MGTC/fine-tuned_model/srl_model_default/BERT_SRL_weight.pth')
    # state_dict = correct_state_dict(t)
    # SRL_model.load_state_dict(state_dict)

    """
    load data
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # get data
    with open('./data/data_correct_formated.json', encoding='utf-8') as f:
        data = json.load(f)


    NUM_EPOCHS = 100
    mp.spawn(main, args=(torch.cuda.device_count(),SRL_model,data,l2i,NUM_EPOCHS), nprocs=torch.cuda.device_count())
    if os.path.isfile('./fine-tuned_model/SRL/BERT_SRL_weight.pth'):
        t1 = torch.load('./fine-tuned_model/SRL/BERT_SRL_weight.pth')
    else:
        pass
    
    state_dict = correct_state_dict(t1)
    SRL_model.load_state_dict(state_dict)
    """
    eval model
    """
    result_pattern_loc = "./out/SRL/eval_result_pattern.txt"
    # Evaluate the models on the respective tasks

    device = torch.device('cuda')
    SRL_model.to(device)
    SRL_model.eval()
    gold_arguments_list = extract_arguments('./data/data_correct_formated.json')

    # Define SRL data
    srl_eval_dataloader,eval_tokens = SRL_eval_data_load(data, l2i, 8)
    print_2dlist_to_file(eval_tokens, './out/SRL/eval_tokens.txt')
    #SRL eval
    srl_eval(SRL_model,srl_eval_dataloader,device,srl_label_set,result_pattern_loc)
    # TODO: this ratio should be synchronized with dataload part, which is editable
    gold_arguments_list = gold_arguments_list[int(0.8*len(gold_arguments_list)):]
    arguments_list = getPredictedSRL('./out/SRL/eval_result_pattern.txt','./out/SRL/eval_tokens.txt')
    p,r,f = calculate_f1_score(arguments_list,gold_arguments_list)
    accuracy = getAccuracy(arguments_list,gold_arguments_list)
    print("accuracy:{:.2f}".format(accuracy))
    print("p:{:.2f} r:{:.2f} f:{:.2f}".format(p,r,f))
    """
    eval model using PR output
    """
    # Evaluate the models on the respective tasks
    result_pattern_loc_PR = "./out/SRL/eval_result_pattern_PRoutput.txt"
    device = torch.device('cuda')
    SRL_model.to(device)
    SRL_model.eval()
    gold_arguments_list = extract_arguments('./data/data_correct_formated.json')

    with open('./out/PR/SRL_input.json', 'r') as file:
        PRoutput_data = json.load(file)
    # Define SRL data
    srl_eval_dataloader,eval_tokens,sentences_tokens,predicates,p_span= SRL_evaldata_loadFromPR(PRoutput_data, l2i, 8)
    print_2dlist_to_file(eval_tokens, './out/SRL/eval_tokens_PRoutput.txt')
    print_2dlist_to_file(p_span,'./out/SRL/eval_PredicateSpan_PRoutput.txt')
    #SRL eval
    srl_eval_FromPRoutput(SRL_model,srl_eval_dataloader,device,srl_label_set,result_pattern_loc_PR)


    gold_arguments_list = gold_arguments_list[int(0.8*len(gold_arguments_list)):]
    arguments_list = getPredictedSRL('./out/SRL/eval_result_pattern_PRoutput.txt','./out/SRL/eval_tokens_PRoutput.txt')
    # p,r,f = calculate_f1_score(arguments_list,gold_arguments_list)
    # accuracy = getAccuracy(arguments_list,gold_arguments_list)
    
    # print("accuracy:{:.2f}".format(accuracy))
    # print("p:{:.2f} r:{:.2f} f:{:.2f}".format(p,r,f))

    #eval
    