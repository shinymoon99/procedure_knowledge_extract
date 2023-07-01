import json
import random
import transformers
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score, classification_report,accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from input_gen.pr_input_gen import pr_get_sequence_label,pr_combine_tokens_by_pattern,pr_convert2bertformat
from transformers import BertTokenizer,BertModel,BertForTokenClassification,BertConfig
pattern = "([-_a-zA-Z()]*\(?([-_a-zA-Z]*)\)?[-_a-zA-Z()]*)"
#Define the BERT-Chinese-Base model and tokenizer
model_name = '/root/autodl-tmp/MGTC/fine-tuned_model/predicate_model_default/model_2'
model = transformers.BertForTokenClassification.from_pretrained(model_name, num_labels=3,ignore_mismatched_sizes=True)



# # Load the BERT base model
# base_model = BertForTokenClassification.from_pretrained('bert-base-chinese',ignore_mismatched_sizes=True)
# pos_model_loc = "/root/autodl-tmp/MGTC/fine-tuned_model/predicate_model_default/model_2"
# pos_model = BertModel.from_pretrained(pos_model_loc)
# # Define a new classification head on top of the loaded BERT base model
# num_labels = 3 # Change this to the number of labels in your task
# config = BertConfig.from_pretrained('bert-base-chinese', num_labels=num_labels)
# model = BertForTokenClassification(config=config)
# model.bert = pos_model
# #model.bert =BertModel.from_pretrained("bert-base-chinese",ignore_mismatched_sizes=True)
# model.dropout = base_model.dropout

class_weight = torch.tensor([ 100.0,100.0,1.0]) # assign higher weight to predicate class
model.loss_fn =nn.CrossEntropyLoss(weight=class_weight)


with open("/root/autodl-tmp/MGTC/data/data_correct_formated.json", encoding='utf-8') as f:
    data = json.load(f)
sentence_seq = []
label_seq = []
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
for sentence in data:
    sentence_text = sentence['sentence']

    tokens, srl = pr_get_sequence_label(sentence_text, sentence['labels'])
    combined_tokens, combined_srl = pr_combine_tokens_by_pattern(tokens, srl, pattern)
    result_srl = [x for x in combined_srl if x != 'X']
    # print(combined_tokens)

    # print(result_srl)
    # print(len(tokens))
    # print(len(srl))
    assert len(tokens) == len(srl) and len(combined_tokens) == len(result_srl)
    # convert to format

    label_set = ('B-REL', 'I-REL', 'O')
    l2i = {x: i for i, x in enumerate(label_set)}
    s, i = pr_convert2bertformat(tokenizer, 512, combined_tokens, result_srl, l2i)
    print(s)
    print(i)
    sentence_seq.append(s)
    label_seq.append(i)
# cut data into train and eval data
ratio=0.8
leng = len(sentence_seq)
# Define the input sentences and their corresponding predicate labels for training
train_tokens = sentence_seq[:int(leng*ratio)]
train_rel = label_seq[:int(leng*ratio)]

eval_tokens = sentence_seq[int(leng*ratio):]
eval_rel = label_seq[int(leng*ratio):]

# train_tokens = sentence_seq[:1]
# train_rel = label_seq[:1]
#
# eval_tokens = sentence_seq[1:2]
# eval_rel = label_seq[1:2]
# Convert the tokenized inputs and labels to PyTorch tensors
train_inputs = torch.tensor(train_tokens)
train_labels = torch.tensor(train_rel)
eval_inputs = torch.tensor(eval_tokens)
eval_labels = torch.tensor(eval_rel)

# Define the data loaders for training and evaluation
batch_size = 8
train_dataset = TensorDataset(train_inputs, train_labels)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
eval_dataset = TensorDataset(eval_inputs, eval_labels)
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)

# Define the optimizer and learning rate scheduler
optimizer = transformers.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_dataloader) * 10  # 10 epochs
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
#define mask_attention


# Define the training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model.train()
for epoch in range(100):
    for step, batch in enumerate(train_dataloader):

        model.zero_grad()
        batch_inputs,batch_labels = batch
        batch_labels = torch.where(batch_labels != -1, batch_labels, torch.tensor(-100))
        attention_mask = torch.where(torch.eq(batch_inputs, 0), torch.zeros_like(batch_inputs), torch.ones_like(batch_inputs))
        #batch_inputs, batch_labels = tuple(t.to(device) for t in batch)
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(batch_inputs, labels=batch_labels,attention_mask=attention_mask)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    print(f"Epoch {epoch+1} loss: {loss.item()}")

# Define the evaluation loop
model.eval()
eval_loss, eval_accuracy, eval_f1 = 0, 0, 0
predictions , true_labels = [], []
masks = []
for batch in eval_dataloader:
    batch_inputs, batch_labels = tuple(t.to(device) for t in batch)
    attention_mask = torch.where(torch.eq(batch_inputs, 0), torch.zeros_like(batch_inputs),
                                 torch.ones_like(batch_inputs))

    attention_mask = attention_mask.to(device)
    masked_batch_labels = torch.where(batch_labels != -1, batch_labels, torch.tensor(-100).to(device))
    with torch.no_grad():
        outputs = model(batch_inputs, labels=masked_batch_labels)
    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    label_ids = batch_labels.to("cpu").numpy()
    attention_numpy = attention_mask.to("cpu").numpy()
    masks.extend(attention_numpy)
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    true_labels.extend(label_ids)

# Flatten the predictions and true labels
predictions_flat = [p for pred in predictions for p in pred]
true_labels_flat = [t for label in true_labels for t in label]
# use attention to delete unneeded ones
attention_mask_flat = [f for mask in masks for f in mask]
predictions_flat = [predictions_flat[i] for i in range(len(predictions_flat)) if attention_mask_flat[i]]
true_labels_flat = [true_labels_flat[i] for i in range(len(true_labels_flat)) if attention_mask_flat[i]]
# Compute the evaluation metrics
eval_accuracy = accuracy_score(true_labels_flat, predictions_flat)
eval_f1 = f1_score(true_labels_flat, predictions_flat, average='macro')
print(f"Accuracy: {eval_accuracy}")
print(f"F1 score: {eval_f1}")
# calculate precision, recall, and f1 score for each label
p, r, f1, _ = precision_recall_fscore_support(true_labels_flat, predictions_flat, average=None)

# print the results
for i in range(len(p)):
    print(f"Label {i}: precision={p[i]}, recall={r[i]}, f1={f1[i]}")
model.save_pretrained("/root/autodl-tmp/MGTC/fine-tuned_model/predicate_model_default/model_2")