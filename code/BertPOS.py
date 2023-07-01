import json
import random
import transformers
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score, classification_report,accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from input_gen.pr_input_gen import get_sequence_label,combine_tokens_by_pattern,convert2bertformat
from transformers import BertTokenizer
from input_gen.pos_input_gen import convert2bertformat,pos_str2seq
import time
with open("/root/autodl-tmp/MGTC/data/data_correct_formated.json", encoding='utf-8') as f:
    data = json.load(f)
sentence_seq = []
label_seq = []
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
for sentence in data:
    tokens, pos = pos_str2seq(sentence["pos"])

    print(tokens)
    print(pos)
    assert len(tokens) == len(pos)

    label_set = (
    'B-n', 'I-n', 'B-np', 'I-np', 'B-ns', 'I-ns', 'B-ni', 'I-ni', 'B-nz', 'I-nz', 'B-m', 'I-m', 'B-q', 'I-q', 'B-mq',
    'I-mq', 'B-t', 'I-t', 'B-f', 'I-f', 'B-s', 'I-s', 'B-v', 'I-v', 'B-a', 'I-a', 'B-d', 'I-d', 'B-h', 'I-h', 'B-k',
    'I-k', 'B-i', 'I-i', 'B-j', 'I-j', 'B-r', 'I-r', 'B-c', 'I-c', 'B-p', 'I-p', 'B-u', 'I-u', 'B-y', 'I-y', 'B-e',
    'I-e', 'B-o', 'I-o', 'B-g', 'I-g', 'B-w', 'I-w', 'B-x', 'I-x')
    l2i = {x: i for i, x in enumerate(label_set)}
    s, i = convert2bertformat(tokenizer, 512, tokens, pos, l2i)
    sentence_seq.append(s)
    label_seq.append(i)
# Define the BERT-Chinese-Base model and tokenizer
model_name = 'bert-base-chinese'
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
model = transformers.BertForTokenClassification.from_pretrained(model_name, num_labels=len(label_set),ignore_mismatched_sizes=True)

pattern = "([-_a-zA-Z()]*\(?([-_a-zA-Z]*)\)?[-_a-zA-Z()]*)"
# cut data into train and eval data
ratio=0.8
leng = len(sentence_seq)
#Define the input sentences and their corresponding predicate labels for training
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
for epoch in range(10):
    start_time = time.time()
    for step, batch in enumerate(train_dataloader):
        batch_inputs, batch_labels = tuple(t.to(device) for t in batch)
        model.zero_grad()

        attention_mask = torch.where(torch.eq(batch_inputs, 0), torch.zeros_like(batch_inputs), torch.ones_like(batch_inputs))
        attention_mask.to(device)
        outputs = model(batch_inputs, labels=torch.where(batch_labels != -1, batch_labels, torch.tensor(-100).to(device)),attention_mask=attention_mask)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    end_time = time.time()
    print("Training time:", end_time-start_time, "seconds")
    print(f"Epoch {epoch+1} loss: {loss.item()}")

# Define the evaluation loop
model.eval()
eval_loss, eval_accuracy, eval_f1 = 0, 0, 0
predictions , true_labels = [], []
masks = []
for batch in eval_dataloader:
    batch_inputs, batch_labels = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        attention_mask = torch.where(torch.eq(batch_inputs, 0), torch.zeros_like(batch_inputs),
                                     torch.ones_like(batch_inputs))
        attention_mask.to(device)
        outputs = model(batch_inputs, labels=torch.where(batch_labels != -1, batch_labels, torch.tensor(-100).to(device)),attention_mask=attention_mask)
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
# Get the BERT model
bert_model = model.bert

# Save the BERT model
bert_model.save_pretrained('/root/autodl-tmp/MGTC/fine-tuned_model/pos_model')