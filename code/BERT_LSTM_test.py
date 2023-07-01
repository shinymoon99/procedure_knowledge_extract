import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset,TensorDataset
import transformers
from transformers import BertTokenizer, BertModel
from torch.utils.data import RandomSampler, SequentialSampler
import json
from sklearn.metrics import precision_recall_fscore_support
from input_gen.srl_input_gen import srl_get_sequence_label,srl_combine_tokens_by_pattern,add_predicate
import numpy as np
import time
pattern = "([-_a-zA-Z()]*\(?([-_a-zA-Z]*)\)?[-_a-zA-Z()]*)"
def masked_cross_entropy(input_tensor,target_tensor,attention_mask):
    # input_tensor has shape (batch_size, seq_len, num_labels)
    # target_tensor has shape (batch_size, seq_len)
    # attention_mask has shape (batch_size, seq_len)

    # Flatten the input and target tensors
    input_flat = input_tensor.view(-1, num_labels)
    target_flat = target_tensor.view(-1)

    # Flatten the attention_mask tensor and set the loss for invalid positions to 0
    mask_flat = attention_mask.view(-1)

    # Compute the cross-entropy loss with the attention mask
    loss = torch.nn.functional.cross_entropy(input_flat, target_flat, ignore_index=0)

    # Compute the mean loss over the valid positions
    masked_loss = loss * (mask_flat != 0).float()
    masked_loss = masked_loss.sum() / mask_flat.sum()

    return masked_loss

class Bert_LSTM(nn.Module):
    def __init__(self, bert, hidden_size, num_labels):
        super(Bert_LSTM, self).__init__()
        self.bert = bert
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        #self.lstm = nn.LSTM(bert.config.hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(input_size=bert.config.hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size * 6, num_labels)
        self.dropout = nn.Dropout(bert.config.hidden_dropout_prob)

    def forward(self, input_ids, spans,attention_mask=None, attention_mask_gru=None,token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        bert_output = outputs.last_hidden_state
        sequence_output, _ = self.gru(bert_output)
        attention_mask_gru = attention_mask_gru.unsqueeze(-1)
        sequence_output = sequence_output*attention_mask_gru
        begin_pos = spans[:,0]
        end_pos = spans[:,1]
        row_indices = torch.arange(len(spans))
        begin_embeddings = sequence_output[row_indices, begin_pos]
        end_embeddings = sequence_output[row_indices, end_pos]
        predicate_output = torch.cat([begin_embeddings, end_embeddings], dim=1)#(batch_size, hidden_size*2)
        sequence_output = torch.cat([sequence_output, predicate_output.unsqueeze(1).repeat(1, sequence_output.shape[1], 1)], dim=2)
        sequence_output = self.dropout(sequence_output)
        logits = self.fc(sequence_output)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
tokenized_sentence = []
rel_span = []
final_srl = []

with open('/root/autodl-tmp/MGTC/data/data_correct_formated.json', encoding='utf-8') as f:
    data = json.load(f)
for sentence in data:
    sentence_text = sentence['sentence']
    for rel in sentence['labels']:
        tokens,srl = srl_get_sequence_label(sentence_text,rel)
        combined_tokens, combined_srl = srl_combine_tokens_by_pattern(tokens, srl, pattern)
        result_srl = [x for x in combined_srl if x != 'X']
        print("combined_tokens:"+str(combined_tokens))

        print("result_srl:"+str(result_srl))
        # print(len(tokens))
        # print(len(srl))
        assert len(tokens)==len(srl) and len(combined_tokens)==len(result_srl)
        #add [CLS] and [SEP]+predicate+[SEP]
        tokens,label,span = add_predicate(combined_tokens,result_srl)
        print("after add [CLS] and [SEP]+predicate+[SEP]")
        print(tokens)
        print(span)
        print(label)
        tokenized_sentence.append(tokens)
        rel_span.append(span)
        final_srl.append(label)
#split into train and eval
# Define hyperparameters
hidden_size = 768

label_set = ("O","REL","A0","A1","A2","A3","A4","ADV","CND","PRP","TMP","MNR")
num_labels = len(label_set)  # Number of labels: B-PRED, I-PRED, B-CON

l2i = {label: i for i, label in enumerate(label_set)}
# model
#bert_model = BertModel.from_pretrained('/root/autodl-tmp/MGTC/fine-tuned_model/predicate_model')
bert_model = BertModel.from_pretrained('bert-base-chinese',ignore_mismatched_sizes=True)
model = Bert_LSTM(bert_model, hidden_size, num_labels)

ratio=0.8
leng = len(tokenized_sentence)
# Define the input sentences and their corresponding predicate labels for training
train_tokens = tokenized_sentence[:int(leng*ratio)]
train_span = rel_span[:int(leng*ratio)]
train_srl = final_srl[:int(leng*ratio)]

eval_tokens = tokenized_sentence[int(leng*ratio):]
eval_span = rel_span[int(leng*ratio):]
eval_srl = final_srl[int(leng*ratio):]

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')




# convert train_data into tensor
# Tokenize the input sentences and their corresponding predicate labels for training
train_token_ids = []
train_label_ids = []
for i,x in enumerate(train_tokens):
    input_ids = tokenizer.convert_tokens_to_ids(x)
    train_token_ids.append(input_ids)
    print(train_tokens[i])
    print(train_srl[i])
    print(i)
    train_label_ids.append([l2i[x] for x in train_srl[i]])


# Tokenize the input sentences and their corresponding predicate labels for evaluation
eval_token_ids = []
eval_label_ids = []
for i,x in enumerate(eval_tokens):
    input_ids = tokenizer.convert_tokens_to_ids(x)
    eval_token_ids.append(input_ids)
    eval_label_ids.append([l2i[x] for x in eval_srl[i]])

#todo : do input id has 0,
# Pad the id_lists of bert to a fixed length
train_inputs = [token_ids + [0] * (512 - len(token_ids)) for token_ids in train_token_ids]
train_labels =  [label_ids + [-1] * (512 - len(label_ids)) for label_ids in train_label_ids]
eval_inputs = [token_ids + [0] * (512 - len(token_ids)) for token_ids in eval_token_ids]
eval_labels =  [label_ids + [-1] * (512 - len(label_ids)) for label_ids in eval_label_ids]

# Convert the tokenized inputs and labels to PyTorch tensors
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
eval_inputs = torch.tensor(eval_inputs)
eval_labels = torch.tensor(eval_labels)
train_span = torch.tensor(train_span)
eval_span = torch.tensor(eval_span)
# Define the data loaders for training and evaluation
batch_size = 8
train_dataset = TensorDataset(train_inputs,train_span, train_labels)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
eval_dataset = TensorDataset(eval_inputs,eval_span, eval_labels)
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)
# define the SRL loss function
class_weight = torch.tensor([1.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weight)
# define the SRL optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.to(device)
model.train()

# define the SRL training loop
num_epochs =100
print("epoch_num:"+str(num_epochs))

for epoch in range(num_epochs):
    start = time.time()
    total_loss = 0.0

    for batch in train_dataloader:
        inputs, span,targets = batch
        optimizer.zero_grad()
        # Create a new tensor with the same shape as input_ids and fill it with 0s
        attention_mask = torch.zeros_like(inputs)
        # Fill the tensor with 1s where input_ids is not equal to 0, and with 0s otherwise
        attention_mask = torch.where(inputs != 0, torch.tensor(1), attention_mask)
        # Create a mask with all values set to 1 initially
        mask_tensor = torch.ones_like(inputs)
        # Find the first occurrence of the [SEP] token along the sequence dimension (dim=1)
        sep_indices = np.argmax(inputs == 102, axis=1)
        # For each example in the batch, set all values in the mask tensor after the first [SEP] token to 0
        for i in range(inputs.size(0)):
            if sep_indices[i] < inputs.size(1):
                mask_tensor[i, sep_indices[i]:] = 0
        attention_mask_gru = mask_tensor
        # put data to gpu
        inputs = inputs.to(device)
        span = span.to(device)
        targets = targets.to(device)
        attention_mask = attention_mask.to(device)
        attention_mask_gru = attention_mask_gru.to(device)

        logits = model(inputs,spans=span,attention_mask=attention_mask,attention_mask_gru=attention_mask_gru)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        #loss = masked_cross_entropy(logits,targets,attention_mask_gru)
        loss.backward()
        total_loss+=loss.item()

        optimizer.step()
    end = time.time()
    print('Epoch: {}, loss: {:.4f} time: {:.4f}'.format(epoch+1, total_loss/len(train_dataloader),end-start))

# define the SRL evaluation loop

criterion = nn.CrossEntropyLoss(ignore_index=-1)
with torch.no_grad():
    total_loss = 0
    total_correct = 0
    total_examples = 0
    true_labels = []
    predictions_list = []
    masks = []
    for batch in eval_dataloader:
        inputs, span, targets = batch

        # Create a new tensor with the same shape as input_ids and fill it with 0s
        attention_mask = torch.zeros_like(inputs)
        # Fill the tensor with 1s where input_ids is not equal to 0, and with 0s otherwise
        attention_mask = torch.where(inputs != 0, torch.tensor(1), attention_mask)

        # Create a mask with all values set to 1 initially
        mask_tensor = torch.ones_like(inputs)

        # Find the first occurrence of the [SEP] token along the sequence dimension (dim=1)
        sep_indices = np.argmax(inputs == 102, axis=1)#get wrong if a sequence do not contain corresponding sign

        # For each example in the batch, set all values in the mask tensor after the first [SEP] token to 0

        for i in range(inputs.size(0)):
            if sep_indices[i] < inputs.size(1):
                mask_tensor[i, sep_indices[i]:] = 0

        attention_mask_gru = mask_tensor[:]

        # put everything to gpu
        inputs = inputs.to(device)
        span = span.to(device)
        targets = targets.to(device)
        attention_mask = attention_mask.to(device)
        attention_mask_gru = attention_mask_gru.to(device)

        logits = model(inputs, spans=span, attention_mask=attention_mask, attention_mask_gru=attention_mask_gru)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        total_loss += loss.item() * inputs.shape[0]

        predictions = torch.argmax(logits, dim=-1)
        total_correct += torch.sum(predictions == targets).item()
        total_examples += torch.sum(torch.ne(targets, -1))

        logits = logits.detach().cpu().numpy()
        predictions_list.extend([list(p) for p in np.argmax(logits, axis=2)])
        label_ids = targets.to("cpu").numpy()
        true_labels.extend(label_ids)
        attention_numpy = attention_mask_gru.to("cpu").numpy()
        masks.extend(attention_numpy)
    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    print("avg_loss:%.6f accuracy:%.6f" % (avg_loss, accuracy))

    # Flatten the predictions and true labels
    predictions_flat = [p for pred in predictions_list for p in pred]
    true_labels_flat = [t for label in true_labels for t in label]
    # use attention to delete unneeded ones
    attention_mask_flat = [f for mask in masks for f in mask]

    for i,x in enumerate(attention_mask_flat):
        if attention_mask_flat[i]== 1:
            assert true_labels_flat[i]!=-1
        if attention_mask_flat[i]== 0:
            assert true_labels_flat[i]==-1

    predictions_flat = [predictions_flat[i] for i in range(len(predictions_flat)) if attention_mask_flat[i]]
    true_labels_flat = [true_labels_flat[i] for i in range(len(true_labels_flat)) if attention_mask_flat[i]]


    label_set = ("O", "REL", "A0", "A1", "A2", "A3", "A4", "ADV", "CND", "PRP", "TMP", "MNR")
    num_labels = len(label_set)  # Number of labels: B-PRED, I-PRED, B-CON
    l2i = {label: i for i, label in enumerate(label_set)}
    label = list(label_set)
    # calculate precision, recall, and f1 score for each label
    p, r, f1, support = precision_recall_fscore_support(true_labels_flat, predictions_flat, average=None)
    class_tags = sorted(set(true_labels_flat) | set(predictions_flat))
    class_tags = [x for x in class_tags if x != -1]
    # print the results
    for i in range(len(p)):
        print(f"Label {label[class_tags[i]]}: precision={p[i]}, recall={r[i]}, f1={f1[i]},support={support[i]}")
