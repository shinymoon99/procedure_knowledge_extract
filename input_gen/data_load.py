import torch
from torch.utils.data import DataLoader, Dataset,TensorDataset,RandomSampler, SequentialSampler
from input_gen import pr_input_gen
from input_gen.pr_input_gen import pr_convert2bertformat,pr_get_sequence_label,pr_combine_tokens_by_pattern
from input_gen.pos_input_gen import pos_str2seq, pos_convert2bertformat
from input_gen.srl_input_gen import srl_get_sequence_label, srl_combine_tokens_by_pattern, add_predicate
from transformers import BertTokenizer

pos_label_set = (
    'B-n', 'I-n', 'B-np', 'I-np', 'B-ns', 'I-ns', 'B-ni', 'I-ni', 'B-nz', 'I-nz', 'B-m', 'I-m', 'B-q', 'I-q', 'B-mq',
    'I-mq', 'B-t', 'I-t', 'B-f', 'I-f', 'B-s', 'I-s', 'B-v', 'I-v', 'B-a', 'I-a', 'B-d', 'I-d', 'B-h', 'I-h', 'B-k',
    'I-k', 'B-i', 'I-i', 'B-j', 'I-j', 'B-r', 'I-r', 'B-c', 'I-c', 'B-p', 'I-p', 'B-u', 'I-u', 'B-y', 'I-y', 'B-e',
    'I-e', 'B-o', 'I-o', 'B-g', 'I-g', 'B-w', 'I-w', 'B-x', 'I-x')
def POS_data_load(data,tokenizer,batch_size):

    sentence_seq = []
    label_seq = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    for sentence in data:
        tokens, pos = pos_str2seq(sentence["pos"])

        print(tokens)
        print(pos)
        assert len(tokens) == len(pos)

        label_set = (
            'B-n', 'I-n', 'B-np', 'I-np', 'B-ns', 'I-ns', 'B-ni', 'I-ni', 'B-nz', 'I-nz', 'B-m', 'I-m', 'B-q', 'I-q',
            'B-mq',
            'I-mq', 'B-t', 'I-t', 'B-f', 'I-f', 'B-s', 'I-s', 'B-v', 'I-v', 'B-a', 'I-a', 'B-d', 'I-d', 'B-h', 'I-h',
            'B-k',
            'I-k', 'B-i', 'I-i', 'B-j', 'I-j', 'B-r', 'I-r', 'B-c', 'I-c', 'B-p', 'I-p', 'B-u', 'I-u', 'B-y', 'I-y',
            'B-e',
            'I-e', 'B-o', 'I-o', 'B-g', 'I-g', 'B-w', 'I-w', 'B-x', 'I-x')
        l2i = {x: i for i, x in enumerate(label_set)}
        s, i = pos_convert2bertformat(tokenizer, 512, tokens, pos, l2i)
        sentence_seq.append(s)
        label_seq.append(i)

    pattern = "([-_a-zA-Z()]*\(?([-_a-zA-Z]*)\)?[-_a-zA-Z()]*)"
    # cut data into train and eval data
    ratio = 0.8
    leng = len(sentence_seq)
    # Define the input sentences and their corresponding predicate labels for training
    train_tokens = sentence_seq[:int(leng * ratio)]
    train_rel = label_seq[:int(leng * ratio)]

    eval_tokens = sentence_seq[int(leng * ratio):]
    eval_rel = label_seq[int(leng * ratio):]

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
    train_dataset = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    eval_dataset = TensorDataset(eval_inputs, eval_labels)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)

    return train_dataloader, eval_dataloader

def PR_data_load(data,tokenizer,batch_size,ratio):
    pattern = "([-_a-zA-Z()]*\(?([-_a-zA-Z]*)\)?[-_a-zA-Z()]*)"
    sentence_seq = []
    label_seq = []
    all_tokens = []
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
        final_tokens,s, i = pr_convert2bertformat(tokenizer, 512, combined_tokens, result_srl, l2i)
        print(s)
        print(i)
        all_tokens.append(final_tokens)
        sentence_seq.append(s)
        label_seq.append(i)
    # cut data into train and eval data
    leng = len(sentence_seq)
    # Define the input sentences and their corresponding predicate labels for training
    train_token_ids = sentence_seq[:int(leng * ratio)]
    train_rel = label_seq[:int(leng * ratio)]

    eval_token_ids = sentence_seq[int(leng * ratio):]
    eval_tokens = all_tokens[int(leng*ratio):]
    eval_rel = label_seq[int(leng * ratio):]

    # train_tokens = sentence_seq[:1]
    # train_rel = label_seq[:1]
    #
    # eval_tokens = sentence_seq[1:2]
    # eval_rel = label_seq[1:2]
    # Convert the tokenized inputs and labels to PyTorch tensors
    train_inputs = torch.tensor(train_token_ids)
    train_labels = torch.tensor(train_rel)
    eval_inputs = torch.tensor(eval_token_ids)
    eval_labels = torch.tensor(eval_rel)

    # Define the data loaders for training and evaluation

    train_dataset = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    eval_dataset = TensorDataset(eval_inputs, eval_labels)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)

    return train_dataloader,eval_dataloader,eval_tokens
def PR_eval_labels_load(data,ratio):
    sentenceslabels = []
    eval_labels = []
    for sentence_info in data:
        sentencelabels = []
        for label_set in sentence_info['labels']:
            sentencelabels.append(label_set['REL'])
        sentenceslabels.append(sentencelabels)
    leng = len(sentenceslabels)
    eval_labels = sentenceslabels[int(leng*ratio):]
    return eval_labels 
def SRL_data_load(data,l2i,batch_size,world_size=None,rank=None):
    pattern = "([-_a-zA-Z()]*\(?([-_a-zA-Z]*)\)?[-_a-zA-Z()]*)"
    tokenized_sentence = []
    rel_span = []
    final_srl = []

    for sentence in data:
        sentence_text = sentence['sentence']
        for rel in sentence['labels']:
            tokens, srl = srl_get_sequence_label(sentence_text, rel)
            combined_tokens, combined_srl = srl_combine_tokens_by_pattern(tokens, srl, pattern)
            result_srl = [x for x in combined_srl if x != 'X']
            print("combined_tokens:" + str(combined_tokens))

            print("result_srl:" + str(result_srl))
            # print(len(tokens))
            # print(len(srl))
            assert len(tokens) == len(srl) and len(combined_tokens) == len(result_srl)
            # add [CLS] and [SEP]+predicate+[SEP]
            tokens, label, span = add_predicate(combined_tokens, result_srl)
            print("after add [CLS] and [SEP]+predicate+[SEP]")
            print(tokens)
            print(span)
            print(label)
            tokenized_sentence.append(tokens)
            rel_span.append(span)
            final_srl.append(label)
    # split into train and eval
    # Define hyperparameters
    hidden_size = 768



    ratio = 0.8
    leng = len(tokenized_sentence)
    # Define the input sentences and their corresponding predicate labels for training
    train_tokens = tokenized_sentence[:int(leng * ratio)]
    train_span = rel_span[:int(leng * ratio)]
    train_srl = final_srl[:int(leng * ratio)]

    eval_tokens = tokenized_sentence[int(leng * ratio):]
    eval_span = rel_span[int(leng * ratio):]
    eval_srl = final_srl[int(leng * ratio):]

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # convert train_data into tensor
    # Tokenize the input sentences and their corresponding predicate labels for training
    train_token_ids = []
    train_label_ids = []
    for i, x in enumerate(train_tokens):
        input_ids = tokenizer.convert_tokens_to_ids(x)
        train_token_ids.append(input_ids)
        print(train_tokens[i])
        print(train_srl[i])
        print(i)
        train_label_ids.append([l2i[x] for x in train_srl[i]])

    # Tokenize the input sentences and their corresponding predicate labels for evaluation
    eval_token_ids = []
    eval_label_ids = []
    for i, x in enumerate(eval_tokens):
        input_ids = tokenizer.convert_tokens_to_ids(x)
        eval_token_ids.append(input_ids)
        eval_label_ids.append([l2i[x] for x in eval_srl[i]])

    # todo : do input id has 0,
    # Pad the id_lists of bert to a fixed length
    train_inputs = [token_ids + [0] * (512 - len(token_ids)) for token_ids in train_token_ids]
    train_labels = [label_ids + [-1] * (512 - len(label_ids)) for label_ids in train_label_ids]
    eval_inputs = [token_ids + [0] * (512 - len(token_ids)) for token_ids in eval_token_ids]
    eval_labels = [label_ids + [-1] * (512 - len(label_ids)) for label_ids in eval_label_ids]

    # Convert the tokenized inputs and labels to PyTorch tensors
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    eval_inputs = torch.tensor(eval_inputs)
    eval_labels = torch.tensor(eval_labels)
    train_span = torch.tensor(train_span)
    eval_span = torch.tensor(eval_span)
    # Define the data loaders for training and evaluation
    train_dataset = TensorDataset(train_inputs, train_span, train_labels)
    #train_sampler = RandomSampler(train_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    eval_dataset = TensorDataset(eval_inputs, eval_span, eval_labels)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)





    return train_dataloader, eval_dataloader
# TODO: this data load is useless, keep to see whether useful later, i may delete it later
def SRL_evaldata_loadFromPR(SRL_input,l2i,batch_size):
  
    sentences_tokens = []
    predicates = []
    input_tokens = []
    p_span = []
    
    for sentence in SRL_input:
        sentence_tokens = sentence['words']
        for p_info in sentence["predicates"]:
            ptokens = [t for t in p_info["ptext"]]
            span = p_info["span"] 
            temp_tokens = sentence_tokens + ["[SEP]"] + ptokens + ["[SEP]"]
            input_tokens.append(temp_tokens)
            p_span.append(span)
           
            sentences_tokens.append(sentence_tokens)
            predicates.append(p_info["ptext"])

    # split into train and eval
    # Define hyperparameters
    hidden_size = 768
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  
    # convert train_data into tensor

    # Tokenize the input sentences and their corresponding predicate labels for evaluation
    eval_token_ids = []
    eval_label_ids = []
    for i, x in enumerate(input_tokens):
        input_ids = tokenizer.convert_tokens_to_ids(x)
        eval_token_ids.append(input_ids)
    # todo : do input id has 0,
    # Pad the id_lists of bert to a fixed length
    eval_inputs = [token_ids + [0] * (512 - len(token_ids)) for token_ids in eval_token_ids]


    # Convert the tokenized inputs and labels to PyTorch tensors
    eval_inputs = torch.tensor(eval_inputs)
    eval_span = torch.tensor(p_span)
    # Define the data loaders for training and evaluation
    eval_dataset = TensorDataset(eval_inputs, eval_span)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)
    
    return eval_dataloader,input_tokens,sentences_tokens,predicates
def SRL_eval_data_load(data,l2i,batch_size,world_size=None,rank=None):
    pattern = "([-_a-zA-Z()]*\(?([-_a-zA-Z]*)\)?[-_a-zA-Z()]*)"
    tokenized_sentence = []
    rel_span = []
    final_srl = []

    for sentence in data:
        sentence_text = sentence['sentence']
        for rel in sentence['labels']:
            tokens, srl = srl_get_sequence_label(sentence_text, rel)
            combined_tokens, combined_srl = srl_combine_tokens_by_pattern(tokens, srl, pattern)
            result_srl = [x for x in combined_srl if x != 'X']
            print("combined_tokens:" + str(combined_tokens))

            print("result_srl:" + str(result_srl))
            # print(len(tokens))
            # print(len(srl))
            assert len(tokens) == len(srl) and len(combined_tokens) == len(result_srl)
            # add [CLS] and [SEP]+predicate+[SEP]
            tokens, label, span = add_predicate(combined_tokens, result_srl)
            print("after add [CLS] and [SEP]+predicate+[SEP]")
            print(tokens)
            print(span)
            print(label)
            tokenized_sentence.append(tokens)
            rel_span.append(span)
            final_srl.append(label)
    # split into train and eval
    # Define hyperparameters
    hidden_size = 768



    ratio = 0.8
    leng = len(tokenized_sentence)
    # Define the input sentences and their corresponding predicate labels for training
    eval_tokens = tokenized_sentence[int(leng * ratio):]
    eval_span = rel_span[int(leng * ratio):]
    eval_srl = final_srl[int(leng * ratio):]

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # convert train_data into tensor


    # Tokenize the input sentences and their corresponding predicate labels for evaluation
    eval_token_ids = []
    eval_label_ids = []
    for i, x in enumerate(eval_tokens):
        input_ids = tokenizer.convert_tokens_to_ids(x)
        eval_token_ids.append(input_ids)
        eval_label_ids.append([l2i[x] for x in eval_srl[i]])

    # todo : do input id has 0,
    # Pad the id_lists of bert to a fixed length

    eval_inputs = [token_ids + [0] * (512 - len(token_ids)) for token_ids in eval_token_ids]
    eval_labels = [label_ids + [-1] * (512 - len(label_ids)) for label_ids in eval_label_ids]

    # Convert the tokenized inputs and labels to PyTorch tensors

    eval_inputs = torch.tensor(eval_inputs)
    eval_labels = torch.tensor(eval_labels)

    eval_span = torch.tensor(eval_span)
    # Define the data loaders for evaluation
    eval_dataset = TensorDataset(eval_inputs, eval_span, eval_labels)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)





    return  eval_dataloader,eval_tokens