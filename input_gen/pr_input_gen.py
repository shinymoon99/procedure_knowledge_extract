import json
import re
from transformers import BertTokenizer
# Define the labels for the predicate and non-predicate tokens
label_list = ["O", "B-PRED", "I-PRED"]
pattern = "([-_a-zA-Z()]*\(?([-_a-zA-Z]*)\)?[-_a-zA-Z()]*)"
#input
#output
tokens = ['P', 'C', 'F', '根', '据', '上', '报', '的', '消', '息', '中', '携', '带', '的', '信', '息', '和', '用', '户', '的', '签', '约', '数', '据', '作', '出', '策', '略', '判', '断', '，', '生', '成', '对', '应', '的', 'A', 'M', '策', '略', '关', '联', '.']
pos = ['A0', 'A0', 'A0', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'ARGM-COM', 'O', 'O', 'predicate', 'predicate', 'predicate', 'predicate', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'predicate', 'predicate', 'predicate', 'predicate', 'O']


#arg labels
#arg sentence
#return
#par tokens
#par srl
#note:contain space' '
def pr_get_sequence_label(sentence,labels):
    tokens = [x for x in sentence]
    srl= ['O']*len(sentence)
    #rel
    for label in labels:
        rspan = label["span"]
        srl[rspan[0]]='B-REL'
        if rspan[0]!=rspan[1]:
            srl[rspan[0]+1:rspan[1]+1]=['I-REL']*(rspan[1]-rspan[0])
    return tokens,srl
def split_sentence(sentence, predefined_expression):
    # Join predefined expressions with '|'
    #predefined_regex = '|'.join(predefined_expressions)

    # Split the sentence by the predefined expressions as a whole
    split_by_predefined = re.split(predefined_expression, sentence)

    # Split the resulting chunks by character
    split_by_char = [chunk.split() for chunk in split_by_predefined]

    # Flatten the nested list and remove empty strings
    split_by_char = [word for sublist in split_by_char for word in sublist if word]

    return split_by_char
def pr_combine_tokens_by_pattern(token,srl,pattern):
    srl_temp = srl
    token_temp = token
    for i in re.finditer(pattern,"".join(token)):
        srl_temp[i.start()+1:i.end()] = ['X']*(i.end()-i.start()-1)
    for i,x in enumerate(token):
        if x==' ':
            srl_temp[i] = 'X'
    token_temp =split_sentence("".join(token),pattern)
    return token_temp,srl_temp
def pr_convert2bertformat(tokenizer,dim,token_seq,label_seq,l2i):
    temp_token_seq = ['CLS']+token_seq
    temp_label_seq = ['O']+label_seq

    #convert to id
    temp_token_seq = tokenizer.convert_tokens_to_ids(temp_token_seq)
    temp_label_seq = [l2i[x] for x in temp_label_seq]
    # add pad
    temp_token_seq = temp_token_seq + [0]*(dim-len(temp_token_seq))
    temp_label_seq = temp_label_seq + [-1]*(dim-len(temp_label_seq))
    return temp_token_seq,temp_label_seq
if  __name__=='__main__':
    with open('/data/data.json', encoding='utf-8') as f:
        data = json.load(f)
    sentence_seq = []
    label_seq = []
    for sentence in data:
        sentence_text = sentence['sentence']

        tokens, srl = get_sequence_label(sentence_text,sentence['labels'])
        combined_tokens, combined_srl = combine_tokens_by_pattern(tokens, srl, pattern)
        result_srl = [x for x in combined_srl if x != 'X']
        #print(combined_tokens)

        #print(result_srl)
        # print(len(tokens))
        # print(len(srl))
        assert len(tokens) == len(srl) and len(combined_tokens) == len(result_srl)
        # convert to format
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        label_set = ('B-REL','I-REL','O')
        l2i = {x:i for i,x in enumerate(label_set)}
        s,i = pr_convert2bertformat(tokenizer,512,combined_tokens,result_srl,l2i)
        print(s)
        print(i)
        sentence_seq.append(s)
        label_seq.append(i)
