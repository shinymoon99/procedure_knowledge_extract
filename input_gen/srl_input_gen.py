import json
import re

tokens = []
srl = []
pattern = "([-_a-zA-Z()]*\(?([-_a-zA-Z]*)\)?[-_a-zA-Z()]*)"

#arg label
#arg sentence
#return
#par tokens
#par srl
#note:contain space' '
def srl_get_sequence_label(sentence,labels):
    tokens = [x for x in sentence]
    srl= ['O']*len(sentence)
    #rel
    rspan = labels["span"]
    srl[rspan[0]:rspan[1]+1]=['REL']*(rspan[1]-rspan[0]+1)
    #arg
    for arg in labels["ARG"].items():
        argname=arg[0]
        text_and_span=arg[1]
        span = text_and_span["span"]
        srl[span[0]:span[1]+1]=[argname]*(span[1]-span[0]+1)
    #argm
    if "ARGM" in labels:
        for argm in labels["ARGM"].items():
            argname=argm[0]
            text_and_span=argm[1]
            span = text_and_span["span"]
            srl[span[0]:span[1]+1]=[argname]*(span[1]-span[0]+1)

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
def srl_combine_tokens_by_pattern(token,srl,pattern):
    srl_temp = srl[:]
    token_temp = token
    for i in re.finditer(pattern,"".join(token)):
        srl_temp[i.start()+1:i.end()] = ['X']*(i.end()-i.start()-1)
    for i,x in enumerate(token):
        if x==' ':
            srl_temp[i] = 'X'
    token_temp =split_sentence("".join(token),pattern)
    return token_temp,srl_temp

import re


#combined_tokens, combined_srl = combine_tokens_by_pattern(tokens, srl, pattern)
# print(combined_tokens)
# print(combined_srl)
# result_srl = [x for x in combined_srl if x != 'X']
# print(result_srl)
# print(len(tokens))
# print(len(srl))
# print(len(combined_tokens))
# print(len(result_srl))

def find_indices(lst, element):
    try:
        first_index = lst.index(element)
        last_index = len(lst) - lst[::-1].index(element) - 1
        return [first_index, last_index]
    except ValueError:
        return None


'''convert tokens_list to id
    :param token_list

'''


def add_predicate(token_list, label_list):
    # get token_list rel
    span = find_indices(label_list, 'REL')
    predicate_list = ['[SEP]'] + [x for x in token_list[span[0]:span[1]+1]] + ['[SEP]']
    temp_token_list = token_list[:]

    temp_token_list = ['[CLS]'] + temp_token_list + predicate_list

    temp_label_list = label_list[:]
    temp_label_list[span[0]:span[1] + 1] = ['O'] * (span[1] + 1 - span[0])
    temp_label_list = ['O'] + temp_label_list

    for i in range(len(temp_label_list)):
        assert temp_label_list[i]!='REL'
    result_span = [x + 1 for x in span]


    return temp_token_list, temp_label_list, result_span
if  __name__=='__main__':
    with open('/data/data.json', encoding='utf-8') as f:
        data = json.load(f)
    for sentence in data:
        sentence_text = sentence['sentence']
        for rel in sentence['labels']:
            tokens,srl = srl_get_sequence_label(sentence_text,rel)
            combined_tokens, combined_srl = srl_combine_tokens_by_pattern(tokens, srl, pattern)
            result_srl = [x for x in combined_srl if x != 'X']
            print(combined_tokens)

            print(result_srl)
            # print(len(tokens))
            # print(len(srl))
            assert len(tokens)==len(srl) and len(combined_tokens)==len(result_srl)
            #add [CLS] and [SEP]+predicate+[SEP]
            token,span,label = add_predicate(combined_tokens,result_srl)
            print(token)
            print(span)
            print(label)
