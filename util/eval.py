from util.utils import get_positions, print_2dlist_to_file, read_2dintlist_from_file,get_different_num_positions,read_2dstrlist_from_file,get_token_labels
def precision(predicted, gold_standard):
    true_positives = 0
    false_positives = 0

    for pred_sent, gold_sent in zip(predicted, gold_standard):
        pred_set = set(pred_sent)
        gold_set = set(gold_sent)
        true_positives += len(pred_set & gold_set)
        false_positives += len(pred_set - gold_set)

    precision = true_positives / (true_positives + false_positives)
    return precision


def recall(predicted, gold_standard):
    true_positives = 0
    false_negatives = 0

    for pred_sent, gold_sent in zip(predicted, gold_standard):
        pred_set = set(pred_sent)
        gold_set = set(gold_sent)
        true_positives += len(pred_set & gold_set)
        false_negatives += len(gold_set - pred_set)

    recall = true_positives / (true_positives + false_negatives)
    return recall

def getAccuracy(predicted, gold_standard):
    correct = 0
    total = 0
    for pred_sent, gold_sent in zip(predicted, gold_standard):
        pred_set = set(pred_sent)
        gold_set = set(gold_sent)        
        correct += len(pred_set & gold_set)
        total += len(pred_set)
    accuracy = correct/total
    return accuracy
def calculate_f1_score(predicted, gold_standard):
    prec = precision(predicted, gold_standard)
    rec = recall(predicted, gold_standard)
    f1 = 2 * ((prec * rec) / (prec + rec))
    return prec,rec,f1

def getPredictedSRL(eval_pattern_file,SRL_eval_token_file):
    # # Rest of your code goes here

    nums = read_2dintlist_from_file(eval_pattern_file)
    tokens = read_2dstrlist_from_file(SRL_eval_token_file)
    #    srl_label_set = ("O", "A0", "A1", "A2", "A3", "A4", "ADV", "CND", "PRP", "TMP", "MNR") 0-10
    positions = []
    for num in nums:
        t = get_different_num_positions(num)
        positions.append(t)
    result = []
    for i in range(len(positions)):
        t1 = get_token_labels(positions[i],tokens[i])
        result.append(t1)

    #print(positions)
    """
        convert to normal form
    """
    result1 = result.copy()
    #only save the first one and delete '|'
    for pro in result1:
        for key,value in pro.items():
            if isinstance(value,list):
                pro[key] = value[0].replace('|','')
            else:
                pro[key] = value.replace('|','')
    #edit the keys from index to actual label
    srl_label_set = ("O","A0","A1","A2","A3","A4","ADV","CND","PRP","TMP","MNR")
    #srl_label_set = ("O","A0","A1","A2")
    num_labels = len(srl_label_set)  # Number of labels: B-PRED, I-PRED, B-CON
    i2l = { i:label for i, label in enumerate(srl_label_set)}
    prediction_SRL_list = []
    for pro in result1:
        new_dict = {i2l.get(k,k):v for k,v in pro.items()}
        new_dict.pop('O',None)
        prediction_SRL_list.append(new_dict)
    return prediction_SRL_list