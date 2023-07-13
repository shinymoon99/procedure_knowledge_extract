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


def calculate_f1_score(predicted, gold_standard):
    prec = precision(predicted, gold_standard)
    rec = recall(predicted, gold_standard)
    f1 = 2 * ((prec * rec) / (prec + rec))
    return prec,rec,f1