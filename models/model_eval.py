from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, classification_report,accuracy_score
import torch
import torch.nn as nn
import numpy as np
from util.utils import print_2dlist_to_file, replace_with_neg1
#POS eval
def pos_eval(model,eval_dataloader,device):
    # Define the evaluation loop
    model.eval()
    eval_loss, eval_accuracy, eval_f1 = 0, 0, 0
    predictions, true_labels = [], []
    masks = []
    for batch in eval_dataloader:
        batch_inputs, batch_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            attention_mask = torch.where(torch.eq(batch_inputs, 0), torch.zeros_like(batch_inputs),
                                         torch.ones_like(batch_inputs))
            attention_mask.to(device)
            outputs = model(batch_inputs,
                            labels=torch.where(batch_labels != -1, batch_labels, torch.tensor(-100).to(device)),
                            attention_mask=attention_mask)
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

    label_set = (
        'B-n', 'I-n', 'B-np', 'I-np', 'B-ns', 'I-ns', 'B-ni', 'I-ni', 'B-nz', 'I-nz', 'B-m', 'I-m', 'B-q', 'I-q',
        'B-mq',
        'I-mq', 'B-t', 'I-t', 'B-f', 'I-f', 'B-s', 'I-s', 'B-v', 'I-v', 'B-a', 'I-a', 'B-d', 'I-d', 'B-h', 'I-h', 'B-k',
        'I-k', 'B-i', 'I-i', 'B-j', 'I-j', 'B-r', 'I-r', 'B-c', 'I-c', 'B-p', 'I-p', 'B-u', 'I-u', 'B-y', 'I-y', 'B-e',
        'I-e', 'B-o', 'I-o', 'B-g', 'I-g', 'B-w', 'I-w', 'B-x', 'I-x')
    num_labels = len(label_set)  # Number of labels: B-PRED, I-PRED, B-CON
    l2i = {label: i for i, label in enumerate(label_set)}
    label = list(label_set)
    # calculate precision, recall, and f1 score for each label
    p, r, f1, support = precision_recall_fscore_support(true_labels_flat, predictions_flat, average=None)
    class_tags = sorted(set(true_labels_flat) | set(predictions_flat))
    # print the results
    for i in range(len(p)):
        print(f"Label {label[class_tags[i]]}: precision={p[i]}, recall={r[i]}, f1={f1[i]},support={support[i]}")

#PR eval
def pr_eval(model,eval_dataloader,device):
    model.eval()
    eval_loss, eval_accuracy, eval_f1 = 0, 0, 0
    predictions, true_labels = [], []
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
    # edit prediction to omit the space by using mask and print
    t = replace_with_neg1(predictions, masks)
    print_2dlist_to_file(t,"/root/autodl-tmp/procedure_knowledge_extract/out/PR/eval_result_pattern.txt")

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

    label_set = ('B-REL', 'I-REL', 'O')
    num_labels = len(label_set)  # Number of labels: B-PRED, I-PRED, B-CON
    l2i = {label: i for i, label in enumerate(label_set)}
    label = list(label_set)
    # calculate precision, recall, and f1 score for each label
    p, r, f1, support = precision_recall_fscore_support(true_labels_flat, predictions_flat, average=None)
    class_tags = sorted(set(true_labels_flat) | set(predictions_flat))
    # print the results
    for i in range(len(p)):
        print(f"Label {label[class_tags[i]]}: precision={p[i]}, recall={r[i]}, f1={f1[i]},support={support[i]}")

#SRL eval
def srl_eval(model,eval_dataloader,device,label_set):

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
            sep_indices = np.argmax(inputs == 102, axis=1)
            # For each example in the batch, set all values in the mask tensor after the first [SEP] token to 0
            for i in range(inputs.size(0)):
                if sep_indices[i] < inputs.size(1):
                    mask_tensor[i, sep_indices[i]:] = 0
            attention_mask_gru = mask_tensor

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
        # edit prediction to omit the space by using mask and print
        t = replace_with_neg1(predictions_list, masks)
        print_2dlist_to_file(t,"/root/autodl-tmp/procedure_knowledge_extract/out/eval_result_pattern.txt")
        # Flatten the predictions and true labels
        predictions_flat = [p for pred in predictions_list for p in pred]
        true_labels_flat = [t for label in true_labels for t in label]
        # use attention to delete unneeded ones
        attention_mask_flat = [f for mask in masks for f in mask]
        for i, x in enumerate(attention_mask_flat):
            if attention_mask_flat[i] == 1:
                assert true_labels_flat[i] != -1
            if attention_mask_flat[i] == 0:
                assert true_labels_flat[i] == -1
        predictions_flat = [predictions_flat[i] for i in range(len(predictions_flat)) if attention_mask_flat[i]]
        true_labels_flat = [true_labels_flat[i] for i in range(len(true_labels_flat)) if attention_mask_flat[i]]

        num_labels = len(label_set)  # Number of labels: B-PRED, I-PRED, B-CON
        l2i = {label: i for i, label in enumerate(label_set)}
        label = list(label_set)
        # calculate precision, recall, and f1 score for each label
        p, r, f1, support = precision_recall_fscore_support(true_labels_flat, predictions_flat, average=None)
        class_tags = sorted(set(true_labels_flat) | set(predictions_flat))
        class_tags = [x for x in class_tags if x != -1]
        # print the results
        for i in range(len(p)):
            print(f"Label {label[class_tags[i]]}: precision={p[i]:.4f}, recall={r[i]:.4f}, f1={f1[i]:.4f},support={support[i]}")
