from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, classification_report,accuracy_score
import torch
import torch.nn as nn
import numpy as np
def srl_run(model,eval_dataloader,device,label_set):

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
