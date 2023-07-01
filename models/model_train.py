import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

def reduce_tensor(tensor):
    # helper function to reduce a tensor across all the GPUs
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor
# Train the POS model
def pos_train(model, pos_train_dataloader, device, optimizer, scheduler):
    for step, batch in enumerate(pos_train_dataloader):
        batch_inputs, batch_labels = tuple(t.to(device) for t in batch)
        model.zero_grad()

        attention_mask = torch.where(torch.eq(batch_inputs, 0), torch.zeros_like(batch_inputs),
                                     torch.ones_like(batch_inputs))
        attention_mask.to(device)
        outputs = model(batch_inputs, labels=torch.where(batch_labels != -1, batch_labels, torch.tensor(-100).to(device)),
                        attention_mask=attention_mask)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
# Train the NER model
def pr_train(model, pr_train_dataloader, device, optimizer,scheduler,weight):
    for step, batch in enumerate(pr_train_dataloader):
        model.zero_grad()
        batch_inputs, batch_labels = batch
        batch_labels = torch.where(batch_labels != -1, batch_labels, torch.tensor(-100))
        attention_mask = torch.where(torch.eq(batch_inputs, 0), torch.zeros_like(batch_inputs),
                                     torch.ones_like(batch_inputs))
        # batch_inputs, batch_labels = tuple(t.to(device) for t in batch)
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        attention_mask = attention_mask.to(device)
        weight = weight.to(device)
        outputs = model(batch_inputs, labels=batch_labels, attention_mask=attention_mask,weight=weight)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
# Train the SRL model
def srl_train(model, srl_train_dataloader, device, optimizer,scheduler,class_weight,rank=None,):
    total_loss = 0.0
    #label_set = ("O", "REL", "A0", "A1", "A2", "A3", "A4", "ADV", "CND", "PRP", "TMP", "MNR")
    #class_weight = torch.tensor([1.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0]).to(device)
    class_weight = class_weight.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1,weight=class_weight)
    for batch in srl_train_dataloader:
        inputs, span, targets = batch
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

        logits = model(inputs, spans=span, attention_mask=attention_mask, attention_mask_gru=attention_mask_gru)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        # loss = masked_cross_entropy(logits,targets,attention_mask_gru)
        loss.backward()
        #todo total lost 好像没用
        total_loss += loss.item()

        # synchronize the gradients across all the GPUs
        torch.distributed.barrier()

        # compute the average loss across all the GPUs
        loss = torch.mean(loss)
        reduced_loss = reduce_tensor(loss)

        # print the loss and synchronize the print statements across all the GPUs
        if rank == 0:
            print(f"Loss: {reduced_loss}")
        dist.barrier()

        optimizer.step()
        scheduler.step()
