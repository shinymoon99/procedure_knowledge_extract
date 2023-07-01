import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput


class Bert_GRU(nn.Module):
    def __init__(self, bert, hidden_size, num_labels):
        super(Bert_GRU, self).__init__()
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

class Bert_SRL(nn.Module):
    def __init__(self, bert, hidden_size, num_labels):
        super(Bert_SRL, self).__init__()
        self.bert = bert
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        #self.lstm = nn.LSTM(bert.config.hidden_size, hidden_size, bidirectional=True, batch_first=True)
        #self.gru = nn.GRU(input_size=bert.config.hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(bert.config.hidden_dropout_prob)

    def forward(self, input_ids, spans,attention_mask=None, attention_mask_gru=None,token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        bert_output = outputs.last_hidden_state
        sequence_output = self.dropout(bert_output)
        logits = self.fc(sequence_output)
        return logits
import torch
from transformers import BertForTokenClassification


import torch
from transformers import BertForTokenClassification

class WeightedBertForTokenClassification(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        weight=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if weight is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()

            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

