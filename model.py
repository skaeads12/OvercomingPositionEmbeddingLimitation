
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, RobertaModel

class BertForQuestionAnswering(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def set_position_embeddings(self, target_length):

        if self.bert.config.max_position_embeddings == target_length:
            return

        source_length = self.bert.config.max_position_embeddings

        position_embeddings = self.bert.embeddings.position_embeddings
        new_position_embeddings = nn.Embedding(target_length, self.bert.config.hidden_size)

        with torch.no_grad():

            for position in range(source_length - 1):
                new_position_embeddings.weight[position * 2] = position_embeddings.weight[position]
                new_position_embeddings.weight[position * 2 + 1] = (position_embeddings.weight[position] + position_embeddings.weight[position + 1]) / 2

            new_position_embeddings.weight[-2] = position_embeddings.weight[-1]
            new_position_embeddings.weight[-1] = (position_embeddings.weight[-1] + position_embeddings.weight[0]) / 2

            self.bert.embeddings.position_embeddings = new_position_embeddings

        self.bert.config.max_position_embeddings = target_length
        self.bert.embeddings.register_buffer("position_ids", torch.arange(self.bert.config.max_position_embeddings).expand((1, -1)))
        self.bert.embeddings.register_buffer("token_type_ids", torch.zeros(self.bert.embeddings.position_ids.size(), dtype=torch.long), persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        start_positions: torch.Tensor = None,
        end_positions: torch.Tensor = None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output

class RobertaForQuestionAnswering(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def set_position_embeddings(self, target_length):

        if self.roberta.config.max_position_embeddings == target_length:
            return

        source_length = self.roberta.config.max_position_embeddings

        position_embeddings = self.roberta.embeddings.position_embeddings
        new_position_embeddings = nn.Embedding(target_length, self.roberta.config.hidden_size)

        with torch.no_grad():

            for position in range(source_length - 1):
                new_position_embeddings.weight[position * 2] = position_embeddings.weight[position]
                new_position_embeddings.weight[position * 2 + 1] = (position_embeddings.weight[position] + position_embeddings.weight[position + 1]) / 2

            # new_position_embeddings.weight[-2] = position_embeddings.weight[-1]
            # new_position_embeddings.weight[-1] = (position_embeddings.weight[-1] + position_embeddings.weight[0]) / 2

            self.roberta.embeddings.position_embeddings = new_position_embeddings

        self.roberta.config.max_position_embeddings = target_length
        self.roberta.embeddings.register_buffer("position_ids", torch.arange(self.roberta.config.max_position_embeddings).expand((1, -1)))
        self.roberta.embeddings.register_buffer("token_type_ids", torch.zeros(self.roberta.embeddings.position_ids.size(), dtype=torch.long), persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        start_positions: torch.Tensor = None,
        end_positions: torch.Tensor = None,
    ):
        outputs = self.roberta(
            input_ids,
            # attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output

if __name__=="__main__":

    model = RobertaForQuestionAnswering.from_pretrained("klue/roberta-base")
    model.set_position_embeddings(1024)

    input_ids = torch.randint(0, 30000, (32, 1024))
    start_positions = torch.randint(0, 1024, (32,))
    end_positions = torch.randint(0, 1024, (32,))

    output = model(input_ids, start_positions=start_positions, end_positions=end_positions)

    print(output[0])
    print(model.roberta.embeddings.position_embeddings.weight)
