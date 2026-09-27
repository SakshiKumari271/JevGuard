import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class JevGuardClassifier(nn.Module):

    def __init__(
        self, model_name: str = "microsoft/deberta-v3-small", num_labels: int = 3
    ):
        super(JevGuardClassifier, self).__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        # Backbone ko explicitly float32 mein load karenge dtype mismatch rokne ke liye
        self.encoder = AutoModel.from_pretrained(
            model_name, config=self.config, torch_dtype=torch.float32
        )

        hidden_size = self.config.hidden_size  # 768
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # [CLS] token representation
        cls_rep = outputs.last_hidden_state[:, 0, :]

        # Explicitly ensure float32 for classification head
        cls_rep = cls_rep.to(torch.float32)

        x = self.dropout(cls_rep)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)

        logits = self.classifier(x)
        return logits