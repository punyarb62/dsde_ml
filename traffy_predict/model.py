import torch
import torch.nn as nn
from transformers import AutoModel

class TraffyBertRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)

        # BERT hidden size + 5 numeric features
        self.feature_dim = self.bert.config.hidden_size + 5  

        self.dense = nn.Linear(self.feature_dim, 64)
        self.relu = nn.ReLU()
        self.regressor = nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask,
                is_type, comment_len, is_weekend, working_hr, month):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # (B, hidden)

        x = self.dropout(cls)

        # concatenate numeric features
        x = torch.cat([
            x,
            is_type.unsqueeze(1),
            comment_len.unsqueeze(1),
            is_weekend.unsqueeze(1),
            working_hr.unsqueeze(1),
            month.unsqueeze(1)
        ], dim=1)

        x = self.relu(self.dense(x))
        return self.regressor(x)
