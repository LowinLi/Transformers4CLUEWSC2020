import os
from transformers import BertModel
import torch
from torch import nn


class Roberta(nn.Module):
    __name__ = "roberta"

    def __init__(self, config, method="origin"):
        super(Roberta, self).__init__()
        if method == "origin":
            self.embeddings = BertModel.from_pretrained(
                os.path.join(config.pretrained_dir, config.pretrained_model)
            )
        else:
            print(config.model_saved_dir)
            self.embeddings = BertModel.from_pretrained(config.model_saved_dir)
        self.device = config.device

    def forward(self, x):
        input_ids, attention_mask, span1_begin, span1_end, span2_begin, span2_end = x
        embedded = self.embeddings(input_ids, attention_mask)
        batch_size = embedded[0].shape[0]
        max_length = embedded[0].shape[1]
        span1_begin_emb = torch.mul(
            torch.zeros(batch_size, max_length)
            .to(self.device)
            .scatter_(1, (span1_begin + 1).unsqueeze(1), 1),
            embedded[0].permute(2, 0, 1),
        ).sum(2)
        span1_end_emb = torch.mul(
            torch.zeros(batch_size, max_length)
            .to(self.device)
            .scatter_(1, (span1_end + 1).unsqueeze(1), 1),
            embedded[0].permute(2, 0, 1),
        ).sum(2)
        span2_begin_emb = torch.mul(
            torch.zeros(batch_size, max_length)
            .to(self.device)
            .scatter_(1, (span2_begin + 1).unsqueeze(1), 1),
            embedded[0].permute(2, 0, 1),
        ).sum(2)
        span2_end_emb = torch.mul(
            torch.zeros(batch_size, max_length)
            .to(self.device)
            .scatter_(1, (span2_end + 1).unsqueeze(1), 1),
            embedded[0].permute(2, 0, 1),
        ).sum(2)
        span1_emb = (span1_begin_emb + span1_end_emb) / 2
        span2_emb = (span2_begin_emb + span2_end_emb) / 2
        cosine = torch.cosine_similarity(span1_emb, span2_emb, dim=0)
        # output = (output + 1)/2
        output = torch.cat(tuple([1 - cosine.unsqueeze(1), cosine.unsqueeze(1)]), 1)
        return nn.functional.softmax(output, dim=1)


if __name__ == "__main__":
    pass
