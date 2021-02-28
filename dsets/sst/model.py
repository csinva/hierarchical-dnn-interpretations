import torch
import torch.nn as nn


class LSTMSentiment(nn.Module):
    def __init__(self, config=None, d_hidden=128, n_embed=18844, d_embed=300, d_out=2, batch_size=50):
        super(LSTMSentiment, self).__init__()
        if config is not None:
            self.hidden_dim = config.d_hidden
            self.vocab_size = config.n_embed
            self.emb_dim = config.d_embed
            self.num_out = config.d_out
            self.batch_size = config.batch_size
        else:
            self.hidden_dim = d_hidden
            self.vocab_size = n_embed
            self.emb_dim = d_embed
            self.num_out = d_out
            self.batch_size = batch_size
        self.use_gpu = True  # config.use_gpu
        self.num_labels = 2
        self.embed = nn.Embedding(self.vocab_size, self.emb_dim)
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim)
        self.hidden_to_label = nn.Linear(self.hidden_dim, self.num_labels)

    def forward(self, batch):
        self.hidden = (torch.zeros(1, batch.text.shape[1], self.hidden_dim),
                       torch.zeros(1, batch.text.shape[1], self.hidden_dim))
        vecs = self.embed(batch.text)
        lstm_out, self.hidden = self.lstm(vecs, self.hidden)
        logits = self.hidden_to_label(lstm_out[-1])
        # log_probs = self.log_softmax(logits)
        # return log_probs
        return logits

    def predict(self, batch):
        pred = self.forward(batch)
        _, pred = pred[0].max(0)
        return pred.data[0]
