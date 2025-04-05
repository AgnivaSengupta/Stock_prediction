import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, ticker_vocab_size, ticker_embed_size):
        super().__init__()
        self.ticker_embedding = nn.Embedding(ticker_vocab_size, ticker_embed_size)
        self.lstm = nn.LSTM(input_size + ticker_embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, ticker_id):
        batch_size, seq_len, _ = x.size()
        embed = self.ticker_embedding(ticker_id)
        embed = embed.unsqueeze(1).repeat(1, seq_len, 1)
        x = torch.cat([x, embed], dim=2)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()
