import torch
import torch.nn as nn
import torch.nn.functional as F

class RNAPredictor(nn.Module):
    def __init__(self, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(5, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv2d(hidden_dim * 4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        B, L = x.shape
        emb = self.embedding(x)
        out, _ = self.lstm(emb) # [B, L, 2H]
        
        out1 = out.unsqueeze(2).expand(-1, -1, L, -1)
        out2 = out.unsqueeze(1).expand(-1, L, -1, -1)
        concat = torch.cat([out1, out2], dim=-1) # [B, L, L, 4H]
        concat = concat.permute(0, 3, 1, 2) # [B, 4H, L, L]
        
        x2d = F.relu(self.conv1(concat))
        x2d = self.conv2(x2d) # [B, 1, L, L]
        
        x2d = (x2d + x2d.transpose(2, 3)) / 2
        return x2d.squeeze(1) # [B, L, L]
