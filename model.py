# dynasty_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicEdgeBiasAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = hidden_dim // num_heads
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.bias_mlp = nn.Sequential(
            nn.Linear(1, num_heads),
            nn.LeakyReLU(),
            nn.Linear(num_heads, num_heads)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A):
        B, N, D = x.shape
        Q = self.q_proj(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)

        bias_input = A.unsqueeze(-1)
        bias = self.bias_mlp(bias_input)
        bias = bias.permute(0, 3, 1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        scores = scores + bias

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class STTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_hidden, dropout=0.1):
        super().__init__()
        self.attn = DynamicEdgeBiasAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, A):
        attn_out = self.attn(x, A)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class Dynasty(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_heads, mlp_hidden, num_layers, hist_len, fut_len, dropout=0.1):
        super().__init__()
        self.hist_len = hist_len
        self.fut_len = fut_len
        self.input_proj = nn.Linear(in_feats, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, in_feats)

        self.temporal_layers = nn.ModuleList([
            STTransformerLayer(hidden_dim, num_heads, mlp_hidden, dropout)
            for _ in range(num_layers)
        ])

        self.time_encoding = nn.Parameter(torch.randn(hist_len, hidden_dim))
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru_dec = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def encode(self, X_hist, A_hist):
        #print(X_hist.shape)
        
        B, L, N, D = X_hist.shape
        
        X_proj = self.input_proj(X_hist) + self.time_encoding[None, :, None, :]
        x_seq = X_proj.view(B * L, N, -1)
        A_seq = A_hist.view(B * L, N, N)

        if self.training:
            mask = (torch.rand_like(A_seq) > 0.1).float()
            A_seq = A_seq * mask

        for layer in self.temporal_layers:
            x_seq = layer(x_seq, A_seq)

        return x_seq.view(B, L, N, -1)

    def decode(self, H, Y_true=None, ss_prob=0.1):
        B, L, N, D = H.shape
        H_flat = H.permute(0, 2, 1, 3).reshape(B * N, L, D)
        _, h = self.gru(H_flat)

        x_in_step = h.squeeze(0).view(B * N, 1, -1)
        outputs = []

        for t in range(self.fut_len):
            x_out_step, h = self.gru_dec(x_in_step, h)
            pred = self.output_proj(x_out_step.squeeze(1)).view(B, N, -1)
            outputs.append(pred.unsqueeze(1))

            if Y_true is not None:
                use_gt = (torch.rand(B, 1, 1, device=H.device) < ss_prob).float()
                gt_t = Y_true[:, t] if t < Y_true.shape[1] else pred
                x_in_step = self.input_proj(use_gt * gt_t + (1 - use_gt) * pred)
            else:
                x_in_step = self.input_proj(pred)

            x_in_step = x_in_step.view(B * N, 1, -1)

        return torch.cat(outputs, dim=1)

    def forward(self, X_hist, A_hist, Y_true=None, epoch=0):
        ss_prob = max(0.5 * (0.98 ** epoch), 0.1) if epoch > 0 else 0.0
        h_ctx = self.encode(X_hist, A_hist)
        return self.decode(h_ctx, Y_true, ss_prob)

