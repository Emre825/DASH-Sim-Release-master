import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=4, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        # Separate Q,K,V to avoid fusing and increase traffic
        self.Wq = nn.Linear(d_model, d_model, bias=True)
        self.Wk = nn.Linear(d_model, d_model, bias=True)
        self.Wv = nn.Linear(d_model, d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, L, D = x.shape  # (batch size, seq length, d_model)
        q = self.Wq(x) 
        k = self.Wk(x)
        v = self.Wv(x)

        # reshape to (B, heads, L, d_head) (d_head*heads = d_model)(transpose(1,2) swaps dimension 1 (which is L) and dimension 2 (which is heads), (B,L,heads,d_head) becomes (B,heads,L,d_head)) 
        # we do this to combine B and heads dimensions. Later, we multiply B and heads (B*heads) and the shape becomes (BatchCount (B*heads), L, d_head) for the calculation of "scores".
        # so, all of the q,k,v have the shape (256, 512, 64) finally. This is exactly what enables us to use cublasSgemmStridedBatched API.
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # explicit attention scores to force large intermediates: (B,h,L,L)
        # transpose(-2,-1) swaps the LAST TWO DIMENSIONS of k, so k becomes (BC, d_head, L), this is necessary to have our "scores" in (BC, L, L).
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        probs  = F.softmax(scores, dim=-1) # normalize "scores" in the last dimension.
        probs  = self.attn_drop(probs)
        ctx    = torch.matmul(probs, v)  # (B,h,L,d_head)

        # merge heads: (B,L,D)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, D)
        out = self.proj_drop(self.proj(ctx))
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h1 = self.fc1(x)            
        h2 = F.gelu(h1)       
        h2 = self.drop(h2)
        out = self.fc2(h2)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=1024, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.mha = MultiHeadSelfAttention(d_model, n_heads, attn_dropout=dropout, proj_dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.mha(self.ln1(x))
        x = x + self.drop(y)
        z = self.ffn(self.ln2(x))
        x = x + self.drop(z)
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size=32000, d_model=1024, n_heads=16, d_ff=4096, n_layers=6, dropout=0.0, max_len=2048):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)      
        return logits