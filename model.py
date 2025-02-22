# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, n_embed, head_size, block_size):
        super().__init__()
        self.key = self.query = nn.Sequential(
            nn.Linear(n_embed, head_size, bias=False),
            nn.GELU(),
            nn.Linear(head_size, head_size, bias=False),
        )

        self.query = nn.Sequential(
            nn.Linear(n_embed, head_size, bias=False),
            nn.GELU(),
            nn.Linear(head_size, head_size, bias=False),
        )
        self.value =self.query = nn.Sequential(
            nn.Linear(n_embed, head_size, bias=False),
            nn.GELU(),
            nn.Linear(head_size, head_size, bias=False),
        )
        # Used to mask future positions in the sequence
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        # Take the value from the original x
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embed):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embed, head_size, block_size=n_embed) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embed)

    def forward(self, x):
        # Concatenate outputs from all heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = F.gelu(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embed)
        self.ff = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class LM(nn.Module):
    """
    A simple Language Model-like Transformer.
    """
    def __init__(self, vocab_size, dim=384, block_size=384, n_head=6, layers=6, device='cpu'):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.dim = dim

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(block_size, dim)
        self.blocks = nn.Sequential(
            *[Block(dim, n_head) for _ in range(layers)]
        )
        self.lm_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, vocab_size)
        )

        self.device = device

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Position embedding
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        pos_embed = self.position_embedding(pos)

        # Forward through Transformer blocks
        x = self.token_embedding(idx) + pos_embed
        x = self.blocks(x)
        logits = self.lm_head(x)

        # Compute loss
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100):
        # Generate tokens in auto-regressive fashion
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Safety check
            if idx_next.max() >= self.vocab_size:
                raise ValueError(f"Generated index {idx_next.max()} out of range")

            idx = torch.cat((idx, idx_next), dim=1)
        return idx