import torch
import torch.nn as nn
from torch.nn import functional as F

MODE = "train"

checkpoint_path = "gpt_model.pt"  # fichier ou on sauvegarde / recharge le modele
output_path = "output.txt"  # fichier ou on ecrit le texte genere
num_generate_tokens = 2500  # nombre de tokens generes (printes + ecrits)

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
if torch.cuda.is_available():
    device = "cuda"  # Nvidia GPU
elif torch.backends.mps.is_available():
    device = "mps"  # Mac M1/M2/M3
else:
    device = "cpu"  # fallback
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2


MOVES = [
    "U",
    "U'",
    "U2",
    "D",
    "D'",
    "D2",
    "L",
    "L'",
    "L2",
    "R",
    "R'",
    "R2",
    "F",
    "F'",
    "F2",
    "B",
    "B'",
    "B2",
]
SPECIAL = ["<PAD>", "<SEP>", "<E>"]
vocab = SPECIAL + MOVES  # <PAD>=0, <SEP>=1, <E>=2, puis les coups

stoi = {tok: i for i, tok in enumerate(vocab)}
itos = {i: tok for i, tok in enumerate(vocab)}
vocab_size = len(vocab)  # 21

PAD_ID = stoi["<PAD>"]
SEP_ID = stoi["<SEP>"]
EOS_ID = stoi["<E>"]


def encode(tokens):
    return [stoi[t] for t in tokens]


def decode(ids):
    return [itos[i] for i in ids]


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left, right = line.split("<SEP>")
            scramble = left.split()
            solution = right.replace("<E>", "").split()
            pairs.append((scramble, solution))
    return pairs


def build_dataset(pairs):
    X, Y = [], []
    for scramble, solution in pairs:
        seq = scramble + ["<SEP>"] + solution + ["<E>"]
        ids = encode(seq)
        if len(ids) > block_size:
            continue
        x = ids[:-1]
        y = ids[1:]
        sep_i = len(scramble)  # position du <SEP> dans x
        y = [(-100 if i < sep_i else y[i]) for i in range(len(y))]
        pad = block_size - len(x)
        x = x + [PAD_ID] * pad
        y = y + [-100] * pad
        X.append(x)
        Y.append(y)
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


def get_batch(split):
    X, Y = (train_X, train_Y) if split == "train" else (val_X, val_Y)
    ix = torch.randint(len(X), (batch_size,))
    return X[ix].to(device), Y[ix].to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # BTC @ BTC -> BTT et on divise par sqrt dk pour variance 1
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # type: ignore
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """multiple head of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # projetction pour le residual layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                n_embd, 4 * n_embd
            ),  # on mutimplie par 4 parxe qu'ils le font dans attention is all yo need
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # projetction pour le residual layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # le x cest une residual connection
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # B,T,C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # B,T, vocab_size
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, idx, max_new_token):
        for _ in range(max_new_token):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # On prend que le dernier T donc B,C
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def solve(self, scramble, max_new_token=60):
        self.eval()
        prefix = encode(scramble + ["<SEP>"])
        idx = torch.tensor([prefix], dtype=torch.long, device=device)
        generated = []
        for _ in range(max_new_token):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            next_id = torch.argmax(
                logits, dim=-1
            )  # On veut le pluis probable quand on resoud -> deterministe
            tok = itos[int(next_id.item())]
            if tok == "<E>":
                break
            generated.append(tok)
            idx = torch.cat((idx, next_id.view(1, 1)), dim=1)
        return generated


def charger_modele(path=checkpoint_path):
    """Reconstruit le GPT et recharge les poids sauvegardes (pour eval / sonde)."""
    modele = GPTLanguageModel().to(device)
    checkpoint = torch.load(path, map_location=device)
    modele.load_state_dict(checkpoint["model_state"])
    modele.eval()
    return modele


if __name__ == "__main__":
    model = GPTLanguageModel()
    m = model.to(device)

    if MODE == "generate":
        model = charger_modele()
        m = model
        print(f"Modele recharge depuis {checkpoint_path}")

    if MODE == "train":
        train_X, train_Y = build_dataset(load_pairs("train.txt"))
        val_X, val_Y = build_dataset(load_pairs("val.txt"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = estimate_loss()
                print(
                    f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
            xb, yb = get_batch("train")

            logits, loss = m(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # sauvegarde du modele (+ le vocab pour pouvoir regenerer plus tard)
        torch.save({"model_state": model.state_dict(), "vocab": vocab}, checkpoint_path)
        print(f"Modele sauvegarde dans {checkpoint_path}")
