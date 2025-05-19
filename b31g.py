#!/usr/bin/env python3
"""
b31g.py  ── BitNet-v2 (60 M)  •  NAT inference  •  CPU-optimised
----------------------------------------------------------------
Fixes TorchScript crash seen on Windows by:
  • removing problem annotations inside nn.Modules
  • trying torch.jit.script first, falling back to torch.jit.trace
CLI speed flags (pick one of --jit | --trace | --compile):
  --threads 8     (use all cores)
  --passes 1      (fastest one-shot NAT)
  --jit           (TorchScript; auto-fallback to trace)
  --trace         (force trace)
  --compile       (PyTorch-2 compile; cannot combine with jit/trace)
"""

import math, time, argparse, os, random, warnings, platform
from collections import Counter

import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer

# ─── Model hyper-params ────────────────────────────────────────────────
D_MODEL, N_LAYER, N_HEADS, RANK_U = 512, 12, 8, 32
TOK_NAME                          = "Qwen/Qwen3-235B-A22B"

# ─── Quantisers ────────────────────────────────────────────────────────
def quant4(x):
    max_abs = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    q = torch.round(x / max_abs * 7).clamp(-8, 7)
    return (q * max_abs / 7) + (x - x.detach())

def ternary_weight(w):
    with torch.no_grad():
        thresh = 0.05 * w.abs().mean()
    sign  = torch.where(w.abs() < thresh, 0, w.sign())
    alpha = w.abs().mean(dim=1, keepdim=True)
    return sign * alpha + (w - w.detach())

class BitLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) / math.sqrt(in_f))
        self.bias   = nn.Parameter(torch.zeros(out_f)) if bias else None
    def forward(self, x):
        return F.linear(quant4(x), ternary_weight(self.weight), self.bias)

# ─── rotary + low-rank MHA ─────────────────────────────────────────────
def rotary(t, n):
    half = t.size(-1) // 2
    inv  = 1 / (10000 ** (torch.arange(half, device=t.device) / half))
    ang  = torch.arange(n, device=t.device)[:, None] * inv
    c, s = torch.cos(ang), torch.sin(ang)
    t1, t2 = t[..., ::2], t[..., 1::2]
    return torch.cat([t1*c - t2*s, t1*s + t2*c], -1)

class LowRankMHA(nn.Module):
    def __init__(s, d=D_MODEL, h=N_HEADS, r=RANK_U):
        super().__init__(); s.h, s.dk = h, d // h
        s.Wq = s.Wk = s.Wv = BitLinear(d, d)
        s.U  = nn.Parameter(torch.empty(s.dk, r)); nn.init.orthogonal_(s.U)
        s.Wo = BitLinear(d, d, bias=False)
    def forward(s, x, mask=None):
        B, N, _ = x.shape
        q = s.Wq(x).view(B, N, s.h, s.dk).transpose(1, 2)
        k = s.Wk(x).view_as(q); v = s.Wv(x).view_as(q)
        q, k = rotary(q, N), rotary(k, N); q, k = q @ s.U, k @ s.U
        att = (q @ k.transpose(-2, -1)) / math.sqrt(s.dk)
        if mask is not None: att += mask
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, N, -1)
        return s.Wo(y)

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.att = LowRankMHA()
        s.ln1 = nn.LayerNorm(D_MODEL); s.ln2 = nn.LayerNorm(D_MODEL)
        s.ff  = nn.Sequential(BitLinear(D_MODEL, 4*D_MODEL),
                              nn.GELU(),
                              BitLinear(4*D_MODEL, D_MODEL))
    def forward(s, x, mask=None):
        x = s.ln1(x + s.att(x, mask)); return s.ln2(x + s.ff(x))

class BitNetV2(nn.Module):
    def __init__(s, vocab):
        super().__init__()
        s.e = nn.Embedding(vocab, D_MODEL)
        s.blocks = nn.ModuleList(Block() for _ in range(N_LAYER))
        s.ln = nn.LayerNorm(D_MODEL)
        s.out = BitLinear(D_MODEL, vocab, bias=False)
    def forward(s, ids, causal=False):
        mask = None
        if causal:
            t = ids.size(1)
            mask = torch.triu(torch.full((1,1,t,t), float('-inf'),
                                         device=ids.device), 1)
        x = s.e(ids)
        for blk in s.blocks: x = blk(x, mask)
        return s.out(s.ln(x))

# ─── NAT decode helpers ────────────────────────────────────────────────
def penalties(logits, hist, ngram=3, freq_pen=2.0):
    if freq_pen>1 and hist:
        tok,freq = zip(*Counter(hist).items())
        logits[..., list(tok)] /= torch.tensor([(1+v)**freq_pen for v in freq],
                                               device=logits.device)
    if len(hist) >= ngram-1:
        key = tuple(hist[-(ngram-1):])
        banned=[hist[i+ngram-1] for i in range(len(hist)-ngram+1)
                if tuple(hist[i:i+ngram-1])==key]
        logits[..., banned] = float('-inf')

def sample(logits, temp=0.8, top_k=50, top_p=0.95):
    shape = logits.shape[:-1]
    flat  = logits.reshape(-1, logits.size(-1))
    flat /= temp
    if top_k>0:
        kth = torch.topk(flat, top_k, -1).values[:, -1, None]
        flat[flat < kth] = float('-inf')
    if top_p < 1:
        srt, idx = flat.sort(-1, True)
        cdf = srt.softmax(-1).cumsum(-1)
        mask = cdf > top_p; mask[:, 0] = False; srt[mask] = float('-inf')
        flat.scatter_(-1, idx, srt)
    out = torch.multinomial(flat.softmax(-1), 1).squeeze(-1)
    return out.view(shape)

@torch.inference_mode()
def nat_generate(model, tok, prompt, max_new, temp, top_k, top_p,
                 passes, ngram, freq):
    pad = tok.pad_token_id or tok.eos_token_id
    src = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    tgt = torch.full((1, max_new), pad, dtype=torch.long)
    ids = torch.cat([src, tgt], 1); hist = ids[0, :-max_new].tolist()

    logits = model(ids)[:, -max_new:]; row = logits.clone()
    penalties(row, hist, ngram, freq)
    ids[0, -max_new:] = sample(row, temp, top_k, top_p).view(-1)
    hist += ids[0, -max_new:].tolist()

    for _ in range(max(0, passes-1)):
        logits = model(ids)[:, -max_new:]
        conf = logits.softmax(-1).max(-1).values
        low = (conf < conf.mean() + .5*conf.std()).nonzero().flatten()
        if not low.numel(): break
        row = logits[0, low].clone(); penalties(row, hist, ngram, freq)
        res = sample(row, temp, top_k, top_p)
        ids[0, -max_new:][low] = res
        for i, pos in enumerate(low): hist[-max_new + pos] = res[i].item()
    return tok.decode(ids[0]), len(hist)

# ─── CLI ────────────────────────────────────────────────────────────────
def cli():
    p = argparse.ArgumentParser()
    inf = p.add_subparsers(dest="cmd", required=True).add_parser("infer")
    inf.add_argument("--ckpt", required=True);   inf.add_argument("--prompt", required=True)
    inf.add_argument("--max_new", type=int, default=120)
    inf.add_argument("--passes",  type=int, default=1)
    inf.add_argument("--temp",    type=float, default=0.8)
    inf.add_argument("--top_k",   type=int,   default=50)
    inf.add_argument("--top_p",   type=float, default=0.95)
    inf.add_argument("--ngram",   type=int,   default=3)
    inf.add_argument("--freq_pen",type=float, default=2.0)
    inf.add_argument("--threads", type=int,   default=None)
    inf.add_argument("--compile", action="store_true")
    jit = inf.add_mutually_exclusive_group()
    jit.add_argument("--jit",   action="store_true", help="try script → trace")
    jit.add_argument("--trace", action="store_true", help="force trace")
    inf.add_argument("--offline",action="store_true")
    inf.add_argument("--seed",  type=int, default=None)
    return p.parse_args()

# ─── Utilities ─────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)

def try_script_or_trace(model, vocab):
    """Try script; if it fails, fallback to trace on dummy input."""
    try:
        return torch.jit.script(model)
    except Exception as e:
        warnings.warn(f"Scripting failed, falling back to trace ({e})")
        class Wrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, ids): return self.m(ids, causal=False)
        dummy = torch.randint(0, vocab, (1, 8))
        return torch.jit.trace(Wrapper(model), dummy, strict=False)

# ─── main ───────────────────────────────────────────────────────────────
def main():
    a = cli()
    if a.seed is not None: set_seed(a.seed)
    if a.threads: torch.set_num_threads(a.threads)

    tok = AutoTokenizer.from_pretrained(
        TOK_NAME, local_files_only=a.offline, trust_remote_code=True
    )

    model = BitNetV2(len(tok))
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["state_dict"])
    model.eval().to(memory_format=torch.channels_last)

    if a.compile and (a.jit or a.trace):
        raise SystemExit("Pick only one of --compile OR --jit/--trace.")

    if a.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead", backend="inductor")

    if a.jit:
        model = try_script_or_trace(model, len(tok))
    elif a.trace:
        class W(nn.Module):
            def __init__(self, m): super().__init__(); self.m=m
            def forward(self, ids): return self.m(ids, causal=False)
        dummy = torch.randint(0, len(tok), (1, 8))
        model = torch.jit.trace(W(model), dummy, strict=False)

    t0 = time.time()
    text, n = nat_generate(
        model, tok, a.prompt, a.max_new,
        a.temp, a.top_k, a.top_p,
        a.passes, a.ngram, a.freq_pen
    )
    dt = time.time() - t0
    print(text)
    print(f"\n✓ {n} tokens in {dt:.2f}s  ({n/dt:.1f} tok/s)")

# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
