#!/usr/bin/env python3
"""
bitnet_v2_nat.py  ── 1-bit-weight / 4-bit-activation BitNet-v2 + NAT
--------------------------------------------------------------------
 • 1-bit ternary weights   ( -1 / 0 / +1 )  with per-output scale α
 • 4-bit symmetric activations (-8 … +7)   via fake-quant STE
 • Low-rank rotary multi-head attn  (512 d, 12 L, rank-U 32)
 • NAT training   (mask-predict)   +   optional “--ar” switch
 • Fits 3 GB RAM  (batch 2 × accum 4 ⇒ effective 8)

CLI
  train       python bitnet_v2_nat.py train [flags]
  train --ar  python bitnet_v2_nat.py train --ar …
  infer       python bitnet_v2_nat.py infer --ckpt … --prompt …
  export_hf   python bitnet_v2_nat.py export_hf --ckpt … --out DIR
"""

import math, time, argparse, warnings
from pathlib import Path
from array   import array
from collections import Counter

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.amp        import GradScaler, autocast
from datasets         import load_dataset
from transformers     import AutoTokenizer
from tqdm             import tqdm

# ─── hyper-params ────────────────────────────────────────────────────────
D_MODEL, N_LAYER, N_HEADS, RANK_U = 512, 12, 8, 32          # 60 M params
D_FF,  BLOCK, TOKEN_LIMIT         = 4*512, 256, 20_000_000  # 20 M tokens
LR_INIT, TOK_NAME, CKPT_DIR       = 3e-4, "Qwen/Qwen3-235B-A22B", "ckpts_v2nat"

# ─── helper: fake INT-4 activation quantiser ─────────────────────────────
def quant4(x: torch.Tensor):
    # symmetric range −8 … +7  (7 = 2^(4-1)-1)
    max_abs = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    q = torch.round(x / max_abs * 7).clamp(-8, 7)
    return (q * max_abs / 7) + (x - x.detach())      # STE

# ─── helper: ternary weight quantiser  (BitNet b1.58) ────────────────────
def ternary_weight(w: torch.Tensor):
    with torch.no_grad():
        thresh = 0.05 * w.abs().mean()               # simple threshold
    sign = torch.where(w.abs() < thresh, 0, w.sign())
    alpha = w.abs().mean(dim=1, keepdim=True)        # per-output scale
    tern = sign * alpha
    return tern + (w - w.detach())                   # STE

# ─── BitLinear:  W1.58 / A4  dense layer ────────────────────────────────
class BitLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) / math.sqrt(in_f))
        self.bias   = nn.Parameter(torch.zeros(out_f)) if bias else None
    def forward(self, x):
        xq = quant4(x)
        wq = ternary_weight(self.weight)
        return F.linear(xq, wq, self.bias)

# ─── rotary + low-rank attention using BitLinear ─────────────────────────
def rotary(t,n):
    half=t.size(-1)//2
    inv = 1/(10000**(torch.arange(half, device=t.device)/half))
    ang = torch.arange(n, device=t.device)[:,None] * inv
    c,s = torch.cos(ang), torch.sin(ang)
    t1,t2 = t[...,::2], t[...,1::2]
    return torch.cat([t1*c - t2*s, t1*s + t2*c], -1)

class LowRankMHA(nn.Module):
    def __init__(s, d=D_MODEL, h=N_HEADS, r=RANK_U):
        super().__init__(); s.h,s.dk=h,d//h
        s.Wq = s.Wk = s.Wv = BitLinear(d, d)
        s.U  = nn.Parameter(torch.empty(s.dk, r)); nn.init.orthogonal_(s.U)
        s.Wo = BitLinear(d, d, bias=False)
    def forward(s,x,mask=None):
        B,N,_ = x.shape
        q = s.Wq(x).view(B,N,s.h,s.dk).transpose(1,2)
        k = s.Wk(x).view_as(q); v = s.Wv(x).view_as(q)
        q,k = rotary(q,N), rotary(k,N); q,k = q@s.U, k@s.U
        att = (q @ k.transpose(-2,-1)) / math.sqrt(s.dk)
        if mask is not None: att += mask
        y = (att.softmax(-1) @ v).transpose(1,2).reshape(B,N,-1)
        return s.Wo(y)

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.att = LowRankMHA()
        s.ln1 = nn.LayerNorm(D_MODEL); s.ln2 = nn.LayerNorm(D_MODEL)
        s.ff  = nn.Sequential(BitLinear(D_MODEL, D_FF),
                              nn.GELU(),
                              BitLinear(D_FF, D_MODEL))
    def forward(s,x,mask=None):
        x = s.ln1(x + s.att(x,mask)); return s.ln2(x + s.ff(x))

class BitNetV2(nn.Module):
    def __init__(s,vocab):
        super().__init__()
        s.e = nn.Embedding(vocab, D_MODEL)            # keep FP16 embeddings
        s.blocks = nn.ModuleList(Block() for _ in range(N_LAYER))
        s.ln = nn.LayerNorm(D_MODEL)
        s.out = BitLinear(D_MODEL, vocab, bias=False)
    def forward(s, ids, causal=False):
        mask = (torch.triu(torch.full((1,1,ids.size(1),ids.size(1)),
                                      float('-inf'), device=ids.device),1)
                if causal else None)
        x = s.e(ids)
        for blk in s.blocks: x = blk(x, mask)
        return s.out(s.ln(x))

# ─── streaming dataset (same) ────────────────────────────────────────────
def build_token_array(tok, limit=TOKEN_LIMIT):
    print("→ streaming WikiText-103 + WikiText-2 …")
    it103 = iter(load_dataset("wikitext","wikitext-103-raw-v1",split="train",streaming=True))
    it2   = iter(load_dataset("wikitext","wikitext-2-raw-v1" ,split="train",streaming=True))
    ids=array('I'); src=[("103",it103),("2",it2)]
    bar=tqdm(total=limit,unit="tok",smoothing=0)
    while len(ids)<limit and any(it for _,it in src):
        for tag,it in src:
            if it is None: continue
            try: new=tok.encode(next(it)["text"]); ids.extend(new); bar.update(len(new))
            except StopIteration: print(f"  {tag} exhausted"); it=None
            for i,(tg,_) in enumerate(src):
                if tg==tag: src[i]=(tg,it)
            if len(ids)>=limit: break
    bar.close(); print(f"  collected {len(ids):,} tokens"); return ids

class Blocks(IterableDataset):
    def __init__(s,ids): s.ids=ids
    def __iter__(s):
        for i in range(0,len(s.ids)-BLOCK,BLOCK):
            yield torch.tensor(s.ids[i:i+BLOCK]),torch.tensor(s.ids[i+1:i+BLOCK+1])
    def __len__(s): return (len(s.ids)-BLOCK)//BLOCK

# ─── NAT helpers (unchanged) ─────────────────────────────────────────────
def guards(logits,hist,ngram=2,pen=1.7):
    if pen>1 and hist:
        t,f=zip(*Counter(hist).items())
        logits[0,list(t)]/=torch.tensor([(1+v)**pen for v in f],device=logits.device)
    if len(hist)>=ngram-1:
        key=tuple(hist[-(ngram-1):])
        banned=[hist[i+ngram-1] for i in range(len(hist)-ngram+1)
                if tuple(hist[i:i+ngram-1])==key]
        logits[0,banned]=float('-inf')
def sample_row(logits,temp,top_k,top_p):
    logits/=temp
    if top_k:
        kth=torch.topk(logits,top_k,dim=-1).values[...,-1,None]
        logits[logits<kth]=float('-inf')
    if top_p<1:
        srt,idx=torch.sort(logits,descending=True,dim=-1)
        cdf=torch.cumsum(torch.softmax(srt,-1),-1)
        mask=cdf>top_p; mask[...,0]=False; srt[mask]=float('-inf')
        logits.scatter_(-1,idx,srt)
    return torch.multinomial(torch.softmax(logits,-1),1).squeeze(-1)
def nat_generate(model,tok,prompt,max_new,temp,top_k,top_p):
    pad=tok.pad_token_id or tok.eos_token_id
    src=torch.tensor([tok.encode(prompt)],dtype=torch.long)
    tgt=torch.full((1,max_new),pad,dtype=torch.long)
    ids=torch.cat([src,tgt],1); hist=ids[0,:-max_new].tolist()
    with torch.no_grad():
        logits=model(ids)[:, -max_new:]; row=logits.clone(); guards(row,hist)
        ids[0,-max_new:]=sample_row(row,temp,top_k,top_p); hist+=ids[0,-max_new:].tolist()
        for _ in range(3):
            logits=model(ids)[:, -max_new:]; conf=torch.softmax(logits,-1).max(-1).values
            low=(conf<conf.mean()+0.6*conf.std()).nonzero().flatten()
            if not low.numel(): break
            row=logits[0,low].clone(); guards(row,hist)
            res=sample_row(row,temp,top_k,top_p); ids[0,-max_new:][low]=res
            for i,pos in enumerate(low): hist[-max_new+pos]=res[i].item()
    return tok.decode(ids[0])

# ─── CLI ────────────────────────────────────────────────────────────────
def cli():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    tr=sub.add_parser("train")
    tr.add_argument("--epochs",type=int,default=180)
    tr.add_argument("--batch", type=int,default=2)
    tr.add_argument("--accum", type=int,default=4)
    tr.add_argument("--ar",action="store_true"); tr.add_argument("--resume")

    inf=sub.add_parser("infer")
    inf.add_argument("--ckpt",required=True); inf.add_argument("--prompt",required=True)
    inf.add_argument("--max_new",type=int,default=120); inf.add_argument("--ar",action="store_true")
    inf.add_argument("--temp",type=float,default=0.45); inf.add_argument("--top_k",type=int,default=30)
    inf.add_argument("--top_p",type=float,default=0.9)

    ex=sub.add_parser("export_hf"); ex.add_argument("--ckpt",required=True)
    ex.add_argument("--out",default="bitnet_v2_nat"); return p.parse_args()

# ─── main ───────────────────────────────────────────────────────────────
def main():
    a=cli(); tok=AutoTokenizer.from_pretrained(TOK_NAME)
    dev="cuda" if torch.cuda.is_available() else "cpu"

    if a.cmd=="train":
        model=BitNetV2(len(tok)).to(dev)
        if a.resume: model.load_state_dict(torch.load(a.resume,map_location="cpu")["state_dict"])
        opt=torch.optim.AdamW(model.parameters(), LR_INIT, betas=(0.9,0.95))
        scaler=GradScaler(); ids=build_token_array(tok)
        loader=DataLoader(Blocks(ids), batch_size=a.batch, num_workers=0)
        mask_ratio=0.4
        for ep in range(1,a.epochs+1):
            loss_sum=0; step=0; t0=time.time()
            bar=tqdm(loader, desc=f"Epoch {ep}/{a.epochs}",unit="batch",
                     total=len(loader),dynamic_ncols=True)
            opt.zero_grad(set_to_none=True)
            for xb,yb in bar:
                step+=1; xb,yb=xb.to(dev), yb.to(dev)
                if a.ar:
                    with autocast(device_type=dev):
                        loss=F.cross_entropy(model(xb,causal=True).view(-1,len(tok)),
                                             yb.view(-1))
                else:
                    mask=(torch.rand_like(xb.float())<mask_ratio)
                    x_mask=xb.clone(); x_mask[mask]=tok.pad_token_id
                    with autocast(device_type=dev):
                        loss=F.cross_entropy(model(x_mask).view(-1,len(tok)),
                                             yb.view(-1))
                scaler.scale(loss / a.accum).backward()
                if step % a.accum == 0:
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                loss_sum+=loss.item(); bar.set_postfix(loss=f"{loss_sum/step:.4f}")
            bar.close()
            if step % a.accum: scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            mask_ratio=max(0.15,mask_ratio*0.9)
            elapsed=time.time()-t0; eta=(a.epochs-ep)*elapsed/60
            print(f"✓ Epoch {ep}  loss {loss_sum/step:.4f}  {elapsed/60:.1f} min  ETA {eta:.1f} min")
            Path(CKPT_DIR).mkdir(exist_ok=True,parents=True)
            torch.save({"state_dict":model.state_dict()},
                       f"{CKPT_DIR}/{('ar' if a.ar else 'nat')}_ep{ep:03d}.pt")

    elif a.cmd=="infer":
        model=BitNetV2(len(tok)); model.load_state_dict(torch.load(a.ckpt,map_location="cpu")["state_dict"])
        model.to(dev).eval()
        if a.ar:
            ids=torch.tensor([tok.encode(a.prompt)],dtype=torch.long,device=dev)
            with torch.no_grad():
                for _ in range(a.max_new):
                    nxt=model(ids,causal=True)[0,-1].argmax(); ids=torch.cat([ids,nxt.view(1,1)],1)
            print(tok.decode(ids[0]))
        else:
            print(nat_generate(model.cpu(),tok,a.prompt,a.max_new,a.temp,a.top_k,a.top_p))

    else:
        m=BitNetV2(len(tok)); m.load_state_dict(torch.load(a.ckpt,map_location="cpu")["state_dict"])
        m.save_pretrained(a.out); tok.save_pretrained(a.out); print("✓ exported to", a.out)

# ─────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    warnings.filterwarnings("ignore")
    main()
