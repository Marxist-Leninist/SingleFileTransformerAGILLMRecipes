# SingleFileTransformerAGILLMRecipes
Single File python file that run end to end transformer training, dataset and inferencing all in one file you can run it using python x.py x being the file name; most of these are usually coded using top AGI LLM at the time ie currently ChatGPT o3

b.py is bitnet_v2_nat.py  ── 1-bit-weight / 4-bit-activation BitNet-v2 + NAT
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
will upload more in time pretrained checkpoints i will link via cloud at some point
Any donation or investment in doing this just dm and ask what you propose or want to send.
Vast.ai and Hetzner cloud are proably best places to train these models for cheapest prices
