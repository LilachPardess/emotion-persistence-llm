"""
Sanity check: load GPT-2 via TransformerLens locally and run one sentence
through it end to end. Run this on your Mac (not in a cloud sandbox) so it
can actually reach huggingface.co to download weights.

Usage:
    pip install torch transformer_lens
    python 00_sanity_check.py

First run downloads gpt2-medium (~1.5GB) - that part needs internet and will
be the slowest step you'll see. Everything after is local.
"""
import time
import warnings
warnings.filterwarnings("ignore")  # silences the HookedTransformer->TransformerBridge deprecation notice

import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-medium"  # swap to "gpt2" (small) if this is too slow, or "gpt2-large" if you have headroom

def main():
    t0 = time.time()
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")
    model.eval()
    print(f"[1/3] Loaded {MODEL_NAME} in {time.time()-t0:.1f}s "
          f"(n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model})")

    sentence = "The rain kept falling on the empty street outside the station."
    tokens = model.to_tokens(sentence)

    t0 = time.time()
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
    fwd_time = time.time() - t0
    print(f"[2/3] Forward pass + activation cache in {fwd_time:.2f}s "
          f"(logits shape {tuple(logits.shape)})")

    # This is the exact access pattern the real pipeline uses to pull the
    # residual stream at a given layer - confirm it works on the real model too.
    layer = model.cfg.n_layers // 2
    resid = cache["resid_post", layer]
    print(f"      resid_post at layer {layer}: shape {tuple(resid.shape)}")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            tokens, max_new_tokens=30, temperature=0.8,
            do_sample=True, stop_at_eos=False, verbose=False,
        )
    gen_time = time.time() - t0
    gen_text = model.to_string(out[0, tokens.shape[1]:])
    print(f"[3/3] Generated 30 tokens in {gen_time:.2f}s ({gen_time/30*1000:.0f}ms/token)")
    print(f"      Sample continuation: {gen_text!r}")

    print()
    print("=" * 60)
    print(f"Per-turn budget estimate for the decay experiment "
          f"(30-token responses): ~{gen_time:.1f}s per generation")
    print(f"6 emotions x ~5 measurement turns x 3 repeats = 90 generations "
          f"-> ~{90*gen_time/60:.1f} min total compute")
    print("=" * 60)
    if gen_time > 8:
        print("That's slower than ideal for a 5-hour budget - consider "
              "switching MODEL_NAME to 'gpt2' (small) at the top of this file.")
    else:
        print("Speed looks fine for the full pipeline - no need to downgrade the model.")

if __name__ == "__main__":
    main()
