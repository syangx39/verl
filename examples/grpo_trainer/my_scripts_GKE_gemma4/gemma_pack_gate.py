#!/usr/bin/env python3
# =============================================================================
# gemma_pack_gate.py -- packed-vs-unpacked correctness gate (single GPU).
#
# Question it answers: when verl's remove-padding path packs many sequences
# into one row (position_ids restart per sequence, attention_mask=None) and
# the attention impl is sdpa/flex instead of flash-varlen, does HF's
# packed-sequence masking really keep sequences from seeing each other?
#
# Method (three forwards, no verl involved):
#   [ref]     sequence B alone                          -> logits_B_ref
#   [pack1]   row = [A ; B], per-seq position_ids       -> logits_B_p1
#   [pack2]   row = [A'; B], A' = shuffled A (same len) -> logits_B_p2
#
# Verdicts:
#   LEAKAGE  = max|logits_B_p2 - logits_B_p1|   (same shapes, only the
#              neighbor's CONTENT differs; if masking is correct this is ~0,
#              if B can see its neighbor this is O(1))
#   NOISE    = max|logits_B_p1 - logits_B_ref|  (benign shape/batching noise;
#              bf16 kernel tiling differences, expected small)
# =============================================================================
import argparse
import torch


def load_model(path: str, attn: str):
    kw = dict(torch_dtype=torch.bfloat16, attn_implementation=attn)
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(path, **kw)
    except Exception as e:
        print(f"[load] AutoModelForCausalLM failed ({type(e).__name__}); trying AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(path, **kw)
    return model.eval().cuda()


@torch.no_grad()
def fwd(model, ids: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    out = model(
        input_ids=ids.unsqueeze(0).cuda(),
        position_ids=pos.unsqueeze(0).cuda(),
        attention_mask=None,
        use_cache=False,
    )
    return out.logits[0].float().cpu()


def make_tokens(tok, approx_len: int, seed_text: str) -> torch.Tensor:
    text = (seed_text + " ") * (approx_len // 8 + 2)
    ids = tok(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    return ids[:approx_len]


def stats(x: torch.Tensor, y: torch.Tensor):
    d = (x - y).abs()
    top1 = (x.argmax(-1) == y.argmax(-1)).float().mean().item()
    return d.max().item(), d.mean().item(), top1


def run_round(model, tok, len_a: int, len_b: int, label: str):
    g = torch.Generator().manual_seed(1234)
    ids_a = make_tokens(tok, len_a, "The integral of x squared from zero to three equals nine.")
    ids_b = make_tokens(tok, len_b, "Let p be a prime number greater than five and consider p modulo six.")
    ids_a2 = ids_a[torch.randperm(len(ids_a), generator=g)]  # same length, different content

    pos_a = torch.arange(len(ids_a))
    pos_b = torch.arange(len(ids_b))

    logits_b_ref = fwd(model, ids_b, pos_b)
    packed1 = fwd(model, torch.cat([ids_a, ids_b]), torch.cat([pos_a, pos_b]))
    packed2 = fwd(model, torch.cat([ids_a2, ids_b]), torch.cat([pos_a, pos_b]))

    b1 = packed1[len(ids_a):]
    b2 = packed2[len(ids_a):]

    leak_max, leak_mean, leak_top1 = stats(b2, b1)
    noise_max, noise_mean, noise_top1 = stats(b1, logits_b_ref)

    print(f"\n===== round: {label} (A={len(ids_a)} tok, B={len(ids_b)} tok) =====")
    print(f"LEAKAGE  (neighbor-content swap, same shapes): max={leak_max:.4f}  mean={leak_mean:.6f}  top1-agree={leak_top1:.4f}")
    print(f"NOISE    (packed vs unpacked reference)      : max={noise_max:.4f}  mean={noise_mean:.6f}  top1-agree={noise_top1:.4f}")

    if leak_max < 1e-3:
        print("VERDICT: NO LEAKAGE -- packing is correctly isolated in this round.")
    elif leak_max < 0.05:
        print("VERDICT: borderline -- tiny neighbor sensitivity; rerun and inspect before trusting packing.")
    else:
        print("VERDICT: LEAKAGE CONFIRMED -- B's logits depend on its neighbor's content. "
              "remove-padding is NOT safe under this attention impl.")
    return leak_max, noise_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "flex_attention", "eager"])
    args = ap.parse_args()

    import transformers
    print(f"transformers {transformers.__version__}, torch {torch.__version__}, attn={args.attn}")
    try:
        from transformers.masking_utils import find_packed_sequence_indices  # noqa: F401
        print("packed-seq masking helper: AVAILABLE")
    except ImportError:
        print("packed-seq masking helper: MISSING -- expect leakage; do not use remove_padding with sdpa/flex")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, args.attn)

    r1 = run_round(model, tok, len_a=96, len_b=80, label="short (within sliding window)")
    r2 = run_round(model, tok, len_a=640, len_b=560, label="long (crosses sliding window)")

    print("\n===== overall =====")
    if max(r1[0], r2[0]) < 1e-3:
        print("PASS: keep use_remove_padding=True with this attention impl; "
              "the 80x grad_norm is then a model property, not a packing bug.")
    else:
        print("FAIL: packing leaks under this attention impl; run with REMOVE_PADDING=False "
              "(and solve its memory cost separately) or move to an impl whose masking passes this gate.")


if __name__ == "__main__":
    main()
