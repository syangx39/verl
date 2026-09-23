#!/usr/bin/env python3
"""Compare two per-parameter gradient files (safetensors, as written by replay_single_step.py --save_grad): global and
per-parameter cosine / relative error with chunked float64 accumulation. Usage: compare_grads.py A.safetensors B.safetensors"""
import math, sys
from safetensors.torch import load_file


def pair_stats(a, b, chunk=1 << 23):
  fa, fb = a.reshape(-1), b.reshape(-1); sa = sb = sab = sd = 0.0
  for i in range(0, fa.numel(), chunk):
    x = fa[i:i + chunk].double(); y = fb[i:i + chunk].double()
    sa += float((x * x).sum()); sb += float((y * y).sum()); sab += float((x * y).sum()); sd += float(((x - y) ** 2).sum())
  return sa, sb, sab, sd


A, B = load_file(sys.argv[1]), load_file(sys.argv[2])
if set(A) != set(B):
  raise SystemExit(f"parameter sets differ: {set(A) ^ set(B)}")
num = den = dot = sq_b = 0.0; rows = []
for n in A:
  assert A[n].shape == B[n].shape, (n, tuple(A[n].shape), tuple(B[n].shape))   # never compare flattened tensors of different shapes
  sa, sb, sab, sd = pair_stats(A[n], B[n]); num += sd; den += sa; dot += sab; sq_b += sb
  if sa > 0 and sb > 0:
    rows.append((n, sab / math.sqrt(sa * sb), math.sqrt(sd / sa)))
cos = dot / math.sqrt(den * sq_b); rel = math.sqrt(num / den)
print(f"gradient A vs B: ||A|| {math.sqrt(den):.6e} ||B|| {math.sqrt(sq_b):.6e} | cosine {cos:.6f} | rel err {rel:.3e}")
for n, c, e in sorted(rows, key=lambda r: r[1])[:5]:
  print(f"    lowest: {n:<52} cos {c:.5f} rel_err {e:.3e}")
