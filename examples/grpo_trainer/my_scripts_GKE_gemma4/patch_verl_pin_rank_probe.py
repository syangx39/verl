#!/usr/bin/env python3
"""Patch verl (pin ace775e8) FSDPEngine with an env-gated per-rank canary probe (diagnostic, off by default).

When VERL_RANK_PROBE_DIR is set, every actor rank runs one extra forward on a fixed canary sequence
(VERL_RANK_PROBE_IDS, comma-separated token IDs) and writes probe_<tag><n>_rank<RR>.json with:
  - sum / abs-sum of the outputs of embed_tokens, embed_tokens_per_layer, every decoder layer, norm, lm_head
  - the canary's next-token log-probs (sum and first 16)
  - dtype / sum / abs-sum of every buffer of the wrapped HF model
Tags: pre_sync = start of get_per_tensor_param (before weights go to the rollout),
      fwd      = start of forward_backward_batch(forward_only=True) (compute_log_prob).
At most VERL_RANK_PROBE_MAX (default 2) probes per tag per process. All ranks run the canary at the
same call sites, so FSDP collectives stay matched. Normal runs (env unset) are untouched.

Idempotent. Prints: RANK-PROBE PATCH: APPLIED | already patched | FAILED <reason>
Usage: python3 patch_verl_pin_rank_probe.py /workspace/verl-pin/verl/workers/engine/fsdp/transformer_impl.py
"""
import ast
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/verl-pin/verl/workers/engine/fsdp/transformer_impl.py"
src = open(path, encoding="utf-8").read()
MARK = "_RANK_PROBE"
if MARK in src:
    print("RANK-PROBE PATCH: already patched")
    sys.exit(0)

FWD_ANCHOR = '''    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> list[TensorDict]:
'''
SYNC_ANCHOR = '''    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
'''
for name, anchor in (("forward_backward_batch", FWD_ANCHOR), ("get_per_tensor_param", SYNC_ANCHOR)):
    if src.count(anchor) != 1:
        print(f"RANK-PROBE PATCH: FAILED {name} anchor found {src.count(anchor)} times (expected 1)")
        sys.exit(1)

METHOD = '''    def _rank_probe(self, tag):  # _RANK_PROBE
        """Canary forward with per-module output checksums and buffer checksums on every rank."""
        import json
        import os

        counts = getattr(self, "_rank_probe_count", {})
        if counts.get(tag, 0) >= int(os.environ.get("VERL_RANK_PROBE_MAX", "2")):
            return
        counts[tag] = counts.get(tag, 0) + 1
        self._rank_probe_count = counts
        out_dir = os.environ["VERL_RANK_PROBE_DIR"]
        os.makedirs(out_dir, exist_ok=True)
        ids = [int(x) for x in os.environ["VERL_RANK_PROBE_IDS"].split(",")]
        dev = get_device_id()
        inp = torch.tensor(ids, dtype=torch.long, device=dev)[None]
        pos = torch.arange(len(ids), dtype=torch.long, device=dev)[None]
        rank = torch.distributed.get_rank()
        rec = {"tag": tag, "call": counts[tag], "rank": rank, "modules": {}, "buffers": {}}
        inner = getattr(self.module, "_fsdp_wrapped_module", self.module)
        hooks, seen = [], set()
        for name, mod in self.module.named_modules():
            clean = (name.replace("_fsdp_wrapped_module.", "").replace("._fsdp_wrapped_module", "")
                     .replace("_checkpoint_wrapped_module.", "").replace("._checkpoint_wrapped_module", ""))
            parts = clean.split(".")
            is_layer = len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit() and "language_model" in clean
            want = is_layer or clean.endswith(("language_model.embed_tokens", "language_model.embed_tokens_per_layer",
                                               "language_model.norm", "lm_head"))
            if not want or clean in seen:
                continue
            seen.add(clean)

            def _hook(m, a, o, _k=clean):
                t = o[0] if isinstance(o, (tuple, list)) else o
                t = getattr(t, "last_hidden_state", t)
                if torch.is_tensor(t):
                    tf = t.detach().float()
                    rec["modules"][_k] = [tf.sum().item(), tf.abs().sum().item()]

            hooks.append(mod.register_forward_hook(_hook))
        try:
            with torch.no_grad(), torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                out = self.module(input_ids=inp, attention_mask=None, position_ids=pos, use_cache=False)
            logits = out.logits[0].float()
            lp = torch.log_softmax(logits[:-1], dim=-1).gather(-1, inp[0, 1:, None]).squeeze(-1)
            rec["logp_sum"] = lp.sum().item()
            rec["logp_head"] = [round(x, 5) for x in lp[:16].tolist()]
            del out, logits
        finally:
            for h in hooks:
                h.remove()
        for bname, b in inner.named_buffers():
            bf = b.detach().double()
            rec["buffers"][bname] = [str(b.dtype), bf.sum().item(), bf.abs().sum().item()]
        with open(os.path.join(out_dir, f"probe_{tag}{counts[tag]}_rank{rank:02d}.json"), "w") as f:
            json.dump(rec, f)
        print(f"[_RANK_PROBE] tag={tag}{counts[tag]} rank={rank} logp_sum={rec['logp_sum']:.6f} "
              f"n_modules={len(rec['modules'])} n_buffers={len(rec['buffers'])}", flush=True)

'''
FWD_INSERT = FWD_ANCHOR + '''        import os as _os

        if forward_only and _os.environ.get("VERL_RANK_PROBE_DIR"):  # _RANK_PROBE
            self._rank_probe("fwd")
'''
SYNC_INSERT = SYNC_ANCHOR + '''        import os as _os

        if _os.environ.get("VERL_RANK_PROBE_DIR"):  # _RANK_PROBE
            self._rank_probe("pre_sync")
'''
new = src.replace(FWD_ANCHOR, METHOD + FWD_INSERT, 1).replace(SYNC_ANCHOR, SYNC_INSERT, 1)
try:
    ast.parse(new)
except SyntaxError as e:
    print(f"RANK-PROBE PATCH: FAILED patched file does not parse: {e}")
    sys.exit(1)
open(path, "w", encoding="utf-8").write(new)
print("RANK-PROBE PATCH: APPLIED")
