#!/usr/bin/env bash
# Assemble the GB200 reference package for recipe gsm8k_2k_async1 (disaggregated, asynchronous GRPO; 32 trainer / 32 rollout
# GB200; Qwen3-0.6B; GSM8K; 2K cap + overlong penalty; 3 seeds x 250 steps) into $H and self-verify it.
# Run on the head pod after `source /workspace/setup_env.sh` (needs RECIPE_DIR, LOG_DIR, CKPT_DIR, MODEL_PATH, DATA_DIR, DISAGG_PYTHON,
# DISAGG_IMAGE, VERL_REPO). Reruns are idempotent. Env knobs: RUN_GLOB (default disagg_t32_r32_dyn_seed), SKIP_CKPT=1, SKIP_DUMPS=1.
set -euo pipefail
: "${RECIPE_DIR:?}" "${LOG_DIR:?}" "${CKPT_DIR:?}" "${MODEL_PATH:?}" "${DATA_DIR:?}" "${DISAGG_PYTHON:?}" "${VERL_REPO:?}"
H=${H:-/workspace/meta-RL/handoff/gsm8k_2k_async1}
G=${G:-/workspace/meta-RL/verl/examples/grpo_trainer/my_scripts_GKE_gsm8k}          # Meta-track tools: band_plot.py, builder, 8K package fixtures
PKG8K=${PKG8K:-/workspace/meta-RL/handoff/gsm8k_8k}
RUN_GLOB=${RUN_GLOB:-disagg_t32_r32_dyn_seed}
TB2K=${TB2K:-/workspace/meta-RL/.home/tensorboard_log/meta_gsm8k_boxed}                 # colocated 2K references (context figures only)
PY=$DISAGG_PYTHON
PIN=$(cat "$RECIPE_DIR/VERL_PIN.txt"); IMG=$(cat "$RECIPE_DIR/IMAGE_REF.txt")
test "$(git -C "$VERL_REPO" rev-parse HEAD)" = "$PIN" || { echo "image checkout $(git -C "$VERL_REPO" rev-parse HEAD) != VERL_PIN.txt $PIN"; exit 2; }
mkdir -p $H/{model,data,fixtures,code,env,runs,band,checkpoints}

# ---------- 0. the three reference runs (latest directory per seed by name) ----------
declare -A E R
for S in 1 2 3; do
  d=$(ls -d $LOG_DIR/${RUN_GLOB}${S}_*/ 2>/dev/null | sort | tail -1); test -n "$d" || { echo "no run dir for seed $S ($LOG_DIR/${RUN_GLOB}${S}_*)"; exit 2; }
  R[$S]=${d%/}; E[$S]=$(basename ${d%/})
  test "$(cat ${R[$S]}/exit_code.txt)" = "0" || { echo "seed $S exit_code != 0"; exit 2; }
  test -f ${R[$S]}/resolved_config.yaml && test -f ${R[$S]}/preflight.json && test -d ${R[$S]}/tensorboard || { echo "seed $S run dir incomplete"; exit 2; }
  test "$(ls ${R[$S]}/val_dump | wc -l)" = "14" && test "$(ls ${R[$S]}/rollout_dump | wc -l)" = "250" || { echo "seed $S: expected 14 val dumps and 250 rollout dumps"; exit 2; }
  echo "seed $S -> ${E[$S]}"
done
$PY - "${R[1]}" "${R[2]}" "${R[3]}" <<'PYX'
import sys, re
skip = re.compile(r"seed|_dir|experiment_name|EXPERIMENT_NAME|RUN_DIR|TB_DIR|TENSORBOARD_DIR|PYTHONHASHSEED|SEED")
cfgs = [[l for l in open(r + "/resolved_config.yaml") if not skip.search(l)] for r in sys.argv[1:]]
assert cfgs[0] == cfgs[1] == cfgs[2], "resolved configs differ beyond seed/paths"
print("resolved configs identical except seed/paths")
PYX

# ---------- 1. model ----------
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt model.safetensors; do cp $MODEL_PATH/$f $H/model/; done
( cd $H/model && sha256sum model.safetensors config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt ) > $H/model/MODEL_SHA256
$PY - "$MODEL_PATH" > $H/model/model_identity.json <<'PYX'
import json, sys, torch
from safetensors.torch import load_file
w = load_file(sys.argv[1] + "/model.safetensors")
print(json.dumps({"n_tensors": len(w), "lm_head_untied": "lm_head.weight" in w,
                  "sum_model_norm_weight": float(w["model.norm.weight"].float().sum()),
                  "eos_token_id": json.load(open(sys.argv[1] + "/generation_config.json"))["eos_token_id"]}, indent=1))
PYX

# ---------- 2. data + per-step prompt order recovered from the V1 rollout dumps ----------
cp $DATA_DIR/gsm8k_boxed_train.parquet $DATA_DIR/gsm8k_boxed_test.parquet $H/data/
[ -d $DATA_DIR/meta_reference ] && cp -r $DATA_DIR/meta_reference $H/data/ || true
$PY - "$H/data" "${R[1]}" "${R[2]}" "${R[3]}" <<'PYX'
import sys, json, glob, os, re, pandas as pd
out = sys.argv[1]; tr = pd.read_parquet(out + "/gsm8k_boxed_train.parquet"); te = pd.read_parquet(out + "/gsm8k_boxed_test.parquet")
q_of = lambda df: [m[-1]["content"] for m in df["prompt"]]
trq, teq = q_of(tr), q_of(te)
json.dump({"train_rows": len(tr), "test_rows": len(te), "train_unique_questions": len(set(trq)), "test_unique_questions": len(set(teq)),
           "train_test_question_overlap": len(set(trq) & set(teq))}, open(out + "/DATA_COUNTS.json", "w"), indent=1)
# V1 dumps carry the rendered prompt text ('input'); match it back to the training row by the question text it contains
idx_of = {}
for i, q in enumerate(trq): idx_of.setdefault(q, i)
for s, r in enumerate(sys.argv[2:], 1):
    manifest, unmatched = {}, 0
    for step in range(1, 251):
        rows = [json.loads(l) for l in open(f"{r}/rollout_dump/{step}.jsonl")]
        groups, sizes = {}, {}
        for row in rows:
            g = row["uid"].rsplit("_", 2)[0]; groups.setdefault(g, row["input"]); sizes[g] = sizes.get(g, 0) + 1
        assert len(groups) == 128 and set(sizes.values()) == {16}, f"seed {s} step {step}: {len(groups)} groups, group sizes {set(sizes.values())}"
        ids = []
        for uid, inp in groups.items():
            hit = [i for q, i in idx_of.items() if q and q in inp]      # exact substring; question texts are unique
            if len(hit) == 1: ids.append(hit[0])
            else: unmatched += 1
        manifest[step] = sorted(ids)
    json.dump({"note": "training-row indices of the 128 prompt groups consumed at each optimizer step (asynchronous: membership follows completion order, order within a step irrelevant)",
               "unmatched_groups": unmatched, "steps": manifest}, open(f"{out}/step_manifest_seed{s}.json", "w"))
    n = sum(len(v) for v in manifest.values()); print(f"seed {s}: {n} prompt groups over 250 steps matched to train rows, {unmatched} unmatched, {len(set(i for v in manifest.values() for i in v))} distinct questions")
    assert unmatched == 0 and n == 250 * 128, f"seed {s}: manifest incomplete ({n} matched, {unmatched} unmatched)"
PYX

# ---------- 3. code + environment ----------
cp $RECIPE_DIR/*.py $RECIPE_DIR/*.sh $RECIPE_DIR/*.yaml $RECIPE_DIR/*.md $RECIPE_DIR/Dockerfile $RECIPE_DIR/VERL_PIN.txt $RECIPE_DIR/IMAGE_REF.txt $H/code/ 2>/dev/null || true
cp $G/band_plot.py $H/code/ 2>/dev/null || true
[ -f /workspace/meta-RL/verl/examples/grpo_trainer/my_scripts_GKE/verl-qwen3-raycluster.yaml ] && cp /workspace/meta-RL/verl/examples/grpo_trainer/my_scripts_GKE/verl-qwen3-raycluster.yaml $H/env/raycluster.yaml
echo "$IMG" > $H/env/IMAGE_REF.txt; echo "$PIN" > $H/env/VERL_PIN.txt; cp $VERL_REPO/uv.lock $H/env/uv.lock
uv pip freeze --python $PY > $H/env/pip_freeze.txt 2>/dev/null || $PY -m pip freeze > $H/env/pip_freeze.txt
$PY -c "import sys,torch,vllm,transformers,ray,flash_attn; print(f'python {sys.version.split()[0]}\ntorch {torch.__version__} cuda {torch.version.cuda}\nvllm {vllm.__version__}\ntransformers {transformers.__version__}\nray {ray.__version__}\nflash_attn {flash_attn.__version__}')" > $H/env/versions.txt
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1 > $H/env/gpu.txt 2>/dev/null || true
for S in 1 2 3; do cp ${R[$S]}/resolved_config.yaml $H/env/resolved_config_seed$S.yaml; cp ${R[$S]}/preflight.json $H/env/preflight_seed$S.json; cp ${R[$S]}/environment_manifest.txt $H/env/environment_manifest_seed$S.txt 2>/dev/null || true; done
$PY - $H/env/resolved_config_seed{1,2,3}.yaml > $H/env/CONFIG_DIFF.txt <<'PYX'
import sys, yaml
def flat(d, p=""):
    out = {}
    for k, v in (d or {}).items():
        out.update(flat(v, f"{p}{k}.") if isinstance(v, dict) else {f"{p}{k}": v})
    return out
cs = [flat(yaml.safe_load(open(f))) for f in sys.argv[1:]]
keys = sorted(k for k in set().union(*cs) if len({str(c.get(k)) for c in cs}) > 1)
print("keys differing across the three seeds:", keys)
PYX

# ---------- 4. fixtures: gates 1-2 (reused from the 8K package when present) + the 2K scorer fixture from real responses ----------
for f in prompt_fixture.json meta_reward_fixtures.json reward_selftest_meta_rule.log; do
  test -f $PKG8K/fixtures/$f || { echo "missing $PKG8K/fixtures/$f: gates 1-2 reuse the 8K package's fixtures (same data build, same scorer); restore that package first"; exit 2; }
  cp $PKG8K/fixtures/$f $H/fixtures/
done
echo "prompt_fixture.json / meta_reward_fixtures.json reused from $PKG8K"
$PY - "${R[1]}" "$H/fixtures" "$RECIPE_DIR" <<'PYX'
import sys, json, os, random, importlib.util
r, out, rd = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("bmr", rd + "/boxed_math_reward.py"); bmr = importlib.util.module_from_spec(spec); spec.loader.exec_module(bmr)
rows = []
for step in (0, 250):
    for l in open(f"{r}/val_dump/{step}.jsonl"):
        d = json.loads(l); rows.append({"step": step, "response": d["output"], "ground_truth": d["gts"], "expected_score": float(d["score"])})
random.Random(0).shuffle(rows); rows = sorted(rows[:800], key=lambda x: x["step"])
bad = 0
with open(out + "/scorer_fixture_2k.jsonl", "w") as f:
    for x in rows:
        s = bmr.compute_score("gsm8k_boxed_test", x["response"], x["ground_truth"], extra_info={"index": 0, "response_len": len(x["response"])})   # eval source: no penalty
        s = s["score"] if isinstance(s, dict) else float(s)
        bad += abs(s - x["expected_score"]) > 1e-6; x["recomputed"] = s; f.write(json.dumps(x) + "\n")
assert bad == 0, f"{bad} rows disagree with the dumped eval score"
print(f"scorer_fixture_2k.jsonl: {len(rows)} eval responses (steps 0/250), recomputed score == dumped score for all; acc mean {sum(x['expected_score']>=1 for x in rows)/len(rows):.4f}")
# training-mode penalty rule on synthetic lengths: penalty = min(0, -(L - (2048-512)) / 512) -- no lower bound; score = raw + penalty
os.environ.update(REWARD_FORMAT_SCORE="0.1", REWARD_OVERLONG_BUFFER="512", REWARD_OVERLONG_PENALTY="1.0", REWARD_MAX_RESP_LEN="2048", REWARD_PENALTY_SOURCES="gsm8k_boxed_train")
spec.loader.exec_module(bmr)
with open(out + "/reward_selftest_2k_penalty.log", "w") as f:
    for L in (100, 1536, 1600, 1792, 2047, 2048, 2560):
        s = bmr.compute_score("gsm8k_boxed_train", "\\boxed{42}", "42", extra_info={"index": 0, "response_len": L})
        s = s["score"] if isinstance(s, dict) else float(s); exp = 1.0 + min(0.0, -(L - 1536) / 512)
        f.write(f"correct answer, response_len={L}: score {s:.4f} expected {exp:.4f}\n"); assert abs(s - exp) < 1e-6
    ev = bmr.compute_score("gsm8k_boxed_test", "\\boxed{42}", "42", extra_info={"index": 0, "response_len": 2560}); ev = ev["score"] if isinstance(ev, dict) else float(ev)
    assert abs(ev - 1.0) < 1e-6; f.write("eval source, response_len=2560: score 1.0000 (no penalty)\n")
    f.write("penalty rule OK: 0 up to 1536 tokens, -1.0 at 2048, no lower bound (2560 -> -2.0); training source only\n")
print(open(out + "/reward_selftest_2k_penalty.log").read().strip().splitlines()[-1])
PYX

# ---------- 5. runs ----------
for S in 1 2 3; do
  D=$H/runs/seed$S; mkdir -p $D
  for f in driver.log command.txt resolved_config.yaml preflight.json start_epoch.txt end_epoch.txt exit_code.txt; do cp ${R[$S]}/$f $D/; done
  cp ${R[$S]}/environment_manifest.txt $D/ 2>/dev/null || true
  # the post-run checks must pass (re-run when the JSON is missing or did not pass); a failure aborts the package
  $PY $RECIPE_DIR/check_smoke.py ${R[$S]} --steps 250 --max-worst-lag 1 > $D/check_smoke.log 2>&1 || { echo "seed $S: check_smoke FAILED (see $D/check_smoke.log)"; exit 2; }
  $PY -c "import json,sys; j=json.load(open('${R[$S]}/check_smoke.json')); assert j.get('status') == 'passed', j.get('status'); print('seed $S check_smoke:', j['status'])"
  cp ${R[$S]}/check_smoke.json $D/
  rm -rf $D/tensorboard; cp -r ${R[$S]}/tensorboard $D/tensorboard; echo "${E[$S]}" > $D/EXPERIMENT_NAME
  if [ "${SKIP_DUMPS:-0}" = "1" ] && [ -d $D/val_dump ] && [ -d $D/rollout_dump ]; then echo "seed $S: keeping existing dumps"; else
    rm -rf $D/val_dump $D/rollout_dump; cp -r ${R[$S]}/val_dump $D/val_dump; cp -r ${R[$S]}/rollout_dump $D/rollout_dump; fi
  test "$(ls $D/val_dump | wc -l)" = "14" && test "$(ls $D/rollout_dump | wc -l)" = "250" || { echo "seed $S: package dumps incomplete"; exit 2; }
  cp $LOG_DIR/${E[$S]}.driver.log $D/launch.log 2>/dev/null || true
done

# ---------- 6. band, step time, diagnostics, summary ----------
T=(); for S in 1 2 3; do T+=($(dirname $(find $H/runs/seed$S/tensorboard -name 'events.out*' | head -1))); done
C1=$(ls -d $TB2K/qwen3_0p6b_base_sf_tis_16n_seed1_16n64g_20260921_1823 2>/dev/null || true); C2=$(ls -d $TB2K/qwen3_0p6b_base_sf_tis_16n_seed2_16n64g_20260922_0040 2>/dev/null || true)
$PY $G/band_plot.py --tb ${T[@]} --labels seed1 seed2 seed3 --steps 0:240:20,250 --metrics "val-core/gsm8k_boxed_test/acc/mean@1=GSM8K test (1,319) accuracy, greedy" \
  --title "GB200 reference band: Qwen3-0.6B, GSM8K, cap 2048 + penalty, disaggregated async (3 seeds, 32 trainer / 32 rollout)" --out $H/band/gb200_band_gsm8k_2k_async1 > $H/band/band_stats.txt
tail -4 $H/band/band_stats.txt
if [ -n "$C1" ] && [ -n "$C2" ]; then
  $PY $G/band_plot.py --tb ${T[@]} $C1 $C2 --labels "disagg s1" "disagg s2" "disagg s3" "colocated s1" "colocated s2" --steps 0:240:20,250 --no_band \
    --metrics "val-core/gsm8k_boxed_test/acc/mean@1=GSM8K test (1,319) accuracy, greedy" --title "disaggregated async vs colocated (context), 2K cap + penalty, 64 GB200" --out $H/band/gb200_vs_colocated_2k > /dev/null
  SE=(); for S in 1 2 3; do SE+=($H/runs/seed$S/start_epoch.txt); done
  $PY $RECIPE_DIR/plot_step_time.py --tb ${T[@]} $C1 $C2 --labels "disagg s1" "disagg s2" "disagg s3" "colocated s1" "colocated s2" --breakdown_idx 0 3 --wait_idx 0 1 2 --target 0.80 --target_rule sustained2 \
    --start_epoch ${SE[@]} $LOG_DIR/../sf_tis_16n_seed1_start_epoch.txt $LOG_DIR/../sf_tis_16n_seed2_start_epoch.txt --out $H/band/gb200_step_time.png > $H/band/step_time.txt 2>&1 || \
  $PY $RECIPE_DIR/plot_step_time.py --tb ${T[@]} --labels "disagg s1" "disagg s2" "disagg s3" --breakdown_idx 0 1 --wait_idx 0 1 2 --target 0.80 --target_rule sustained2 --start_epoch ${SE[@]} --out $H/band/gb200_step_time.png > $H/band/step_time.txt
fi
for S in 1 2 3; do $PY $RECIPE_DIR/plot_phase0.py --tb ${T[$((S-1))]} --rollout $H/runs/seed$S/rollout_dump --groups 128 --group_size 16 --cap 2048 ${C1:+--ref_tb $C1 --ref_labels "colocated 2K seed1"} \
  --title "gsm8k_2k_async1 seed$S (sync every step, worst-lag<=1, span<=2; grey = colocated 2K seed1)" --out $H/band/diagnostics_seed$S.png > $H/band/diagnostics_seed$S.txt 2>&1; done
$PY - "$H" "${R[1]}" "${R[2]}" "${R[3]}" <<'PYX'
import sys, re, json, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
H = sys.argv[1]; out = {}
for s, r in enumerate(sys.argv[2:], 1):
    L = open(r + "/driver.log").read()
    a = EventAccumulator(next(iter(__import__("glob").glob(f"{H}/runs/seed{s}/tensorboard/*/"))), size_guidance={"scalars": 0}); a.Reload()
    S = lambda t: {e.step: e.value for e in a.Scalars(t)} if t in a.Tags()["scalars"] else {}
    W = lambda t: {e.step: e.wall_time for e in a.Scalars(t)} if t in a.Tags()["scalars"] else {}
    val, vwt, st, wt = S("val-core/gsm8k_boxed_test/acc/mean@1"), W("val-core/gsm8k_boxed_test/acc/mean@1"), S("timing_s/step"), W("timing_s/step")
    steady = [k for k in st if k >= 20 and k % 20 and k % 50]
    t0 = float(open(r + "/start_epoch.txt").read()); t1 = float(open(r + "/end_epoch.txt").read())
    vs = sorted(val); sus = [b for x, b in zip(vs, vs[1:]) if val[x] >= 0.80 and val[b] >= 0.80]
    ev = [float(x) for x in re.findall(r"evicted_samples:([0-9.]+)", L)]
    spans2 = len(re.findall(r"trajectory_spans/max:np\.int64\(2\)", L)); stale = max(int(x) for x in re.findall(r"trajectory_staleness_worst/max:np\.int64\((\d+)\)", L))
    d = lambda t, lo, hi: float(np.mean([v for k, v in S(t).items() if lo <= k <= hi])) if S(t) else None
    out[f"seed{s}"] = {"val": {k: round(val[k], 4) for k in vs}, "step0": round(val[0], 4), "final": round(val[250], 4),
        "sustained2_0.80": {"step": sus[0], "minutes_e2e": round((vwt[sus[0]] - t0) / 60, 1)} if sus else None,
        "steady_step_s": {"median": round(float(np.median([st[k] for k in steady])), 2), "p90": round(float(np.percentile([st[k] for k in steady], 90)), 2), "n": len(steady)},
        "components_median_s": {t.split("/")[1]: round(float(np.median([v for k, v in S(t).items() if k in steady])), 2) for t in ("timing_s/update_actor", "timing_s/update_weights", "timing_s/adv", "timing_s/old_log_prob", "timing_s/gen") if S(t)},
        "e2e_minutes": round((t1 - t0) / 60, 1), "eval_s_median": round(float(np.median(list(S("timing_s/testing").values()))), 1) if S("timing_s/testing") else None,
        "evicted_groups": int(sum(ev)), "steps_with_span2": spans2, "staleness_worst_max": stale,
        "diag_mean_1_250": {t: (round(d(t, 1, 250), 5) if d(t, 1, 250) is not None else None) for t in ("actor/rollout_corr/k3_kl", "actor/rollout_corr/rollout_is_eff_sample_size", "actor/rollout_corr/rollout_is_ratio_fraction_high", "response_length/mean", "actor/grad_norm")},
        "diag_mean_last10": {t: round(d(t, 241, 250), 4) for t in ("actor/entropy_loss", "response_length/mean", "response_length/clip_ratio", "critic/score/mean")}}
steps = sorted(out["seed1"]["val"]); per = {k: [out[f"seed{s}"]["val"][k] for s in (1, 2, 3)] for k in steps}
w = np.array([max(v) - min(v) for v in per.values()]) * 100
out["band"] = {"steps": steps, "per_step_min": {k: min(v) for k, v in per.items()}, "per_step_max": {k: max(v) for k, v in per.items()}, "per_step_mean": {k: round(float(np.mean(v)), 4) for k, v in per.items()},
    "width_pp_median": round(float(np.median(w)), 2), "width_pp_p90": round(float(np.percentile(w, 90)), 2), "width_pp_max": round(float(w.max()), 2),
    "final_mean": round(float(np.mean(per[250])), 4), "final_min": min(per[250]), "final_max": max(per[250]), "step0": per[0]}
out["recipe"] = "gsm8k_2k_async1"; out["image"] = open(H + "/env/IMAGE_REF.txt").read().strip(); out["verl_pin"] = open(H + "/env/VERL_PIN.txt").read().strip()
json.dump(out, open(H + "/band/summary.json", "w"), indent=1); print("band:", {k: out["band"][k] for k in ("final_mean", "final_min", "final_max", "width_pp_median", "width_pp_p90", "width_pp_max")})
for s in (1, 2, 3): o = out[f"seed{s}"]; print(f"seed{s}: final {o['final']} sustained0.80 {o['sustained2_0.80']} steady {o['steady_step_s']} e2e {o['e2e_minutes']} min evicted {o['evicted_groups']} span2-steps {o['steps_with_span2']}")
# step-0 per-question agreement across seeds (V1 val dumps: input/output/score)
D = {s: {json.loads(l)["input"]: json.loads(l) for l in open(f"{H}/runs/seed{s}/val_dump/0.jsonl")} for s in (1, 2, 3)}
qs = set(D[1]); res = {"n_questions": len(qs), "inputs_identical": all(set(D[s]) == qs for s in (2, 3)), "pairs": {}}
for a, b in ((1, 2), (1, 3), (2, 3)):
    res["pairs"][f"{a}-{b}"] = {"identical_outputs": sum(D[a][q]["output"] == D[b][q]["output"] for q in qs), "score_flips": sum((D[a][q]["score"] >= 1) != (D[b][q]["score"] >= 1) for q in qs)}
json.dump(res, open(H + "/band/step0_greedy_variability.json", "w"), indent=1); print("step-0 variability:", res["pairs"])
PYX

# ---------- 7. checkpoints (HF export at step 250) ----------
if [ "${SKIP_CKPT:-0}" != "1" ]; then
  for S in 1 2 3; do src=$CKPT_DIR/${E[$S]}/global_step_250/actor/huggingface; test -d $src || { echo "missing $src"; exit 2; }
    rm -rf $H/checkpoints/seed${S}_step250; cp -r $src $H/checkpoints/seed${S}_step250; ( cd $H/checkpoints/seed${S}_step250 && sha256sum *.safetensors > SHA256 ); done
fi

# ---------- 8. docs, README, manifest ----------
for D in TPU_GPU_RL_Parity_Rulebook_gsm8k_2k_async1.md Wenjun_TPU_Parity_Guide_gsm8k_2k_async1.md; do [ -f $RECIPE_DIR/$D ] && cp $RECIPE_DIR/$D $H/; done
cat > $H/README.md <<TXT
# GB200 reference package: Qwen3-0.6B / GSM8K / cap 2048 + overlong penalty / disaggregated asynchronous GRPO (recipe gsm8k_2k_async1)

Three 250-step runs on 64 GB200 (32 trainer + 32 rollout; seeds 1-3), everything needed to reproduce them on TPU, and the fixtures
for the alignment gates. Read TPU_GPU_RL_Parity_Rulebook_gsm8k_2k_async1.md (recipe, execution model, gates, reference numbers,
comparison rule) and Wenjun_TPU_Parity_Guide_gsm8k_2k_async1.md (step-by-step, Chinese). Generated $(date -u +%Y-%m-%dT%H:%MZ) by code/package_gsm8k_2k_async1.sh.
Software: image $IMG; verl verl-project/verl @ $PIN (env/versions.txt, env/uv.lock).

Verify: sha256sum -c --quiet PACKAGE_MANIFEST.sha256

| dir | contents |
|---|---|
| model/ | Qwen/Qwen3-0.6B weights + configs + tokenizer; MODEL_SHA256; model_identity.json |
| data/ | gsm8k_boxed_train.parquet (7473), gsm8k_boxed_test.parquet (1319); step_manifest_seed{1,2,3}.json (training-row indices per optimizer step, recovered from the rollout dumps); DATA_COUNTS.json |
| fixtures/ | prompt_fixture.json (gate 1); meta_reward_fixtures.json + scorer_fixture_2k.jsonl + reward_selftest_2k_penalty.log (gate 2) |
| code/ | recipe (recipe_gpu_disagg.yaml), launcher, preflight, reward, check_smoke, plots, Dockerfile, VERL_PIN / IMAGE_REF |
| env/ | IMAGE_REF, VERL_PIN, uv.lock, pip_freeze, versions, gpu, raycluster.yaml, resolved_config_seed{1,2,3}.yaml, preflight_seed*.json, CONFIG_DIFF.txt |
| runs/seed{1,2,3}/ | driver.log, launch.log, command.txt, resolved_config.yaml, preflight.json, check_smoke.json, tensorboard/, val_dump/<step>.jsonl (14), rollout_dump/<step>.jsonl (250), start/end epoch |
| band/ | gb200_band_gsm8k_2k_async1.png/.json, gb200_vs_colocated_2k.png, gb200_step_time.png, diagnostics_seed{1,2,3}.png, summary.json (all reference numbers), step0_greedy_variability.json |
| checkpoints/ | seed{1,2,3}_step250/ (HF safetensors) |

Reference experiment ids: ${E[1]}, ${E[2]}, ${E[3]}.
TXT
for f in model/model.safetensors model/MODEL_SHA256 model/model_identity.json data/gsm8k_boxed_train.parquet data/gsm8k_boxed_test.parquet data/DATA_COUNTS.json \
         data/step_manifest_seed1.json data/step_manifest_seed2.json data/step_manifest_seed3.json fixtures/prompt_fixture.json fixtures/meta_reward_fixtures.json \
         fixtures/scorer_fixture_2k.jsonl fixtures/reward_selftest_2k_penalty.log env/IMAGE_REF.txt env/VERL_PIN.txt env/uv.lock env/versions.txt env/CONFIG_DIFF.txt \
         env/resolved_config_seed1.yaml env/resolved_config_seed2.yaml env/resolved_config_seed3.yaml band/summary.json band/gb200_band_gsm8k_2k_async1.png \
         band/diagnostics_seed1.png band/diagnostics_seed2.png band/diagnostics_seed3.png band/step0_greedy_variability.json code/recipe_gpu_disagg.yaml code/run_gpu_disagg.sh \
         runs/seed1/check_smoke.json runs/seed2/check_smoke.json runs/seed3/check_smoke.json README.md; do
  test -s $H/$f || { echo "required file missing or empty: $H/$f"; exit 2; }
done
[ "${SKIP_CKPT:-0}" = "1" ] || for S in 1 2 3; do test -s $H/checkpoints/seed${S}_step250/SHA256 || { echo "checkpoint seed $S missing"; exit 2; }; done
( cd $H && find . -type f ! -name PACKAGE_MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum ) > $H/PACKAGE_MANIFEST.sha256
( cd $H && sha256sum -c --quiet PACKAGE_MANIFEST.sha256 ) && echo "PACKAGE OK: $(wc -l < $H/PACKAGE_MANIFEST.sha256) files -> $H" && du -sh $H
