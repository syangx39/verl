#!/usr/bin/env bash
# Assemble the GSM8K / Qwen3-0.6B / 8K-cap-no-penalty handoff package for the TPU side.
# Layout (same as the stab_kl0 package): model/ data/ fixtures/ code/ env/ runs/ band/ checkpoints/ + PACKAGE_MANIFEST.sha256
#
# Run on the head pod. Steps 4-5 need GPUs for ~10 minutes (a 2-step fixture job + one reference replay); everything else is CPU.
#   bash package_gsm8k_8k.sh            # full
#   SKIP_FIXTURE_JOB=1 bash package_gsm8k_8k.sh   # if the fixture job was already produced
set -euo pipefail
source /workspace/setup_env.sh
export RAY_ADDRESS=auto LOG_DIR=/workspace/meta-RL/logs CKPT_DIR=/workspace/meta-RL/ckpt
export DATA_DIR=/workspace/meta-RL/data/gsm8k_boxed MODEL_PATH=/workspace/meta-RL/models/Qwen3-0.6B META=/workspace/meta-RL/meta_pkg
unset TB_ROOT LOGPROB_FIXTURE_DIR LOGPROB_FIXTURE_STEP INJECT_BATCH_NPZ INJECT_BATCH_STEP
G=/workspace/meta-RL/verl/examples/grpo_trainer/my_scripts_GKE_gsm8k
G0=/workspace/meta-RL/verl/examples/grpo_trainer/my_scripts_GKE
H=${HANDOFF:-/workspace/meta-RL/handoff/gsm8k_8k}          # gcsfuse mount -> gs://xiaotongyang-bucket/meta-rl/GKE_repro/meta-RL/handoff/gsm8k_8k
TBROOT=/tmp/tb_local/meta_gsm8k_boxed
mkdir -p $H/{model,data,fixtures,code,env,runs,band,checkpoints}
rm -f $H/PACKAGE_MANIFEST.sha256

# ---------------------------------------------------------------- 0. the three reference runs
declare -A E
for S in 1 2 3; do
  L=$(ls -t $LOG_DIR/sf_tis_16n_cap8k_nopen_seed${S}_*.log | head -1)
  E[$S]=$(grep -o "qwen3_0p6b_base_sf_tis_16n_cap8k_nopen_seed${S}_[0-9a-z_]*" $L | tail -1)
  grep -q "driver exited with rc=0" $L || { echo "seed $S did not finish cleanly: $L"; exit 2; }
  [ "$(grep -cE 'injection ENABLED|_LOGPROB_FIXTURE\] wrote' $L)" = "0" ] || { echo "seed $S log shows injection/fixture activity -- not a clean reference run"; exit 2; }
  echo "seed $S -> ${E[$S]}  ($L)"; cp $L $H/runs/launch_seed${S}.log
done

# ---------------------------------------------------------------- 1. model (weights + configs + hashes)
cp $MODEL_PATH/{config.json,generation_config.json,tokenizer_config.json,tokenizer.json,vocab.json,merges.txt} $H/model/ 2>/dev/null || true
cp $MODEL_PATH/model.safetensors $H/model/
( cd $MODEL_PATH && sha256sum config.json generation_config.json tokenizer_config.json tokenizer.json vocab.json merges.txt model.safetensors ) > $H/model/MODEL_SHA256
python3 - <<EOF > $H/model/model_identity.json
import json, torch
from safetensors.torch import load_file
w=load_file("$MODEL_PATH/model.safetensors"); s=float(w["model.norm.weight"].float().sum())
print(json.dumps({"hf_id":"Qwen/Qwen3-0.6B (post-trained)","n_tensors":len(w),"lm_head_untied":"lm_head.weight" in w,"sum_model_norm_weight":s,
                  "eos_token_id":json.load(open("$MODEL_PATH/generation_config.json"))["eos_token_id"],"note":"sum(model.norm.weight) 3932.626953125 == post-trained checkpoint (Meta TPU doc fingerprint); Base reads 3926.20"},indent=1))
EOF

# ---------------------------------------------------------------- 2. data (Meta jsonl + our parquets + reference csv)
cp $DATA_DIR/gsm8k_boxed_train.parquet $DATA_DIR/gsm8k_boxed_test.parquet $DATA_DIR/gsm8k_boxed_test512.parquet $H/data/
cp $META/data/*.jsonl $H/data/ 2>/dev/null || true
rm -rf $H/data/meta_reference && mkdir -p $H/data/meta_reference && cp -r $META/reference/. $H/data/meta_reference/
python3 - <<EOF > $H/data/DATA_COUNTS.json
import pandas as pd, json
tr=pd.read_parquet("$DATA_DIR/gsm8k_boxed_train.parquet"); te=pd.read_parquet("$DATA_DIR/gsm8k_boxed_test.parquet")
def q(row):                                   # question text: extra_info.question if present, else the last (user) turn
    ei=row["extra_info"]
    if isinstance(ei,dict) and ei.get("question"): return " ".join(str(ei["question"]).split())
    msgs=list(row["prompt"]); return " ".join(str(msgs[-1]["content"]).split())
trq={q(r) for _,r in tr.iterrows()}; teq={q(r) for _,r in te.iterrows()}
print(json.dumps({"train_rows":len(tr),"test_rows":len(te),"train_unique_questions":len(trq),"test_unique_questions":len(teq),
                  "train_test_question_overlap":len(trq&teq),"test_index_base":int(te["extra_info"].iloc[0]["index"])},indent=1))
EOF
# per-seed data order (which 128 questions each step consumed), recovered from the rollout dumps
python3 - <<EOF
import json, glob, os, pandas as pd
for S,e in ((1,"${E[1]}"),(2,"${E[2]}"),(3,"${E[3]}")):
    rows=[]; man=[]
    for st in range(1,251):
        f=f"$LOG_DIR/{e}/rollout_dump/{st}.jsonl"
        recs=[json.loads(l) for l in open(f)]
        key=next((k for k in ("qid","index") if k in recs[0]), None)
        if key is None: raise SystemExit(f"rollout dump {f} carries no qid/index field ({list(recs[0])}); cannot recover the data order")
        seen=[]; [seen.append(int(r[key])) for r in recs if int(r[key]) not in seen]
        uids=len({r["uid"] for r in recs}) if "uid" in recs[0] else None
        assert len(recs)==2048 and len(seen)==128, (st,len(recs),len(seen))
        rows+= [{"step":st,"pos":i,"qid":q} for i,q in enumerate(seen)]; man.append({"step":st,"n_rows":len(recs),"n_questions":len(seen),"n_uids":uids})
    pd.DataFrame(rows).to_parquet(f"$H/data/train_order_seed{S}.parquet", index=False); json.dump(man, open(f"$H/data/step_manifest_seed{S}.json","w"))
    print(f"seed {S}: {len(rows)} rows, {len({r['qid'] for r in rows})} distinct questions over 250 steps")
EOF

# ---------------------------------------------------------------- 3. code + env
rm -rf $H/code && mkdir -p $H/code
cp $G/*.py $G/*.sh $G/*.md $H/code/ 2>/dev/null || true; cp $G0/make_logprob_fixture.py $G0/scorer_fault_injection.py $H/code/ 2>/dev/null || true
for f in compare_grads.py replay_single_step.py boxed_math_reward.py band_plot.py compare_two_runs.py; do test -s $H/code/$f || { echo "code/$f missing from $G -- commit and sync it before packaging"; exit 2; }; done
RT=$(python3 -c "import verl.trainer.ppo.ray_trainer as m; print(m.__file__)" | tail -1)
( cd $(dirname $RT)/../../.. && git rev-parse HEAD 2>/dev/null || echo unknown ) > $H/env/verl_commit.txt
cp $RT $H/env/ray_trainer_executed.py; grep -c "_DUMP_UID\|_LOGPROB_FIXTURE\|_INJECT_BATCH" $RT > $H/env/ray_trainer_patch_markers.txt
python3 -c "import vllm,torch,transformers,verl; print(f'vllm {vllm.__version__}\ntorch {torch.__version__}\ntransformers {transformers.__version__}\nverl {getattr(verl,\"__version__\",\"?\")}')" > $H/env/versions.txt
pip freeze > $H/env/pip_freeze.txt; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1 > $H/env/gpu.txt
for S in 1 2 3; do cp $LOG_DIR/${E[$S]}/resolved_config_preflight.yaml $H/env/resolved_config_seed${S}.yaml; done
python3 - <<EOF > $H/env/CONFIG_DIFF.txt
import yaml, json
c=[yaml.safe_load(open(f"$H/env/resolved_config_seed{s}.yaml")) for s in (1,2,3)]
def flat(d,p=""):
    for k,v in d.items():
        if isinstance(v,dict): yield from flat(v,p+k+".")
        else: yield p+k, json.dumps(v, sort_keys=True)
f=[dict(flat(x)) for x in c]; keys=set().union(*f)
diff=[k for k in sorted(keys) if len({x.get(k) for x in f})>1]
print("keys differing across the three seeds:", diff)   # expected: only data.seed / experiment_name / paths derived from them
EOF

# ---------------------------------------------------------------- 4. fixtures: prompt, scorer, logprob (2-step fixture job), replay reference
python3 $G/build_gsm8k_boxed_data.py --meta_data $META/data --out /tmp/fx_build --model_in $MODEL_PATH --model_out /tmp/fx_model --overwrite \
  --prompt_example $META/reference/prompt_example.json --n_eval 512 2>&1 | grep -E "fixture|mismatch|prompt" | head -5
cp /tmp/fx_build/prompt_fixture.json $H/fixtures/prompt_fixture.json
cp $META/reference/reward_fixtures.json $H/fixtures/meta_reward_fixtures.json
REWARD_MAX_RESP_LEN=8192 REWARD_PENALTY_SOURCES="" python3 $G/boxed_math_reward.py > $H/fixtures/reward_selftest_8k_nopenalty.log 2>&1 || true
REWARD_MAX_RESP_LEN=2048 REWARD_PENALTY_SOURCES=gsm8k_boxed_train python3 $G/boxed_math_reward.py > $H/fixtures/reward_selftest_meta_rule.log 2>&1 || { echo "reward self-test FAILED"; exit 2; }
# scorer fixture from real completions (step 0 and 250 of seed 1): expected acc/fmt/score under the 8K no-penalty rule
python3 - <<EOF
import json, sys, random
sys.path.insert(0, "$G"); import os; os.environ["REWARD_MAX_RESP_LEN"]="8192"; os.environ["REWARD_PENALTY_SOURCES"]=""
import importlib; R=importlib.import_module("boxed_math_reward")
rows=[]
for st in (0,250):
    for l in open(f"$LOG_DIR/${E[1]}/val_dump/{st}.jsonl"):
        r=json.loads(l); rows.append(r)
random.Random(0).shuffle(rows); rows=rows[:800]
out=[]
for r in rows:
    s=R.compute_score("gsm8k_boxed_train", r["output"], r["gts"], extra_info={"index": r["qid"], "response_len": 0})
    out.append({"output": r["output"], "gts": r["gts"], "expected": {k: s[k] for k in ("acc","fmt","length_penalty","score","reward_raw") if k in s}})
with open("$H/fixtures/scorer_fixture_8k.jsonl","w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False)+"\n")
print("scorer fixture rows:", len(out), "| acc mean", sum(o["expected"]["acc"] for o in out)/len(out))
EOF
SKIP_GPU=${SKIP_GPU:-0}                       # 1 = do not touch GPUs: reuse an existing fixture job + replay output (error if absent)
if [ "$SKIP_GPU" != "1" ] && [ "${SKIP_FIXTURE_JOB:-0}" != "1" ]; then
  # 2-step fixture job under the frozen recipe (two-pass so old_log_probs is the trainer's pre-update log-prob), dumps step 1+2, saves both checkpoints
  RUN_TAG=fx8k SINGLE_FWD=0 IS_MODE=tis META_ACTOR_LR=2e-6 META_RESP_CAP=8192 META_PENALTY_SOURCES="" SEED=1 NNODES=16 GPUS_PER_NODE=4 TOTAL_STEPS=2 TEST_FREQ=-1 SAVE_FREQ=1 EVAL_FULL=0 COLLAPSE_GUARD=0 \
    LOGPROB_FIXTURE_DIR=$LOG_DIR/fx8k/raw LOGPROB_FIXTURE_STEP=1,2 \
    bash $G/run_qwen3_0p6b_base_gsm8k_boxed.sh trainer.val_before_train=False data.val_files="['$DATA_DIR/gsm8k_boxed_test.parquet']" > $LOG_DIR/fx8k.log 2>&1
  grep -q "driver exited with rc=0" $LOG_DIR/fx8k.log || { echo "fixture job failed"; tail -20 $LOG_DIR/fx8k.log; exit 2; }
fi
EF=$(grep -o 'qwen3_0p6b_base_fx8k_seed1_[0-9a-z_]*' $LOG_DIR/fx8k.log | tail -1)
test -s $LOG_DIR/fx8k/raw/fixture_step1.npz || { echo "fixture dumps missing ($LOG_DIR/fx8k/raw); run without SKIP_GPU/SKIP_FIXTURE_JOB"; exit 2; }
python3 $G0/make_logprob_fixture.py --dump $LOG_DIR/fx8k/raw/fixture_step1.npz --out $H/fixtures/logprob_fixture_8k --n 96 --n_long 24 --n_trunc 8 | tail -3   # tool appends .json
test -s $H/fixtures/logprob_fixture_8k.json || { echo "logprob fixture not written"; exit 2; }
cp $LOG_DIR/fx8k/raw/fixture_step{1,2}.npz $LOG_DIR/fx8k/raw/fixture_step{1,2}.json $H/fixtures/
GN=$(grep -o "actor/grad_norm:[0-9.e-]*" $LOG_DIR/fx8k.log | head -1 | cut -d: -f2)
if [ "$SKIP_GPU" != "1" ]; then
  # reference replay of step 1: ratio-form loss -A*w*exp(logp-logp.detach()) has the SAME gradient as the single-forward
  # REINFORCE loss; the loss scalars differ by construction. --save_grad writes the per-parameter gradient (safetensors, ~2.4 GB)
  rm -rf $LOG_DIR/fx8k/replay_grad
  CUDA_VISIBLE_DEVICES=0 python3 $G/replay_single_step.py --dumps $LOG_DIR/fx8k/raw/fixture_step1.npz --lrs 0 --model $MODEL_PATH --micro 8 --reported_grad_norm $GN \
    --save_grad $LOG_DIR/fx8k/replay_grad --out $LOG_DIR/fx8k/replay_step1_reference_8k.json 2>&1 | grep -E "^\[2a\]|^\[2b\]|reported grad" | tee $LOG_DIR/fx8k/replay_step1_reference_8k.log
fi
test -s $LOG_DIR/fx8k/replay_grad/grad_step1.safetensors || { echo "replay gradient missing; run without SKIP_GPU"; exit 2; }
# GB200 trainer step-1 gradient (Adam exp_avg/0.1 of the fixture job) vs the reference gradient: CPU, ~3 min
python3 $G/verl_grad_from_optim.py --actor_dir $CKPT_DIR/$EF/global_step_1/actor --model $MODEL_PATH \
  --replay_grad $LOG_DIR/fx8k/replay_grad/grad_step1.safetensors --out $LOG_DIR/fx8k/grad_compare_8k.json 2>&1 | grep -A6 "global cosine" | tee $LOG_DIR/fx8k/grad_compare_8k.log
if [ "$SKIP_GPU" != "1" ]; then
  # two-step replay (lr 0, then 2e-7) vs the trainer's theta_2: GPU, ~4 min
  CUDA_VISIBLE_DEVICES=0 python3 $G/replay_single_step.py --dumps $LOG_DIR/fx8k/raw/fixture_step{1,2}.npz --lrs 0 2e-7 --model $MODEL_PATH --micro 8 \
    --post_weights $CKPT_DIR/$EF/global_step_2/actor/huggingface --out $LOG_DIR/fx8k/replay_delta_8k.json 2>&1 | grep -E "^\[4\]|rel_err|cos" | tee $LOG_DIR/fx8k/replay_delta_8k.log
fi
test -s $LOG_DIR/fx8k/replay_delta_8k.json || { echo "two-step replay missing; run without SKIP_GPU"; exit 2; }
cp $LOG_DIR/fx8k/replay_step1_reference_8k.json $LOG_DIR/fx8k/replay_step1_reference_8k.log $LOG_DIR/fx8k/grad_compare_8k.json $LOG_DIR/fx8k/grad_compare_8k.log \
   $LOG_DIR/fx8k/replay_delta_8k.json $LOG_DIR/fx8k/replay_delta_8k.log $H/fixtures/
rm -rf $H/fixtures/replay_grad_step1 && mkdir -p $H/fixtures/replay_grad_step1 && cp $LOG_DIR/fx8k/replay_grad/grad_step1.safetensors $H/fixtures/replay_grad_step1/
cat > $H/fixtures/replay_grad_step1/README.txt <<'TXT'
grad_step1.safetensors: per-parameter PRE-CLIP gradient of the reference implementation (pure PyTorch/HF, fp32 master, bf16 autocast,
micro-batch 8) on fixtures/fixture_step1.npz, loss = token-mean(-A * w * exp(logp - logp.detach())) with w = min(exp(logp.detach()-logp_sampler), 3).
Its gradient equals that of the single-forward REINFORCE loss -A*w*logp used by the trainer; the loss SCALARS differ by construction and
must not be compared. Compare a TPU gradient on the same batch with compare_grads.py (global/per-parameter cosine, relative error, float64).
GB200 trainer (exp_avg/0.1 at step 1) vs this file: see replay_step1_reference_8k.log and band/summary.json.
TXT
rm -rf $H/checkpoints/fixture_seed1_step2 && mkdir -p $H/checkpoints/fixture_seed1_step2 && cp -r $CKPT_DIR/$EF/global_step_2/actor/huggingface/. $H/checkpoints/fixture_seed1_step2/
cat > $H/checkpoints/fixture_seed1_step2/README.txt <<'TXT'
theta_2 of the fixture job = theta_0 -> update 1 on fixture_step1.npz at lr 0 (weights unchanged, Adam moments initialised from the
step-1 gradient) -> update 2 on fixture_step2.npz at lr 2e-7 (= 2e-6/10, warmup step 2). To reproduce, both steps must be replayed
in order with the same Adam hyperparameters (betas 0.9/0.999, eps 1e-8, wd 0, bias correction at steps 1 and 2).
TXT

# ---------------------------------------------------------------- 5. runs: TB, val dumps, rollout dumps, guard log, timing anchors
for S in 1 2 3; do
  R=$H/runs/seed${S}; rm -rf $R; mkdir -p $R
  cp -r $TBROOT/${E[$S]}/. $R/tensorboard/
  cp -r $LOG_DIR/${E[$S]}/val_dump/. $R/val_dump/; cp -r $LOG_DIR/${E[$S]}/rollout_dump/. $R/rollout_dump/
  cp $LOG_DIR/${E[$S]}/collapse_guard.log $R/ 2>/dev/null || true
  cp $LOG_DIR/sf_tis_16n_cap8k_nopen_seed${S}_start_epoch.txt $R/start_epoch.txt; cp $LOG_DIR/sf_tis_16n_cap8k_nopen_seed${S}_end_epoch.txt $R/end_epoch.txt 2>/dev/null || true
  echo "${E[$S]}" > $R/EXPERIMENT_NAME
  rm -rf $H/checkpoints/seed${S}_step250 && mkdir -p $H/checkpoints/seed${S}_step250 && cp -r $CKPT_DIR/${E[$S]}/global_step_250/actor/huggingface/. $H/checkpoints/seed${S}_step250/
done

# ---------------------------------------------------------------- 6. band + summary
T1=$TBROOT/${E[1]}; T2=$TBROOT/${E[2]}; T3=$TBROOT/${E[3]}
python3 $G/band_plot.py --tb $T1 $T2 $T3 --labels "seed1" "seed2" "seed3" --steps 0:240:20,250 \
  --metrics "val-core/gsm8k_boxed_test/acc/mean@1=GSM8K test (1,319) accuracy, greedy" --title "GB200 reference band: Qwen3-0.6B, GSM8K, cap 8192, no length penalty (3 seeds, 64 GPUs)" \
  --out $H/band/gb200_band_gsm8k_8k | tee $H/band/band_stats.txt
python3 $G/band_plot.py --tb $T1 $T2 $T3 --labels "seed1" "seed2" "seed3" --no_band --steps 0:240:20,250 \
  --metrics "val-core/gsm8k_boxed_test/acc/mean@1=GSM8K test (1,319) accuracy, greedy" --title "GB200 reference runs: Qwen3-0.6B, GSM8K, cap 8192, no length penalty" --out $H/band/gb200_curves_gsm8k_8k >/dev/null
for S in 1 2 3; do python3 $G/plot_phase0.py --tb $TBROOT/${E[$S]} --rollout $LOG_DIR/${E[$S]}/rollout_dump --groups 128 --group_size 16 --cap 8192 --out $H/band/diagnostics_seed${S}.png > $H/band/diagnostics_seed${S}.txt 2>&1; done
python3 - <<EOF > $H/band/summary.json
from tensorboard.backend.event_processing import event_accumulator as ea
import numpy as np, json
S={}
for s,tb,ep in ((1,"$T1","sf_tis_16n_cap8k_nopen_seed1"),(2,"$T2","sf_tis_16n_cap8k_nopen_seed2"),(3,"$T3","sf_tis_16n_cap8k_nopen_seed3")):
    a=ea.EventAccumulator(tb, size_guidance={ea.SCALARS:0}); a.Reload(); t=lambda k:{e.step:e.value for e in a.Scalars(k)} if k in a.Tags()["scalars"] else {}
    ev=a.Scalars("val-core/gsm8k_boxed_test/acc/mean@1"); st=a.Scalars("timing_s/step"); t0=int(open(f"$LOG_DIR/{ep}_start_epoch.txt").read())
    steady=[x.value for x in st if 20<=x.step<=250 and x.step%20!=0 and x.step%50!=0 and x.step!=250]   # rulebook window: steps 20-250 minus eval/ckpt steps
    def hit(thr):
        h=next((ev[i] for i in range(1,len(ev)) if ev[i].value>=thr and ev[i-1].value>=thr), None)
        return None if h is None else {"step":h.step,"e2e_min":(h.wall_time-t0)/60,"gpu_hours_64":64*(h.wall_time-t0)/3600}
    m=lambda d,lo,hi: float(np.mean([v for k,v in d.items() if lo<=k<=hi])) if d else None
    S[f"seed{s}"]={"eval":{e.step:e.value for e in ev},"fmt_step0":t("val-aux/gsm8k_boxed_test/fmt/mean@1").get(0),"steady_step_s_median":float(np.median(steady)),"steady_step_s_p10_p90":[float(np.percentile(steady,10)),float(np.percentile(steady,90))],"steady_n_steps":len(steady),
        "time_to_0.78":hit(0.78),"time_to_0.80":hit(0.80),"e2e_total_min":(st[-1].wall_time-t0)/60,
        "diag_mean_1_250":{"entropy":m(t("actor/entropy_loss"),1,250),"response_length":m(t("response_length/mean"),1,250),"cap_hit":m(t("response_length/clip_ratio"),1,250),"grad_norm":m(t("actor/grad_norm"),1,250),"train_score":m(t("critic/score/mean"),1,250),"rollout_probs_diff_mean":m(t("training/rollout_probs_diff_mean"),1,250),"rollout_corr_kl":m(t("rollout_corr/kl"),1,250)},
        "diag_last10":{"entropy":m(t("actor/entropy_loss"),241,250),"response_length":m(t("response_length/mean"),241,250),"grad_norm":m(t("actor/grad_norm"),241,250),"train_score":m(t("critic/score/mean"),241,250)}}
steps=sorted(S["seed1"]["eval"]); W=[max(S[f"seed{s}"]["eval"][k] for s in (1,2,3))-min(S[f"seed{s}"]["eval"][k] for s in (1,2,3)) for k in steps]
fin=[S[f"seed{s}"]["eval"][250] for s in (1,2,3)]
S["band"]={"steps":steps,"width_median":float(np.median(W)),"width_p90":float(np.percentile(W,90)),"width_max":float(max(W)),"final_mean":float(np.mean(fin)),"final_min":min(fin),"final_max":max(fin),
           "step0":[S[f"seed{s}"]["eval"][0] for s in (1,2,3)],"gain":[S[f"seed{s}"]["eval"][250]-S[f"seed{s}"]["eval"][0] for s in (1,2,3)]}
print(json.dumps(S, indent=1))
EOF
python3 -c "import json; b=json.load(open('$H/band/summary.json'))['band']; print('band:', {k:(round(v,4) if isinstance(v,float) else v) for k,v in b.items()})"

# ---------------------------------------------------------------- 7. step-0 eval nondeterminism evidence (same weights, greedy) + docs + manifest
python3 - <<EOF > $H/band/step0_greedy_variability.json
import json, itertools, numpy as np
def load(p): return {json.loads(l)["qid"]: json.loads(l) for l in open(p)}
D={s:load(f"$H/runs/seed{s}/val_dump/0.jsonl") for s in (1,2,3)}; qs=set(D[1]); out={"n_questions":len(qs),"inputs_identical":all(D[1][q]["input"]==D[2][q]["input"]==D[3][q]["input"] for q in qs),"gts_identical":all(D[1][q]["gts"]==D[2][q]["gts"]==D[3][q]["gts"] for q in qs),"pairs":{}}
for a,b in itertools.combinations((1,2,3),2):
    out["pairs"][f"{a}-{b}"]={"identical_outputs":sum(D[a][q]["output"]==D[b][q]["output"] for q in qs),"acc_flips":sum(D[a][q]["acc"]!=D[b][q]["acc"] for q in qs),"acc":[np.mean([D[a][q]["acc"] for q in qs]),np.mean([D[b][q]["acc"] for q in qs])]}
acc=np.array([[D[s][q]["acc"] for q in sorted(qs)] for s in (1,2,3)]); out["unstable_questions_1_or_2_of_3"]=int(((acc.sum(0)>0)&(acc.sum(0)<3)).sum()); out["identical_across_all_three"]=sum(1 for q in qs if D[1][q]["output"]==D[2][q]["output"]==D[3][q]["output"])
print(json.dumps(out, indent=1))
EOF
for D in TPU_GPU_RL_Parity_Rulebook_gsm8k_8k.md Tianyu_TPU_Parity_Guide_gsm8k_8k.md; do
  if [ -s $G/handoff/$D ]; then cp $G/handoff/$D $H/; elif [ -s $G/$D ]; then cp $G/$D $H/; else echo "WARNING: $D not found under $G or $G/handoff"; fi
done
( cd $H && find . -type f ! -name PACKAGE_MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum ) > $H/PACKAGE_MANIFEST.sha256
( cd $H && sha256sum -c --quiet PACKAGE_MANIFEST.sha256 ) && echo "PACKAGE OK: $(wc -l < $H/PACKAGE_MANIFEST.sha256) files -> $H" && du -sh $H
