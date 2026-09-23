# TPU 侧复现指南 — GSM8K / Qwen3-0.6B / 8K response cap（recipe `gsm8k_8k_v1`）

给 Tianyu。目标：在 TPU 上用**同一个 recipe**跑出和 GB200 三条参考 run 相容的 250 步曲线，并按 rulebook 的判据报告。这轮和上一轮（OMI2 `stab_kl0`）的做法一样：先过 5 道门，每道门都便宜、都能把问题定位到一层；门没过之前不要开长 run。

包在 `gs://xiaotongyang-bucket/meta-rl/GKE_repro/meta-RL/handoff/gsm8k_8k/`，先校验：

```bash
gcloud storage rsync -r gs://xiaotongyang-bucket/meta-rl/GKE_repro/meta-RL/handoff/gsm8k_8k ./gsm8k_8k
cd gsm8k_8k && sha256sum -c --quiet PACKAGE_MANIFEST.sha256 && echo PACKAGE OK
```

判据、容差、每个参考数字的出处都在 `TPU_GPU_RL_Parity_Rulebook_gsm8k_8k.md`，本指南只讲"怎么做"。

---

## 0. 这轮和上一轮的差别（先读）

| | 上轮 `stab_kl0` | 这轮 `gsm8k_8k_v1` |
|---|---|---|
| 模型 | Qwen3-0.6B | Qwen3-0.6B（同一个 post-trained checkpoint，**不是 Base**）|
| 数据 | OMI2 100 万题 | GSM8K train 7,473 题，反复 ≈4.3 遍 |
| eval | OMI2 1k + GSM8K 1,319 | 只有 GSM8K **全量 1,319**，每 20 步一次 |
| batch / 步数 | 256×8，300 步 | **128×16**，250 步 |
| lr | 1e-6 常数 | **2e-6，warmup 10 步，cosine 到 0** |
| reward | maxtext_math_reward（math-verify）| **boxed_math**，纯字符串规则，1.0 / 0.1 / 0 |
| 长度 | prompt 8192 / response 8192，有软惩罚 | prompt 512 / **response 8192，无惩罚** |
| loss | 两遍前向、ratio≡1、无 IS | **单前向 REINFORCE + detached TIS（截 3.0）** |
| 硬件 | 64 GB200 ↔ 64 v7x | 同 |

三处最容易走样的地方：**TIS 权重的形式**、**token-mean 的分母**、**chat template 的 thinking 设置**。下面每处都有对应的门。

---

## 1. Recipe 一览（Tier 1，一个都不能改）

```
model        : Qwen/Qwen3-0.6B  (package/model/, MODEL_SHA256; 311 tensors, lm_head untied, sum(model.norm.weight)=3932.626953125)
data         : data/gsm8k_boxed_train.parquet (7473)   eval: data/gsm8k_boxed_test.parquet (1319)
prompt       : system + user（parquet 里的原文）→ Qwen3 chat template, add_generation_prompt=True, enable_thinking=True(默认)
               渲染结果以 <|im_start|>assistant\n 结尾，不含任何 <think> token；token ids 见 fixtures/prompt_fixture.json
batch        : 128 prompts × 16 = 2048 / step，μ=1（整批一次 optimizer.step），250 步
optimizer    : AdamW lr 2e-6, warmup 10 (update1 lr=0, update2 lr=2e-7), cosine→0 @250, betas (0.9,0.999), eps 1e-8, wd 0, clip 1.0
advantage    : GRPO: (R − 组均值) / (组 std(ddof=1) + 1e-6)，组 = 同一 prompt 的 16 条，广播到每个 response token；零方差组保留
loss         : token-mean over 整批 valid response tokens of  −A · w · log π_θ
               w = min( exp( clamp( logπ_θ.detach() − logπ_sampler, −20, 20 ) ), 3.0 )   ← 在 no_grad 里算，不反传
               分母 = 整批 2048 条的 valid response token 总数（含终止 token，不含 prompt）
               KL 0（无参考模型），entropy 0，无 dynamic sampling，无 clip 作用（ratio≡1）
sampling     : T=1.0, top_p 1.0, top_k 关, stop=[151645,151643], response cap 8192, 到 cap 的样本整条保留
reward       : boxed_math：最后一个 \boxed{}，括号配平，minimal normalize；对 1.0 / 有 box 但错 0.1 / 无 box 0；无长度惩罚
eval         : greedy n=1, cap 8192, 步 0,20,…,240,250 共 14 次；acc = (raw reward == 1)
precision    : fp32 master + fp32 Adam；bf16 compute / rollout / KV；logp 在 fp32 里算
data order   : seed k 用 data/train_order_seed{k}.parquet（250 步 × 128 题），shuffle 关；做不到就原生 shuffle 并记录每步 qid
```

---

## 2. 门 1 · Prompt 渲染（CPU，5 分钟）

用你们的 tokenizer + template 渲染 `fixtures/prompt_fixture.json` 里的每一行，逐 token 对比 `token_ids`：

```python
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gsm8k_8k/model")
fx = json.load(open("gsm8k_8k/fixtures/prompt_fixture.json"))
bad = 0
for r in fx["rows"]:
    ids = tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=True)   # enable_thinking 用模板默认
    if ids != r["token_ids"]:
        bad += 1; print("MISMATCH", r.get("qid"), len(ids), len(r["token_ids"]), ids[:5], ids[-5:])
print("mismatch rows:", bad, "/", len(fx["rows"]))
```

要求：0 行不一致。典型走样：Tunix 自己拼 ChatML 标记时多/少一个换行、注入了空的 `<think></think>`、system 轮被丢掉。渲染结果的最后三个 id 必须是 `151644, 77091, 198`，倒数第四个之前不应出现 `151667`（`<think>`）。

---

## 3. 门 2 · Scorer（CPU，5 分钟）

`code/boxed_math_reward.py` 是纯 Python 字符串逻辑，直接 import 或逐字 port。两组 fixture 都要逐行相等：

```bash
cd gsm8k_8k/code
# (a) Meta 的 21 条规则用例：这一组按 cap 2048 + 惩罚开启验证规则本身（惩罚只在这一组里用到）
REWARD_MAX_RESP_LEN=2048 REWARD_PENALTY_SOURCES=gsm8k_boxed_train python3 boxed_math_reward.py      # 期望 RESULT PASS 21/21
# (b) 800 条真实回答（step 0 / 250），8K 无惩罚规则下的期望 acc/fmt/score
REWARD_MAX_RESP_LEN=8192 REWARD_PENALTY_SOURCES="" python3 - <<'EOF'
import json, importlib
R = importlib.import_module("boxed_math_reward"); bad = 0; n = 0
for l in open("../fixtures/scorer_fixture_8k.jsonl"):
    r = json.loads(l); n += 1
    s = R.compute_score("gsm8k_boxed_train", r["output"], r["gts"], extra_info={"index": 0, "response_len": 0})
    if any(abs(s[k] - r["expected"][k]) > 1e-9 for k in r["expected"]): bad += 1
print("scorer fixture mismatches:", bad, "/", n)
EOF
```

如果你们 port 到 JAX/Python 侧，把上面的 `R.compute_score` 换成你们的函数。注意三条规则：取**最后一个** `\boxed{`；括号不配平（被截断）→ 0；`18.0` 和 `18` 不相等（不做数值归一）。

---

## 4. 门 3 · Step-0 eval（TPU，~10 分钟）

不训练，只用初始权重对 1,319 题做 greedy（T=0，n=1，cap 8192），输出成 `val_dump/0.jsonl`（格式见第 8 节），算 acc。

GPU 参考（**同一权重的三次 greedy**）：0.7453 / 0.7400 / 0.7582，均值 0.748；格式率 0.92–0.93。

- 要求：TPU 的 acc 在 **[0.728, 0.768]** 内。
- 同时报告逐题 agreement（和 `runs/seed1/val_dump/0.jsonl` 比：输出完全相同的题数、acc 翻转的题数）。这只是诊断，不是判据——GPU 自己三次之间也只有 215–241 题输出相同、约 130 题翻转（rulebook "Evaluation variability"）。

超出范围先查：thinking 是否被关（格式率会变，长度分布会变）、cap 是否 8192、stop token、tokenizer 版本。

---

## 5. 门 4 · Trainer vs sampler 数值（TPU，~30 分钟）

`fixtures/logprob_fixture_8k.json`：96 条序列（含 24 条长、8 条到 cap 的），每条有 `prompt_ids`、`response_ids`、`response_mask`、`position_ids`、`logp_sampler`（vLLM）、`logp_trainer`（FSDP，更新前）。

(a) 用 TPU 的 trainer 在同一权重上对这 96 条算逐 token logp（fp32），和 `logp_trainer` 比：mean |Δ|、p95、max。
(b) 用 TPU 自己的 sampler 在冻结采样设置下采一批，算 sampler-vs-trainer 的同样统计。

GPU 参考值在 fixture 文件头和 `band/summary.json`（250 步均值：probability MAE ≈ 0.005，`rollout_corr/kl` ≈ 0.0007）。触发调查的阈值：非负误差量（MAE、mean |Δlogp|、尾分位）比 GPU 大一个数量级。这是调查触发器，不是通过标准；常见原因：温度/概率归一化不同、mask 不同、logp 没在 fp32 算、权重同步没完成。

**TIS 的实现在这一步一起核**：w 必须用 (b) 里的 sampler logp 和训练 pass 的 logp（detach）算，截到 3.0。检查三件事：`sampler` 返回的是采样 token 在 T=1 下的 logp（不是 greedy 的）；w 不带梯度；出界比例（w>3）应接近 0。**不要**把 sampler logp 放进 PPO ratio 的分母（Meta TPU 组 a26d 的做法）——我们在同一批上验过，那和 TIS 不等价（梯度余弦 0.99、方向差 14%，250 步慢 60 步）。

---

## 6. 门 5 · 单步重放（推荐，TPU，~1 小时）

`fixtures/fixture_step1.npz`（+ `.json` sidecar）是 GPU seed-1 fixture job 第 1 步的**完整批**：`prompts`、`responses`、`attention_mask`、`response_mask`、`position_ids`、`rollout_log_probs`、`old_log_probs`（trainer 更新前）、`token_level_scores`、`advantages`、`nt__uid`、`nt__qid`。

(a) **advantage**：按 uid 分组用 `token_level_scores` 重算，和 `advantages` 比（GPU vs 独立参考：max |Δ| 4e-7）。这一步核 ddof、eps、广播。
(b) **loss / 梯度**：把这一批原样注入你们的 trainer（同权重），算 −A·w·logπ 的 token-mean 和梯度，和 `fixtures/replay_step1_reference_8k.json` 比：梯度范数、方向余弦（GPU vs 参考：余弦 ≈ 0.99、rel err ≈ 14%，post-trained 低 entropy 下的 bf16 kernel 差）。loss 标量只在同一 loss 形式下才可比。
(c) **optimizer**：从 θ₀ 用第 1 步（lr 0）和第 2 步（lr 2e-7，`fixture_step2.npz`）各更新一次，和 `checkpoints/fixture_seed1_step2_after_first_nonzero_update/` 比 Δθ（GPU 自身 Adam 应用误差：1 ulp）。

---

## 7. 门 6 · 三条 250 步 run

```
seed k:  data order = data/train_order_seed{k}.parquet；eval 每 20 步 + 250；保存 step 250 权重
记录：TB 标量（第 8 节的 tag）、val_dump/<step>.jsonl、rollout_dump/<step>.jsonl、启动/结束时间戳、resolved config
```

跑之前预期（GB200 三条）：
- step 0 → 250：0.748 → 0.806 [0.796, 0.823]；三条曲线互相差 2–6pp 是正常的。
- entropy 从 0.5 略降到 0.50–0.54 就不再下降；长度从 1,350 涨到 2,300 左右再慢慢漂；cap-hit 1–2%；训练 reward（T=1）0.90。
- **entropy 掉到 0.35 以下、长度掉到 1,000 以下**是 loss 形式或 cap 不对的信号（那是 2K 那套 recipe 的形态），立刻停下来查。
- 64 v7x 上的步时你们自己测；GB200 是 39–40 s。

时间指标：从启动时刻记 epoch，达标点用 eval 的 wall time；报 **time-to-0.78**（连续两次 eval ≥0.78）为主，0.80 为辅——GB200 三条里只有一条连续过 0.80。

---

## 8. 交付格式（GPU 工具能直接读）

**val_dump/<step>.jsonl**（每次 eval 一个文件，每题一行）：

```json
{"input": "<渲染后的 prompt 文本，去掉特殊 token>", "output": "<模型输出>", "gts": "<标准答案>",
 "score": 1.0, "reward_raw": 1.0, "acc": 1.0, "fmt": 1.0, "length_penalty": 0.0, "qid": 10000659, "step": 20}
```

**rollout_dump/<step>.jsonl**（每步 2048 行）：`uid`（同 prompt 的 16 条共享）、`qid`、`acc`、`score`、`fmt`。

**TensorBoard 标量**（tag 名要完全一致）：`val-core/gsm8k_boxed_test/acc/mean@1`、`val-aux/gsm8k_boxed_test/fmt/mean@1`、`critic/score/mean`、`actor/entropy_loss`（或 `actor/entropy`）、`actor/grad_norm`、`actor/lr`、`response_length/mean`、`response_length/clip_ratio`、`timing_s/step`、`training/rollout_probs_diff_mean`、`rollout_corr/kl`。

拿到这些后我这边一条命令出对照图：

```bash
python3 code/band_plot.py --tb runs/seed1/tensorboard runs/seed2/tensorboard runs/seed3/tensorboard --labels "GB200 s1" "GB200 s2" "GB200 s3" \
  --extra <tpu_seed1_tb> <tpu_seed2_tb> <tpu_seed3_tb> --extra_labels "TPU s1" "TPU s2" "TPU s3" \
  --steps 0:240:20,250 --metrics "val-core/gsm8k_boxed_test/acc/mean@1=GSM8K test (1,319) acc, greedy" --out tpu_vs_gb200
```

---

## 9. 常见坑（这轮新增的）

1. **`enable_thinking`**：Qwen3 模板默认 True = 不注入 think token，模型自己决定；设成 False 会注入空 `<think></think>`，输出分布完全不同。门 1 会抓到。
2. **TIS 的三个要求**：sampler 返回的是采样 token 的 logp、w 不反传、截上界 3.0 不截下界。不要用 PPO-ratio 形式。
3. **分母**：全批 token 总数，不是逐序列均值再平均（Meta TPU 组用的 seq-mean-token-mean 会改变梯度）。
4. **终止 token 在 loss 里**：`<|im_end|>` 那个位置 mask=1。
5. **lr 调度**：第 1 次 update 的 lr 是 0（θ₁=θ₀），第 2 次 2e-7；打印实际进 optimizer 的 lr 核一次。
6. **fp32 logp**：logits 转 fp32 再 log_softmax；bf16 下 logp 会差 0.02 nats 量级，门 4 会变大。
7. **greedy eval 本身不可复现**：逐题输出对不上不是 bug；看均值和分布。
8. **没有长度惩罚、没有 KL**：长度会慢慢漂（GB200 250 步内到 2.3–2.6k），这是预期；跑更长的话要另议。

有任何一道门过不去，先发我该门的输出，别往下跑。
