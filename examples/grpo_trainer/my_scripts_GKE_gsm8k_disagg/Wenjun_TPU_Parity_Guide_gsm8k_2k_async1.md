# TPU 侧复现指南 — GSM8K / Qwen3-0.6B / 2K cap + 长度惩罚 / 分离式异步 GRPO（recipe `gsm8k_2k_async1`）

给 Wenjun 的 TorchTitan TPU 组。目标：在 TPU 上用**同一个 recipe、同一套异步执行语义**跑出和 GB200 三条参考 run 相容的 250 步曲线，并按 rulebook 的判据报告。判据、参考数字和交付格式以英文 rulebook（`TPU_GPU_RL_Parity_Rulebook_gsm8k_2k_async1.md`）为准，本指南是操作顺序。

包：`gs://xiaotongyang-bucket/meta-rl/GKE_repro/meta-RL/handoff/gsm8k_2k_async1/`（先 `sha256sum -c --quiet PACKAGE_MANIFEST.sha256`，再读 `README.md`）。

## 0. 这轮和 Meta 那轮（`gsm8k_8k_v1`）的差别（先读）

| | Meta 轮 `gsm8k_8k_v1` | 这轮 `gsm8k_2k_async1` |
|---|---|---|
| 执行模型 | colocated、同步、on-policy | **分离 + 异步**：32 张训练卡、32 张采样卡；sampler 领先一批；每次更新后推权重 |
| response cap / 惩罚 | 8192，无惩罚 | **2048，训练时有超长惩罚**（eval 无惩罚）|
| 算法 | 单前向 REINFORCE + 截断 IS（3.0） | 相同 |
| 采样策略版本 | 永远 = 当前策略 | **落后一步**（生成 batch t+1 时 trainer 在更新 θ_t）；一条回答可能跨两个版本 |
| 门 4/5 | logp fixture + 梯度重放 | 换成**异步语义门**（staleness / spans / 丢弃 / IS 统计）+ loss 合约自检；没有梯度 fixture |
| 参考数字 | 终点 0.806 [0.796, 0.823] | 终点 **0.827 [0.823, 0.832]**，band 宽 median 2.0 pp |

门的性质：门 1、2 是**硬性一致**（不等就是实现错误，先修）；门 3、4 是**诊断参考**（超出范围按 rulebook 查原因，不是自动失败）；门 5 是推荐检查。门 1、2 没过之前不要开长 run。

## 1. Recipe 一览（Tier 1，一个都不能改）

- 模型：`model/` 里的 post-trained Qwen3-0.6B（`MODEL_SHA256`；`sum(model.norm.weight)=3932.626953125`；eos `[151645, 151643]`）。
- 数据：`data/gsm8k_boxed_train.parquet`（7,473）/ `gsm8k_boxed_test.parquet`（1,319）。
- Prompt：parquet 里的 system+user，Qwen3 chat template，`add_generation_prompt=True`，`enable_thinking=True`（不注入 `<think>`），以 `<|im_start|>assistant\n` 结尾，prompt 上限 512。
- 每步：128 题 × 16 条 = 2,048 条；一次 AdamW 更新；250 步。
- 优化器：lr 2e-6，warmup 10 步（第 1 次更新 lr=0，第 2 次 2e-7 …），cosine 到 0 @250；betas (0.9, 0.999)，eps 1e-8，wd 0，全局 grad clip 1.0。
- 优势：GRPO，组内 (R − mean)/(std + 1e-6)，样本标准差（ddof=1），广播到每个 response token；零方差组保留在分母里。
- 损失：`−A · w · logπ_θ` 的**全批 token 均值**（分母 = 2,048 条的有效 response token 总数，含结束 token）；`w = min(exp(logπ_θ.detach() − logπ_sampler), 3.0)`，不带梯度；**`logπ_sampler` 是 sampler 自己记录的、生成该 token 时那个策略版本下的逐 token logp**；ratio ≡ 1，clip 无效；无 KL、无 entropy 项。
- 采样：T=1，top_k −1，top_p 1，stop `[151645, 151643]`。
- **Reward**：`code/boxed_math_reward.py`：最后一个括号平衡的 `\boxed{}`，精确匹配 1.0 / 有框但错 0.1 / 无框 0。**训练时**再加惩罚 `−min(1, max(0, (L − 1536)/512))`（L = 生成 token 数含结束 token；1536 以下无惩罚，2048 时 −1.0）；eval 不加。
- Eval：greedy，n=1，cap 2048，全量 1,319，step 0/20/…/240/250（14 次），在 sampler 池上用刚同步的权重；指标 `acc = (raw reward == 1)`。
- 精度：fp32 master + Adam，bf16 计算/采样/KV，logp 用 fp32。

## 2. 异步执行模型（Tier 2，这轮的核心，必须一致）

用主语说清楚谁在做什么：

1. **sampler** 先在 θ₀ 下生成一整批（warmup 1 批）；**trainer** 用它做第 1 次更新（staleness 0）。
2. 之后 **sampler** 始终在生成"下一批"，**trainer** 在用"当前批"更新；**trainer** 每次更新完立刻把 θ_t 推给全部 sampler 副本（GB200 上 NCCL，0.8–1.0 s）。**sampler 不会中断正在生成的请求**：已生成的 token 保留，后面的 token 用新权重接着采。
3. **sampler** 逐题下发（每题 16 条），完成的题组进队列；**trainer** 按完成顺序取 128 个完整题组做一次更新。
4. 一个题组如果最老的 token 是在 **2 个以上**版本之前生成的，**trainer** 直接丢掉它（阈值 2，策略 drop），不重采、不顺延。
5. eval 时 **trainer** 等待，**sampler** 用被评估的那个 checkpoint 的权重跑 1,319 题。

配置本身只保证：被消费的题组最老 token 最多落后 **2** 个版本（阈值 2），一条回答最多跨 3 个版本；跨版本回答是允许且预期的。GB200 三条 run 的**实测**（`band/summary.json`，是观测值不是保证）：被消费 token 的最大滞后 = 1（每一步都是）；**一半以上的步（136/138/144 of 250）里至少有一条回答跨了两个版本**（同步发生在它生成中途）——这是常态；丢弃 **11 / 8 / 8 组**（占 32,000 组的 0.03%，全是接近 2048 的长回答，staleness 3）。TPU 上出现 staleness 2 的题组被消费是合法的，报告分布即可。IS 权重按 token 记录、按 token 修正，所以跨版本的回答不需要特殊处理；`k3_kl`（trainer vs sampler）稳态 7e-4，15–40 步策略变化最快时升到 1.6e-3 再回落——这是滞后的唯一可见特征。

TPU 侧要做到：同样的"领先一批、每次更新推权重、warmup 1、阈值 2 + drop"（必须一致的是配置）；记录每步被消费题组的 staleness 分布、每条回答的 spans、丢弃数（这些是报告项，期望和 GB200 相近，不要求相等）。如果 TPU 的 sampler 在权重更新时**不能**接着生成 in-flight 请求（只能中断重来），明确写出来：那样 spans 恒为 1、丢弃率和批次构成会变，是要记录的偏差，不是违规。

## 3. 门 1 · Prompt 渲染（CPU，5 分钟）

和 8K 包完全一样：用同一 tokenizer + chat template 渲染 `fixtures/prompt_fixture.json` 里的 10 条 messages（5 条训练、5 条测试），渲染文本和 token id 逐位相等。

```bash
python3 - <<'EOF'
import json; from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("model"); fx = json.load(open("fixtures/prompt_fixture.json"))
bad = 0
for ex in fx["rows"]:
    text = tok.apply_chat_template(ex["messages"], add_generation_prompt=True, tokenize=False)   # 默认 enable_thinking=True
    ids = tok(text, add_special_tokens=False)["input_ids"]
    bad += (text != ex["rendered_text"]) or (ids != ex["token_ids"])
print("prompt fixture:", "OK" if bad == 0 else f"{bad} mismatches", "| first prompt", fx["rows"][0]["n_tokens"], "tokens")
EOF
```

不一致的常见原因：`enable_thinking` 没传（模板给 assistant 段注入了 `<think>\n\n</think>`）、system prompt 被改、tokenizer 不是包里的。

## 4. 门 2 · Scorer（CPU，5 分钟）

```bash
( cd code && python3 boxed_math_reward.py --fixtures ../fixtures/meta_reward_fixtures.json )        # 21 条规则用例
python3 - <<'EOF'
import json, sys; sys.path.insert(0, "code"); import boxed_math_reward as r
bad = 0
for l in open("fixtures/scorer_fixture_2k.jsonl"):
    x = json.loads(l); s = r.compute_score("gsm8k_boxed_test", x["response"], x["ground_truth"], extra_info={"index": 0, "response_len": 1})
    s = s["score"] if isinstance(s, dict) else s; bad += abs(s - x["expected_score"]) > 1e-6
print("scorer fixture (800 real eval responses):", "OK" if bad == 0 else f"{bad} mismatches")
EOF
cat fixtures/reward_selftest_2k_penalty.log        # 训练模式惩罚：len 100→1.0, 1536→1.0, 1792→0.5, 2048→0.0
```

TPU 侧用自己的 scorer 实现跑同一份 800 条，期望分数逐条相等；再用几个长度核惩罚公式（注意 L 含结束 token、只对训练 source 生效）。

## 5. 门 3 · Step-0 eval（TPU sampler，~5 分钟）

用初始权重 greedy 跑 1,319 题。GB200：0.7066 / 0.7066 / 0.7028。差 1 pp 以内正常；差 3 pp 以上先查 prompt 渲染、eos、cap（2048）、greedy 设置。逐题对比只用来定位（见 `band/step0_greedy_variability.json`）。

## 6. 门 4 · 异步语义门（TPU，20 步，~10 分钟）

跑 20 步（eval 关掉），每步记录并检查：

| 量 | 期望（GB200 20 步实测）|
|---|---|
| 最坏滞后 `staleness_worst/max` | step 1 = 0，之后 ≤ 2（GB200 实测恒 1）|
| `spans/max` | ≤ 3（GB200 实测 ≤ 2）|
| 丢弃组数 | 20 步内 0 |
| IS 权重均值 / 有效样本比例 / 超阈值比例 | ≈ 1.000 / ≥ 0.99 / ≲ 1e-5 |
| `k3_kl`（trainer vs sampler，逐 token）| ≲ 2e-3 |
| 前三次更新的 lr | 0 / 2e-7 / 4e-7 |
| 权重同步 | 每步一次，每次都成功 |

GPU 的 `code/check_smoke.py` 是这些检查在 GPU 日志格式上的实现，每条阈值都有注释，可以照着写 TPU 版。

## 7. 门 5 · Loss 合约自检（推荐，TPU，~30 分钟）

这轮**没有梯度 fixture**（V1 异步 trainer 不产出逐 token logp dump，8K 包的重放工具用不上）。改为直接核合约：随便取一批，把 trainer 用到的逐 token (logπ_θ, logπ_sampler, A, mask) 导出，用 NumPy 按第 1 节公式算 loss，和 trainer 的 loss 在 bf16 误差内相等；分母 = 这一批有效 response token 总数。单前向和两遍实现的等价性（同批梯度余弦 0.99995）在 colocated 轮已验证，对这条损失不变。

## 8. 门 6 · 三条 250 步 run

seed 1、2、3，各 250 步，eval 在 14 个点。GB200 参考（`band/summary.json`）：

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| step 0 → 250 | 0.7066 → 0.8317 | 0.7066 → 0.8271 | 0.7028 → 0.8234 |
| 连续两次 ≥ 0.80 的确认步 | 80 | 100 | 100 |
| 稳态步时（中位 / p90）| 7.16 / 10.07 s | 7.34 / 9.53 s | 7.22 / 10.25 s |
| 端到端 250 步 | 59.4 min | 59.2 min | 59.3 min |
| 丢弃组 / spans=2 的步数 | 11 / 136 | 8 / 138 | 8 / 144 |

band 宽度 median 2.0 pp、p90 2.5 pp、max 2.9 pp；终点 mean 0.8274 [0.8234, 0.8317]。判读：三条 TPU 曲线叠在三条 GPU 曲线上，band 是描述性范围不是判据；落在里面算相容，落在外面是要查的发现；同时报告门 4 那些异步统计和步时分解（`band/gb200_step_time.png` 的口径：稳态步时排除 eval/ckpt 步；wall clock 从 launch 算；到 0.80 用"连续两次"的确认步）。

## 9. 交付格式（GPU 工具能直接读）

1. TensorBoard 事件文件，tag 与 GPU 一致：`val-core/gsm8k_boxed_test/acc/mean@1`、`critic/score/mean`、`response_length/mean`、`response_length/clip_ratio`、`actor/entropy_loss`、`actor/grad_norm`、`actor/rollout_corr/{k3_kl,rollout_is_mean,rollout_is_eff_sample_size,rollout_is_ratio_fraction_high}`、`training/off_policy/trajectory_staleness_worst/max`、`training/off_policy/trajectory_spans/max`、`training/off_policy/evicted_samples`、`timing_s/{step,update_actor,update_weights,gen}`；或每 seed 一个 CSV 含这些列。
2. step 0 和 250 的 eval 逐题 dump（question / output / score）；有条件的话每步 rollout dump（`uid`、`score`、`input`、`output`）。
3. resolved config、软件版本、实际用的异步参数（sync 周期、warmup、阈值、drop）。

## 10. 常见坑（这轮新增的）

- **在训练卡上顺手起 rollout**：不允许（GB200 明确关掉了 hybrid rollout）；训练池和采样池分开。
- **`logπ_sampler` 用 trainer 重算**：那样 IS 权重恒为 1，滞后完全没被修正；必须用 sampler 生成时记录的逐 token 值。
- **按整条回答取一个版本的 logp**：跨版本的回答前后半段来自不同策略，逐 token 才对。
- **阈值/丢弃策略改了**：阈值 3 或"顺延到下一批"都是另一个 recipe，要重跑参考。
- **惩罚用到 eval 上**或 **L 不含结束 token**：门 2 会抓到，但只有你跑了惩罚自检才会。
- **按长度排序/分桶下发 prompt**：会系统性改变 staleness 和批次构成；下发顺序只能是 seeded shuffle。
- **eval 时不等 trainer**：要用被评估的那个 checkpoint 的权重，sampler 上的权重版本要和 step 号对上。
- **`dataloader`/reward 的多进程**：GB200 用单进程读数据、8 个 reward worker；不影响语义，只是记录。

门 1、2 不一致就是实现错误，先发输出再修；门 3、4 的数值超出参考范围，按 rulebook 查原因、把发现写进报告；门 5 是推荐项，做了就附结果。
