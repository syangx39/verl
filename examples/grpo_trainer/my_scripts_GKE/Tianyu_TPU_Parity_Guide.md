# TPU GPU RL Parity 执行指南

Tianyu，这份指南说明如何使用 GB200 交付包，在 TPU 上完成输入、评分、初始质量、数值路径和训练曲线的对齐。目标是用同一份冻结的 `stab_kl0` recipe，比较两套系统计算和学习的结果。

当前 GPU 参考为 Qwen3-0.6B、64 GB200、三个 seed 各 300 步。TPU 对应使用 64 v7x chips，执行模式为 colocated、synchronous、on-policy。本文的“实验第 1–6 步”对应第 3–8 节；第五步 replay 仍为可选项。

## 1 Bucket 目录和使用方式

交付包根目录：

```text
gs://xiaotongyang-bucket/meta-rl/GKE_repro/meta-RL/handoff/stab_kl0/
```

下文未写完整 URI 的文件路径，都相对于这个根目录。下载后可将本地包目录记为 `PARITY_ROOT`。文件身份以交付的 SHA256 为准；转换模型格式或数据布局时，保留来源哈希和转换记录。

| 目录或文件 | 放什么 | 什么时候使用 |
| --- | --- | --- |
| `model/` | `Qwen3-0.6B/` 内为实际使用的初始权重、model/generation config、tokenizer 和模板；另有 `SHA256`、身份记录及导出的 `chat_template.jinja` | 实验第 1、3、4、5、6 步；所有新训练都从这里的初始权重开始 |
| `data/` | 原始训练集、固定评估集、逐 seed 的训练顺序和逐步 row-ID manifest；另有数据校验和、过滤及重叠检查记录 | 第 3 步评估、第 4 步内部 rollout 检查、第 6 步训练 |
| `fixtures/` | 小规模固定输入及 GPU 预期结果：prompt、scorer、log-prob；`raw/` 内为完整固定训练 batch；另有 replay 标量参考和故障测试 | 第 1、2、4、5 步，用于逐层定位差异 |
| `code/` | GPU launcher、共享 scorer、画图/配对检验工具、fixture 构建工具和部署 patch 等 | 核对计算语义、复用评分及结果分析。TPU 使用自己的训练栈，无需运行 GPU launcher |
| `env/` | GPU 软件版本和容器记录，如 `pip_versions.txt`、`container_image.txt` | 核对 scorer 依赖、解释实现差异；GPU 与 TPU 不要求使用同一容器 |
| `runs/` | 三个 GPU run 的配置、日志、TensorBoard、逐题 `val_dump/`、rollout 汇总和原始数据位置；另有启动命令、精度及时间记录 | 第 3 步逐题比较、第 6 步曲线和诊断比较 |
| `band/` | `gb200_band_stab_kl0.json`、`.png`、band 数值表及逐 seed 分析 | 第 3 步读取初始均值，第 6 步读取每个 eval step 的 GPU min/max |
| `checkpoints/` | 三个 GPU seed 的 step-300 HF 导出，以及 `seed1_step1_after_one_update/`；附 `SHA256` | step-300 用于训练后评估复核；step-1 用于第五步更新量比较，均不是初始模型 |
| `PACKAGE_MANIFEST.sha256` | 包内文件的校验和清单 | 下载完整包后，在包根目录执行 `sha256sum -c PACKAGE_MANIFEST.sha256` |

**训练数据的两个用途要区分：** `data/train_qsplit.parquet` 是原训练数据；正式 parity 使用 `data/train_order_seed{k}.parquet`。后者每个 seed 有 76,800 行，已经按 300 步 × 256 prompts 分好 batch。行 ID 唯一，但不同训练行可能属于同一道题，不能按题目文本去重。

**两种 dump 也要区分：**

- `runs/<GPU_EXP>/val_dump/{step}.jsonl` 是评估结果：每次 2,319 题、每题一个 greedy 回答，共 31 次评估。
- 完整训练 rollout dump 每步有 2,048 条回答，每个 seed 约 10 GB，按需读取。位置在包外：`gs://xiaotongyang-bucket/meta-rl/GKE_repro/meta-RL/logs/<GPU_EXP>/rollout_dump/`。准确地址也记录于 `runs/<GPU_EXP>/rollout_dump_location.txt`。

GPU run 名称与 seed 对应如下；下文的 `<GPU_EXP>` 用这里的完整名称替换。

| Seed | GPU_EXP |
| --- | --- |
| 1 | `qwen3_0p6b_phase0_stab_kl0_seed1_16n64g_20260913_0844` |
| 2 | `qwen3_0p6b_phase0_stab_kl0_seed2_16n64g_20260913_1914` |
| 3 | `qwen3_0p6b_phase0_stab_kl0_seed3_16n64g_20260913_2333` |

共享 Rulebook 定义冻结语义，本指南说明执行方法。开始前记录所用 Rulebook 版本和包 manifest。第 4、5 步尚未填写的数值容差，需要双方在查看对应 TPU 结果前书面冻结；缺少配置或源码证据时，由 GPU 侧补齐，不用框架默认值代替。

## 2 Alignment sequence Overview

| 实验步骤 | 检查的问题 | 要做什么 | 是否更新模型 |
| --- | --- | --- | --- |
| 1. Prompt fixture | 同一道题是否变成相同输入？ | 用 TPU 正式预处理流程生成 15 条 prompt 的 token IDs，与 GPU 逐 token 比较 | 否；不需要加载权重 |
| 2. Scorer fixture | 同一个回答是否获得相同 reward？ | 重放 797 条固定评分输入，再检查异常、超时、卡死和 worker 恢复 | 否；不需要模型生成 |
| 3. Step-0 evaluation | 训练前的起点是否一致？ | 初始模型对全部 2,319 道评估题做 greedy evaluation | 否 |
| 4. Trainer-vs-sampler numerics | 概率计算是否一致？ | 固定 tokens 比 GPU/TPU trainer；再用 TPU 自己的 rollouts 比 sampler/trainer | 否 |
| 5. Single-step replay（可选） | 同一份训练 batch 是否得到接近的更新？ | 用完整 GPU 固定 batch 比 advantages、loss、grad norm 和一次更新的 Δθ | 是，仅一次 |
| 6. Training comparison | 完整训练的学习行为是否一致？ | TPU 三个 seed 各跑 300 步，与 GPU 全部曲线及诊断比较 | 是，每 seed 300 次 |

前四步通过后进入正式训练；第五步建议在长跑前完成，若跳过则记录范围。第五步或第六步都从初始权重和全新 optimizer 状态独立开始，不能沿用前面测试更新过的模型。

每一步保存输入身份、实际配置、结果和判定。状态统一使用：`PASS` 满足已冻结标准；`FAIL` 已执行但未达标；`INCOMPLETE` 数据或执行不完整；`PENDING` 容差尚未冻结；`NOT_RUN` 可选实验未执行；`NOT_TESTED` 当前材料不支持的子项。

## 3 实验第一步 Prompt fixture

**目的：确认 TPU 正式训练看到的有效 prompt tokens 与 GPU 完全相同。**

### 3.1 需要哪些文件

| 文件 | 用途 |
| --- | --- |
| `fixtures/prompt_fixture.json` | 15 条输入的 `messages`、GPU `rendered_text`、完整 `token_ids` 和长度 |
| `model/Qwen3-0.6B/` | 实际 tokenizer、配置和 chat template |
| `model/chat_template.jinja`、`model/SHA256` | 模板及 tokenizer 身份核对 |

### 3.2 具体做法

1. 读取 fixture 的 15 条 `messages`，通过准备用于 TPU 正式训练的预处理路径渲染和 tokenize。
2. 使用相同 chat template、thinking 设置和 `add_generation_prompt=True`。保持 GPU 的 `add_special_tokens=False` 行为，不额外加入 BOS。
3. 按有效 mask 去掉 batch padding，与每条 GPU `token_ids` 比较完整序列，同时检查长度。允许不同 padding/packing 布局，但记录布局及 position-ID 处理方式，供第四步复查。
4. 保存逐条报告：`source/index`、双方长度、是否一致；不一致时记录第一个差异的位置及两个 token ID。

这份输入有两个需要保留的细节：user content 中已有 Gemma 风格的 `<start_of_turn>` 等文本，不能被删除、重新解析或替换；外层仍使用交付的 Qwen chat template。GPU 没有通过 `enable_thinking=False` 注入空 `<think>…</think>`，prompt 以 assistant 前缀结尾，fixture 尾部为 `[151644, 77091, 198]`。头尾 ID 只用于定位，不能代替完整比较。

### 3.3 通过标准

- **15/15 条完整有效 token 序列逐元素相同，长度相同。**
- 使用的确实是 TPU 正式预处理路径，模板、tokenizer 身份和 padding/position 记录完整。
- 任意一条有差异即不通过；不能只核对前后几个 token，或删除失败样例后统计。

## 4 实验第二步 Scorer fixture

**目的：确认同一个回答、标准答案和指定长度，在两边得到相同评分，并验证故障处理可以恢复。**

### 4.1 需要哪些文件

| 文件 | 用途 |
| --- | --- |
| `fixtures/scorer_fixture.jsonl` | 797 条固定评分输入和 GPU expected；GPU 已重生成，expected 包含全部实测 `mv_*` flags |
| `fixtures/scorer_fixture.v1.jsonl`、`scorer_fixture_regen.log` | 原版备份及本次重生成记录；用于核对旧评分未变 |
| `code/maxtext_math_reward.py` | 冻结的共享评分实现 |
| `fixtures/scorer_fault_injection.py`、`scorer_controlled_tests.md` | 可执行故障测试及预期行为 |
| `fixtures/scorer_fault_injection.log` | 已有 GPU 故障测试记录 |
| `env/pip_versions.txt` 及交付的依赖记录 | 核对 `math-verify==0.9.0` 和 scorer 依赖 |

### 4.2 具体做法

**A. 固定评分配置。** 在导入 scorer 之前设置以下变量；正常 fixture 和正式评分使用同一套配置。

```bash
export REWARD_DUMP_DIR=""
export REWARD_FMT_WEIGHT=0
export REWARD_OVERLONG_BUFFER=1024 REWARD_OVERLONG_PENALTY=1.0
export REWARD_MAX_RESP_LEN=8192
export REWARD_MV_POOL=1 REWARD_MV_PROCS=4 REWARD_MV_TIMEOUT=5
export REWARD_MATH_VERIFY_MAX_CHARS=400
```

当前训练 reward 是 `acc + length_penalty`，`fmt` 只记录。长度 ≤7168 时惩罚为 0，7168–8192 之间线性下降，8192 时为 −1。例如正确回答在指定长度 7680 时应为 `acc=1, length_penalty=-0.5, score=0.5`。

**GPU 正常 fixture 已重生成并验证。** GPU 使用上述固定配置重放原 797 条，保留 `output / gts / response_len` 和顺序；原有 `acc / fmt / length_penalty / score / mv_used` 全部不变，实测 `mv_timeout / mv_exc / mv_lenrej` 全为 0，已写入新版 expected。原版保存在 `scorer_fixture.v1.jsonl`，记录见 `scorer_fixture_regen.log`。

日志中的 SHA256 前缀：原版 `95a9b98d93c821fb`，新版 `71a4a3c67b5f3f3f`。这些前缀只用于辨认版本，完整文件校验仍使用更新后的 manifest。**已完成：新版 fixture 已写入交付目录，`PACKAGE_MANIFEST.sha256` 已重算，日志显示 242 个文件、校验全部 OK。** 据 GPU 侧交付说明，该目录是 bucket 挂载路径；Tianyu 从 GCS 读取后按完整 manifest 校验，确认拿到同一版内容。242 是该次 package 快照的文件数，后续增删文件应重新生成 manifest。Tianyu 应读取包含八个 expected 字段的新版，不能给旧文件的缺失字段直接补默认 0。

这里的全零是这 797 条的实测结果。一般情况下，提取答案过长导致的 `mv_lenrej=1` 仍可能是确定性结果；不能将“任何确定性 fixture 都必须 lenrej=0”作为普遍规则。

**B. 重放全部正常评分输入。** 逐行调用 TPU 正式训练准备使用的 scorer。若直接复用共享模块，`scorer` 即加载后的 `code/maxtext_math_reward.py`：

```python
got = scorer.compute_score(
    "fixture",
    row["output"],
    row["gts"],
    extra_info={"index": 0, "response_len": row["response_len"]},
)
```

这里不需要生成回答。`row` 是 JSONL 中的一条记录；直接把它的文本、答案和长度传入，比较 `got` 与 `row["expected"]`。保存 `acc / fmt / length_penalty / score / mv_used` 及全部 `mv_*` flags。若使用 TPU 自写实现，就调用该实现，输入和 expected 保持不变。

**直接使用 fixture 的 `response_len`，不要重新 tokenize 回答计算长度。** 这是指定长度的评分测试，包含惩罚边界；不代表原始 rollout 的生成长度被完整重放。

**C. 运行故障测试。** 在上述环境变量生效后运行，并保留完整输出。`PARITY_ROOT` 应指向下载后的包目录。

```bash
set -o pipefail
timeout 120s python3 "$PARITY_ROOT/fixtures/scorer_fault_injection.py" \
  "$PARITY_ROOT/code/maxtext_math_reward.py" \
  2>&1 | tee scorer_fault_injection_tpu.log
test_rc=${PIPESTATUS[0]}
echo "fault-injection exit=$test_rc"
```

测试覆盖 verifier exception、timeout 分类、真正 hang、dead worker 及恢复。故障返回 `acc=0`，正确标记 `mv_exc` 或 `mv_timeout`，保留长度惩罚；长度为 8192 的评分失败样本应为 `score=-1`。真正 hang 必须被 watchdog 结束，worker 被替换，后续正常调用成功。故障注入本身可能产生预期的异常栈，判定以全部测试结果和退出码为准。

该脚本针对共享 scorer 的 worker 实现；若 TPU 更换了 scorer/worker 实现，应做等效故障注入，不能只测试一个未接入正式训练的模块。

**D. 检查正式接入路径。** 本单轮任务的 `response_len` 定义为 response 区间有效 mask 的和，对应 verl 的 `valid_response_length`：**包含实际生成且有效的终止 EOS/stop token，不含 prompt 和 padding。** 即使解码后的 `output` 文本去掉了特殊 token，长度仍按原始 token/mask 计算，不能用字符数或重新 tokenize 的长度替代。

达到长度上限时 `response_len=8192`，不人为追加或用 EOS 替换最后一个 token。若第 8192 个 token 恰好是有效 EOS，它仍计入长度；所以长度等于 8192 不必然表示“没有 EOS”。`151643` 也可能用于 padding，必须依据 mask 区分，不能按 token ID 把它全部删除。两边记录实际最后一个有效 token 和停止原因，区分“长度达到 cap”与“因长度上限截断”。长度定义同样用于 reward、loss mask 和第四步 log-prob 统计。

**返回材料：** 全部 797 条 expected/got 比较结果、完整 `mv_*`、实际配置、fault-injection 日志和退出码，以及正式路径传入长度的核对记录。

### 4.3 通过标准

- **797/797 条全部 expected 字段匹配**；数值使用 `rtol=0, atol=1e-9`，离散 flags 完全相同。expected 至少包含 `acc / fmt / length_penalty / score / mv_used / mv_timeout / mv_exc / mv_lenrej`。必须检查总条数和字段完整性，不能仅断言“至少有一条通过”。
- 每条正常样本的全部 flags 与 GPU 实测 expected 一致；**当前新版 797 条的 `mv_timeout / mv_exc / mv_lenrej` 均为 0，因此 TPU 对这版输入也应全为 0**。保存全部 flags。若下载到 expected 尚未补齐的旧文件，记录正常 fixture 验收为 `INCOMPLETE`，先获取新版，不能用默认 0 补缺失字段。
- 故障测试全部通过，日志为 `RESULT: PASS`、退出码为 0；包含真实 hang/watchdog 和恢复检查，不能只抛出 TimeoutException。
- TPU 正式调用传入正确的有效生成长度。正常评分与故障处理两部分均通过，才记录 `STEP 2: PASS`。

## 5 实验第三步 Step 0 evaluation

**目的：确认任何训练发生之前，TPU 与 GPU 的初始答案正确率处于约定范围。**

### 5.1 需要哪些文件

| 文件 | 用途 |
| --- | --- |
| `model/Qwen3-0.6B/`、`model/SHA256` | 同一份初始权重和配置 |
| `model/Qwen3-0.6B/generation_config.json`、`tokenizer_config.json` 及 `runs/<GPU_EXP>/` 的实际引擎配置/日志 | 核对 token ID 映射、最终生效的 stop 集合和 EOS 处理，不只看模型默认值 |
| `fixtures/stop_token_evidence.txt` | 固定 batch 的末 token 统计、seed1 的 `ignore_eos` 及交付模型 EOS 配置 |
| `data/gsm8k_test.parquet` | 完整 GSM8K test，1,319 题 |
| `data/val_1k_qsplit.parquet` | OMI2 held-out evaluation，1,000 题 |
| `code/maxtext_math_reward.py` | 第二步已验证的 scorer |
| `runs/<GPU_EXP>/val_dump/0.jsonl` | 三个 GPU seed 的初始逐题结果 |
| `band/gb200_band_stab_kl0.json` | GPU step-0 精确数值和均值 |

### 5.2 具体做法

1. 从交付的初始模型加载权重；如需转换成 MaxText 格式，记录来源哈希和参数映射。评估按 rollout 路径使用 BF16 权重。此步骤不用 step-1 或 step-300 checkpoint。
2. 对两个完整数据集各题生成 **一个 greedy 回答**：`do_sample=False, n=1`，response cap 8192。使用第一步对齐的 prompt、tokenizer、thinking 设置，并按下方显式 EOS/stop 约定配置引擎。
3. 用第二步已验证的 scorer 评分，分别计算 GSM8K 和 OMI2 的平均 `acc`。`score` 包含长度惩罚，不能用它代替 accuracy。
4. 导出逐题结果，按数据来源及稳定题目身份与 GPU 配对；不能只靠 JSONL 行号。GPU 的 `val_dump/0.jsonl` 是训练前结果，后面的 `10.jsonl`、`20.jsonl` 等是对应更新完成后的评估。

**EOS/stop 约定及运行证据。** Qwen3-0.6B 的预期 EOS/stop 集合为：

```text
<|im_end|>    = 151645
<|endoftext|> = 151643
```

配置语义是遇到任一有效生成的终止 token 即停止；不忽略 EOS，不在 response cap 8192 后追加 EOS，也不强制把最后一个生成 token 改成 EOS。长度及有效 EOS 的计数按 §4.2 D 执行。单纯达到 8192 与因上限截断分别记录；若 EOS 恰好出现在最后一个位置，仍保留它。

**已有 GPU 运行证据，范围如下。** `fixtures/stop_token_evidence.txt` 记录：

| 证据来源 | 已观察到的结果 |
| --- | --- |
| `fixtures/raw/fixture_step1.npz` 中长度小于 8192 的 1,460 条回答 | 最后一个有效 token 为 `151645` 的有 1,458 条，为 `151643` 的有 2 条，无其他末 token |
| 同一固定 batch 中另外 588 条回答 | 有效长度均为 8192，后续检查确认末 token 为 EOS 的有 0 条；本 batch 没有恰在第 8192 位以 EOS 结束的样本，符合长度上限截断的表现 |
| seed1 / seed2 / seed3 的 resolved config，以及 fixture job 日志 | 四者均记录 `ignore_eos=False`；各自出处见 `fixtures/logprobs_mode_evidence.txt` |
| seed1 / seed2 / seed3 的 resolved config | 三者均包含训练采样 `do_sample=True, temperature=1.0`，以及评估 `do_sample=False, temperature=0` |
| seed1 resolved config 的 `val_kwargs` | `do_sample=False, n=1, temperature=0, top_k=-1`，确认 greedy eval 参数 |
| 交付模型的 `generation_config.json` | `eos_token_id=[151645,151643]` |

**已核实的范围已覆盖交付模型、三个参考 run、fixture job 和上述 greedy eval 参数。** 两个 EOS ID 为 `[151645,151643]`，四个 job 均不忽略 EOS；固定 batch 的短回答以这两个 ID 结束，全部 588 条达到 cap 的回答末 token 均不是 EOS，因此本 batch 未观察到 cap 位置追加或替换为 EOS 的行为。保留通用边界规则：其他 batch 仍可能恰好在第 8192 位生成 EOS。

`val_kwargs` 证明评估参数，不单独证明训练与评估使用同一个物理引擎实例；grep 未输出 `stop` 或 `min_tokens` 也不能证明不存在默认值或请求级覆盖。若要断言无额外 stop、无强制 EOS 及完整参数继承关系，应由部署源码或最终请求参数佐证。TPU 按上述明确的 EOS/长度约定实现并记录实际配置，不以“共用实例”作为额外硬性要求。此前 OSL 差异仍只作为排查线索，不能据此断定 stop 设置就是原因。

**可选诊断：HF → MaxText → HF 往返转换。** 当 Step-0 存在无法解释的差异时，可在同一 vLLM-TPU 版本、同 dtype、同评估输入和协议下，分别评估原始 HF 权重及往返转换后的 HF 权重，保留逐题差异。先比较正确参数映射下的权重；如有需要，再比较固定 tokens 的 teacher-forced log-prob。

这用于定位转换链路对结果的影响，不把 accuracy 差直接称作“转换保真度”：accuracy 相同不证明权重无损，往返转换也可能抵消错误，不能验证原生 MaxText forward。本诊断保持可选，不替代实验第四步的实际 trainer 比较，也不新增 Step-0 数值门槛。

**返回材料：** 两个数据集的 accuracy、逐题输出/评分、与 GPU 各 seed 的逐题正确性一致率、模型哈希和实际评估配置，包括最终生效的 stop 集合及来源。逐题记录至少保留 `data_source`、可稳定配对的题目身份、`input / output / gts / acc / fmt / score / length_penalty / mv_*`，并保存原始有效 response 长度、末 token 和停止原因以便核查。

### 5.3 通过标准

- 初始模型身份正确，未经 optimizer update；评估配置已对齐，最终生效的 stop 集合及 EOS/长度处理有实际运行证据，不能只引用默认配置。
- 完整覆盖 **1,319 个 GSM8K 唯一题目和 1,000 个 OMI2 唯一题目**，无漏题、重复或跨数据集错配。
- 两个数据集分别满足下式，使用交付结果中的未四舍五入数值计算：

```text
abs(acc_TPU − mean(acc_GPU_seed1, acc_GPU_seed2, acc_GPU_seed3)) <= 0.01
```

| 数据集 | GPU 初始均值展示约数 | 允许偏差 |
| --- | --- | --- |
| GSM8K | 71.9% | 精确均值 ±1.0 个百分点 |
| OMI2 | 36.9% | 精确均值 ±1.0 个百分点 |

两个数据集都达标且逐题报告完整，才记录 `STEP 3: PASS`。正确性一致率需要报告，但不另设尚未约定的数值门槛。

## 6 实验第四步 Trainer vs sampler numerics

**目的：分别检查 GPU/TPU trainer 的数值差异，以及 TPU 内部 sampler/trainer 的数值差异。整个步骤不更新权重。**

### 6.1 需要哪些文件

| 文件 | 用途 |
| --- | --- |
| `fixtures/logprob_fixture.json` | 96 条固定序列，GPU 每 token `logp_trainer / logp_sampler / logp_trainer_repeat` 和统计值 |
| `fixtures/raw/fixture_step1.npz`、`.json` | 完整 2,048 条 batch、完整 masks/positions，以及采集配置、模型哈希和环境 |
| `fixtures/raw/fixture_seed1_official.log` | GPU fixture 采集过程和实际配置 |
| `fixtures/logprobs_mode_evidence.txt` | 引擎默认值 `raw_logprobs` 与 seed1/2/3、fixture job 的运行配置记录；四者均为 `processed_logprobs` |
| `model/Qwen3-0.6B/`、`model/SHA256` | 同一初始模型 |
| `data/train_order_seed1.parquet`、`data/step_manifest_seed1.json` | TPU 内部 rollout 检查使用的训练 step-1 prompts |

### 6.2 具体做法

**A. 固定 tokens 比较 TPU trainer 与 GPU trainer。**

读取 `logprob_fixture.json` 的 `sequences`，按 `batch_row` 保留身份。直接使用 `prompt_ids + response_ids`，不重新 tokenize 或生成。通过 TPU 正式 trainer 路径做 teacher-forced forward，使用 FP32 master、BF16 compute。

对第 k 个 response token 计算：

```text
logp[k] = log P(response_ids[k] | prompt_ids, response_ids[:k])
```

注意 causal shift：预测当前 token 的 logits 来自前一个位置。使用 T=1、全词表 `log_softmax`、自然对数；只统计 `response_mask` 指定的有效 response tokens，排除 prompt/padding，保留有效 EOS。重做 padding/packing 时保持逻辑 position IDs、序列边界和 attention 隔离。

将 TPU log-prob 与 GPU `logp_trainer` 逐 token 比较。固定相同权重和输入，再执行一次 TPU forward，单独统计重复误差。

**B. 用 TPU 自己的 rollouts 比较 sampler 与 trainer。**

1. 使用 `train_order_seed1.parquet` 前 256 行，即**训练 step 1**，核对 manifest；不是第一步 Prompt fixture 的 15 条输入。
2. 初始模型按 `T=1, top_p=1, top_k=disabled, n=8, response cap=8192` 生成，共 2,048 条。
3. 保存 sampler 对实际生成 token 返回的**实际采样分布下的 log-prob**，使用自然对数且与对应生成位置一一对齐。显式记录实际 `logprobs_mode` 或引擎等效配置；返回的是归一化后的 log-prob，不是 logits。当前 recipe 要核实 T=1、全词表采样且没有有效 processor 改变分布；在此前提下，raw 或 processed 都可满足同一概率语义，不要求两边模式名称相同。
4. 保持同一 policy checkpoint，**任何 optimizer update 之前**，由 TPU trainer 对这批相同 tokens 重算 log-prob，按同一 mask 比较。

TPU 无需生成与 GPU 完全相同的随机回答。A 固定跨硬件输入；B 固定 TPU 内部两条路径的输入。

**固定 fixture 与三个参考 run 的模式已核实一致：均配置为 `processed_logprobs`。** `fixtures/logprobs_mode_evidence.txt` 已收录 seed1/2/3 的 resolved config，以及 `fixtures/raw/fixture_seed1_official.log` 中的对应记录；四者也均记录 `ignore_eos=False`。三个参考 run 的训练 temperature 为 1.0；fixture job 的 T=1、top_p=1、top_k=-1 另见其已记录的 recipe 行。固定 fixture 的模式不再是待核对项。

当前 vLLM `0.20.2rc1.dev49+g9b4e83934` 的 ModelConfig 默认虽为 `raw_logprobs`，实际 run 配置已显式选择 processed。0.0057 等历史基线和 96 条 fixture 的测量值继续保留，并注明 sampler 返回模式为 processed；不能以框架默认值改写这些记录。

**等价条件：确认所有有效 processors，而不只确认 T/top-k/top-p。** T=1、top_p=1、top_k disabled，且所有 logits processors 均不改变分布时，raw 与 processed 在数学上对应相同概率；这不保证不同计算路径的数值逐位相同。核对 repetition/presence/frequency penalties、min-tokens EOS 屏蔽、logit bias、bad-words/allowed-token 限制及 grammar 等，保存最终生效配置和实际返回行为的验证记录。模式含义可参见 [vLLM ModelConfig 文档](https://docs.vllm.ai/en/latest/api/vllm/config/model/#vllm.config.model.ModelConfig.logprobs_mode)。

模式一致已由上述日志确认；“没有任何有效 processor 改变分布”是另一项语义条件，不能仅由 `processed_logprobs`、T=1 或 grep 未命中其他字段推导。以最终配置和部署路径为依据核对该条件后，再据此解释 processed 与 trainer 原始全词表概率的对应关系。

“完整词表归一化”描述概率的计算方式；只需保存实际生成 token 的 log-prob，无需导出每个位置的整个词表概率。

若发现有效的分布修改，先记录并排查其是否属于冻结 recipe。不能为了降低 log-prob 差异而静默修改 GPU 参考语义或 TPU 采样配置。

**C. 统一统计口径。** A 中令 `d = logp_TPU_trainer − logp_GPU_trainer`；B 中令 `d = logp_TPU_trainer − logp_TPU_sampler`。

| 指标 | 定义 | 单位 |
| --- | --- | --- |
| Probability MAE | 所有有效 tokens 的 `mean(abs(exp(logp_A) − exp(logp_B)))` | 无单位 |
| 绝对 log-prob 误差 | 所有有效 tokens 的 `abs(d)`，报告 mean、p95、p99、max | nats |
| Signed mean | `mean(d)`；用于观察偏置，正负可能抵消 | nats |
| Trainer repeat error | 同权重同输入两次 TPU forward 的绝对 log-prob 差 | nats |

概率绝对差也报告 p95、p99、max。汇总时按有效 token 加权，不先对每条回答求均值再等权平均。另报 response-length 分布、长回答/截断分层、最大误差对应的序列和 token 位置。

GPU 的 **96 条固定 fixture** 共 429,638 个有效 response tokens，trainer-vs-sampler 基线如下：

| 指标 | GPU 测量值 |
| --- | --- |
| Probability MAE | 0.005662 |
| Mean abs(Δlogp) | 0.017295 nats |
| Abs(Δlogp) p95 / p99 / max | 0.093496 / 0.150547 / 0.859093 nats |
| Signed mean，trainer − sampler | −0.000801 nats |
| Trainer repeat error | 本次测试测得 0 |

300 步训练日志中 probability MAE 约为 0.0057、`rollout_corr/kl` 约为 0.0007；上述绝对误差及尾部统计来自固定 fixture，不能描述为“在 300 步中一直平稳”。`rollout_corr/kl` 不等于 mean abs(Δlogp)，也不是 reference-model KL。分层抽取的 96 条与 TPU 随机 rollout 的长度分布可能不同，需要结合分层结果解读。

**返回材料：** A 的按 `batch_row` 对齐的 TPU per-token log-prob 和重复结果；B 的 tokens/masks/UID、两套 per-token log-prob；两项统计、长度分布、模型哈希、代码和实际精度配置；sampler 实际 `logprobs_mode`、有效 processor 列表和配置证据。HF FP32 辅助锚点未计算，仅在出现差异时按需补充，不是必需项。

### 6.3 通过标准

先满足输入和执行条件：A 覆盖原 **96/96 条、429,638 个有效 response tokens**，无缺失或重复；B 为 **256 UID × 8 = 2,048 条**，sampler/trainer tokens、mask、checkpoint 和更新时点一致。核实 sampler 返回实际采样分布的 log-prob；在本 recipe 下，完整词表归一化且无有效 processor 改变分布，两边 processor 设置有对应证据。**GPU fixture 的 processed 模式已确认；TPU 的模式标签可为 raw 或 processed，按实际概率语义验收。** 所有有效 log-prob 和比较统计均为有限值，报告采用上述口径；TPU 返回模式或两边归一化/processor 语义未核实时，相关协议检查记录为 `INCOMPLETE`。

随后按事前冻结的数值预算验收：

| 检查 | 需要填写的通过门槛 | 当前状态 |
| --- | --- | --- |
| A：TPU trainer vs GPU trainer | Probability MAE 上限、mean abs(Δlogp) 上限、p99 abs(Δlogp) 上限 | **PENDING** |
| TPU trainer 重复计算 | Mean / max abs(Δlogp_repeat) 上限 | **PENDING** |
| B：TPU trainer vs TPU sampler | Probability MAE、mean abs(Δlogp)、p99 abs(Δlogp) 上限，以及固定的长度分层/比较方式 | **PENDING** |

阈值、统计/分层口径、协议版本和确认日期必须在查看对应 TPU 结果前书面记录。GPU 内部误差不是 GPU↔TPU trainer 的容差；GPU repeat=0 也不要求跨硬件零误差。

对 B，非负误差指标较同口径 GPU 参考高约 10× 时触发调查，结合长度分层检查 temperature、归一化、mask、精度、kernels 和 weight sync。**这是调查触发条件，不是通过门槛**；不用于 signed mean 或零 repeat error。最大 token 误差必须报告，是否设置硬门槛也须事前声明。

A、B、重复检查均满足已冻结预算，且异常调查已关闭，才记录 `STEP 4: PASS`。预算未填时记录 `PENDING`，不能因“看起来接近”宣告通过。

## 7 实验第五步 Single step replay

**目的：用 GPU 保存的同一份完整 batch，在 TPU 上验证 advantages、loss、grad norm 和一次更新的权重变化。此步骤可选，建议在长跑前完成。**

### 7.1 需要哪些文件

| 文件 | 用途 |
| --- | --- |
| `fixtures/raw/fixture_step1.npz`、`.json` | 完整 2,048 条 pre-update batch 和 provenance |
| `fixtures/replay_step1_reference.json` | GPU step-1 loss、grad norm、advantage/reward 汇总 |
| `model/Qwen3-0.6B/` | 初始权重 |
| `checkpoints/seed1_step1_after_one_update/` | GPU 一次更新后的 FP32 权重 |
| `fixtures/raw/fixture_seed1_official.log`、`code/` 中对应实现证据 | fixture job 的配置、loss/optimizer 语义 |
| `checkpoints/SHA256`、`PACKAGE_MANIFEST.sha256` | 文件身份校验 |

### 7.2 具体做法

使用完整 raw batch，不用第四步的 96 条子集。不重新生成、重新评分或剔除样本。当前交付没有 GPU 梯度向量，因此支持完整更新比较，但不支持原始梯度方向比较或“输入相同梯度”的独立 optimizer 验证。

**A. 重算 advantages。** 从 NPZ 读取 `nt__uid`、`token_level_rewards`、`response_mask` 和 GPU `advantages`。每条 completion 的 reward 及 advantage 为：

```text
R_i = sum(token_level_rewards[i])
A_i = (R_i − mean(R_group)) / (std(R_group, correction=1) + 1e-6)
```

确认 256 个 UID、每组 8 条。按 UID 计算组内 sample standard deviation，将 `A_i` 广播到 response tokens 后乘 mask，与保存的 GPU `advantages` 逐元素比较。等 reward 组应为零，padding 为零；报告 mean/max 绝对差及异常行。

**B0. 先做不需要梯度的 loss 聚合预检查。** 把 GPU 保存的 advantages 和 mask 送入 TPU 正式 loss 聚合路径，固定 ratio=1，包括 micro-batch 和跨设备汇总；不执行 backward 或更新。

```text
token_mean_advantage = sum(advantages × response_mask) / sum(response_mask)
                     ≈ −0.11552895605564117
policy_loss_at_ratio_one = −token_mean_advantage
                        ≈ +0.11552895605564117
```

组内按序列平均的 advantage 为零，不代表按 token 平均也为零。分母必须包含所有有效 response tokens，包括零 advantage 组和截断回答。若预检查不一致，先查输入、mask、全局分母，以及是否误用了 sequence-mean 或 micro-batch 等权平均。

**B1. 正式 forward/backward。** 从初始模型计算可求导的 TPU current log-prob，将 GPU 保存的 **`old_log_probs` 和 `advantages` 作为常量**注入。不要用 `rollout_log_probs` 替换 old-policy anchor，也不要用 TPU 重算值覆盖 GPU anchor。

通过真实 current/old log-prob 计算 ratio；**不能把 B0 的固定 ratio=1 带进 backward 或更新**。使用冻结的 GRPO loss、ratio clipping 和 mask，保持全 batch 的全局 token-mean 归一化。所有 micro-batch 梯度累积后，计算 clip 前的 global grad norm。

| GPU 参考标量 | 数值 |
| --- | --- |
| `actor/pg_loss` | +0.11552895605564117 |
| `actor/grad_norm`，全局、累积后、clip 前 | 0.07904955744743347 |
| `critic/advantages/mean`，有效 token 均值 | −0.11552895605564117 |
| `critic/score/mean` | 0.11516809463500977 |

**C. 执行一次完整更新。** 从同一 FP32 初始权重和全新 AdamW 状态开始；moments=0、optimizer step=0。FP32 master 和 moments，BF16 compute；LR=1e-6 constant、无 warmup，betas=(0.9,0.999)、eps=1e-8、weight decay=0.01、global norm clip=1.0。保持相同的参数更新及 weight-decay 范围。

完整 batch 累积后只执行一次 clipping 和一次 optimizer update。GPU 此步 norm 小于 1，未触发梯度缩放。将 TPU 参数转换到相同 HF 参数名称、shape 和布局，以 FP32 计算：

```text
Δθ_GPU = θ_GPU_after − θ_initial
Δθ_TPU = θ_TPU_after − θ_initial
relative_update_error = norm(Δθ_TPU − Δθ_GPU) / norm(Δθ_GPU)
```

报告整体及逐层 update RMS、相对更新误差、非零 update 的 cosine similarity。比较重点是 Δθ，而不是由大部分未变化权重主导的“更新后权重接近程度”。更新方向接近也不证明原始梯度方向已通过验证。

**返回材料：** advantage 逐元素误差；B0 和 B1 的 loss/norm 及差值；FP32 更新后权重或可复查的逐参数更新数据；整体/逐层 Δθ 报告，以及实际精度、loss reduction、optimizer 配置和模型哈希。

### 7.3 通过标准

共同前提是完整 **2,048 条、256 UID × 8** 固定 batch，tokens、mask、rewards、old-policy anchor 和初始权重一致，比较值及误差统计均有限。

| 子项 | 通过条件 | 待冻结预算 |
| --- | --- | --- |
| A Advantages | 每个元素满足 `abs(A_TPU−A_GPU) <= atol_A + rtol_A × abs(A_GPU)`；padding 为零，等 reward 组在约定误差内为零 | `atol_A`、`rtol_A`：**PENDING** |
| B0 聚合预检查 | Advantage token mean 与负参考值、ratio-one policy loss 与正参考值分别满足绝对误差上限 | `epsilon_agg`：**PENDING** |
| B1 实际 loss / grad norm | Policy loss 与 GPU 标量的绝对差达标，clip 前 global grad norm 与 GPU 标量的相对差达标 | Loss 绝对差、norm 相对差上限：**PENDING** |
| C 完整单次更新 | 恰好一次更新；参数映射正确；整体相对 Δθ 误差、非零 update cosine 及逐层更新误差达标 | 整体相对误差上限、cosine 下限、逐层绝对/相对预算：**PENDING** |

逐层可用 `norm(Δθ_TPU_layer−Δθ_GPU_layer) <= atol_layer + rtol_layer × norm(Δθ_GPU_layer)`。GPU 零 update 层使用绝对误差预算，cosine 标 `N/A`，不能除零或静默跳过。

这些预算需在查看对应 TPU 结果前书面冻结。四个子项都通过，可记录 `STEP 5: PASS (available replay checks)`，同时记录 `gradient direction: NOT_TESTED` 和 `isolated optimizer: NOT_TESTED`。标量 loss/norm 接近不能证明梯度方向一致。

若不执行本步骤，记录 `STEP 5: NOT_RUN (optional)`。缺少梯度向量不会自动成为第六步的额外前置条件。

## 8 实验第六步 Training comparison

**目的：用冻结的 stab_kl0 recipe 完成 TPU 三 seed 训练，检查完整学习曲线，并记录同口径性能。开始前实验第 1–4 步应已通过；第五步记录实际覆盖范围。**

### 8.1 需要哪些文件

| 文件 | 用途 |
| --- | --- |
| `model/Qwen3-0.6B/` | 三个 seed 共用的初始模型、tokenizer 和配置 |
| `data/train_order_seed{1,2,3}.parquet` | 对应 GPU seed 的 76,800 行训练顺序 |
| `data/step_manifest_seed{1,2,3}.json` | 每个训练 step 的 256 个 row IDs |
| `data/gsm8k_test.parquet`、`data/val_1k_qsplit.parquet` | 全部评估题 |
| `code/maxtext_math_reward.py`、对应 loss/launcher 实现及冻结 Rulebook | 训练语义和评分配置 |
| `band/gb200_band_stab_kl0.json`、`.png` | GPU 逐 eval step 的精确 min/max、均值和曲线 |
| `runs/<GPU_EXP>/` | GPU 配置、训练诊断、TensorBoard 和逐题评估参考 |
| `code/band_plot.py`、`paired_eval_bootstrap.py` | 曲线和逐题配对分析工具 |

### 8.2 具体做法

**A. 固定 recipe，独立启动三个 seed。** 每个 seed 从同一初始模型和全新 optimizer 状态开始，使用独立输出目录，不从 replay 或其他 seed 的 checkpoint 恢复。

| 项目 | 固定设置 |
| --- | --- |
| 资源与执行模式 | 64 TPU v7x chips；colocated、synchronous、on-policy |
| 训练长度 | seeds 1、2、3，各 300 optimizer steps |
| 每步工作量 | 256 个训练 rows × 8 completions = 2,048 条；全 batch 累积后一次更新，μ=1 |
| 精度 | FP32 master/Adam moments；BF16 compute、rollout weights、KV cache |
| AdamW | LR 1e-6 constant、无 warmup；betas (0.9,0.999)、eps 1e-8、weight decay 0.01 |
| Gradient clip | 全 batch 累积后，global norm clip 1.0 |
| Sampling | T=1、top_p=1、top_k=disabled、n=8 |
| Reference KL / entropy loss | 均为 0 |
| Reward | Answer accuracy + overlong soft penalty；buffer 1024、penalty 1.0、cap 8192；fmt_w=0，保留 fmt 日志 |
| 长度 | Prompt limit 8192，过滤超长 prompts；response cap 8192 |

**训练 rollout 和 greedy eval 使用同一份明确记录的 EOS/stop 约定。** 交付模型 EOS IDs 为 `[151645,151643]`，三个参考 run 及 fixture job 均记录 `ignore_eos=False`，固定 batch 的末 token 和 cap 样本检查见 §5.2。任一有效生成的 EOS 即停；不因 cap 追加或强制替换 EOS；将最终生效集合写入 TPU resolved config。`response_len` 含有效 EOS、不含 prompt/padding。GPU 三个参考 run 及 fixture job 均记录为 processed（见 §6.2 B），不强制 TPU 使用 raw 标签；两边核对有效 processors，TPU 记录自己实际生效的模式及采样配置。

GRPO advantage、ratio clipping、old-policy log-prob、loss mask 和全局 token-mean 分母沿用冻结 Rulebook 及已对齐实现。保留截断回答和零 advantage 组的有效 tokens；不能额外加 truncated-sample masking、重采样或训练目标。

**B. 按 GPU 的逐步 prompt 成员训练。** Seed k 顺序读取 `train_order_seed{k}.parquet`，关闭所有额外 shuffle。第 s 步读取第 `(s−1)×256` 到 `s×256−1` 行，并核对全局 row-ID 集合与 manifest 一致，防止 filtering/sharding 改变 batch。

保留数据行的重复题目关系；每个 prompt 实例独立 UID，每 UID 8 条回答。单步内可以重排/packing，但不能改变分组或全局 loss 权重。每个 seed 的数据顺序及采样随机性均需记录。

每步由 **TPU 当前策略重新生成回答**，不使用 GPU 回答做正式训练。收齐完整 batch、评分、更新及 weight sync 后再开始下一步 rollout；允许同一步内部并发调度和评分，不引入跨步 policy lag，不只保留最快完成的样本。

**C. 每 10 步评估。** 在 `0, 10, 20, …, 300` 共 31 个时点，按第三步相同协议评估全部 2,319 题。Step 0 在任何更新前，其余使用对应更新完成后的模型。

分别记录两个数据集的 answer accuracy、actual reward、fmt、length penalty、overlong 和 scorer flags；accuracy 是主要质量指标。每次保留逐题 JSONL，不挑最佳 checkpoint。至少兼容下列 scalar 名称：

```text
val-core/gsm8k/acc/mean@1
val-core/omi2_val1k/acc/mean@1
val-aux/<dataset>/fmt/mean@1
val-aux/<dataset>/length_penalty/mean@1
critic/score/mean
actor/entropy
actor/grad_norm
response_length/mean
response_length/clip_ratio
```

**D. 保留训练诊断。** 每步记录 train acc、实际 reward、fmt、length penalty；按 UID 和实际 reward 算的 `frac_zero_std`；response length、cap-hit、有效 token 数；entropy、clip 前 global grad norm、实际 LR、policy loss；第四步定义的 trainer/sampler 误差；`mv_timeout / mv_exc / mv_lenrej`；各阶段和完整 step time。

`frac_zero_std` 不能用 solve_all + solve_none 代替，因为长度惩罚也会产生组内差异。保留逐步 rollout dump，含 UID、row ID、回答、有效生成长度、最后一个有效 token、停止原因、各评分字段和 flags；每步可验证 2,048 条、256 UID × 8。`response_length/clip_ratio` 继续按长度达到 cap 的既有口径比较，另外记录因长度限制终止的比例，不能把两者自动等同；EOS 恰在第 8192 位时需保留这一边界情况。运行约定的 collapse guard，保存触发原因和退出状态。

**E. 记录可比的性能。** 在查看性能结果前约定双方使用的训练 step 窗口和计时边界；当前具体窗口尚待确定。记录 steady-state end-to-end step time，排除 warmup/compilation、evaluation、checkpoint save；包含关键路径上的生成、必要 log-prob、reward、训练、weight sync 和协调开销。

单列 diagnostic dump 开销，保持 instrumentation 可比；同时报告 token 总数、长度分布和 cap-hit。GPU 约 45 秒的现有统计来自训练过程，包含约 2.8 秒 dump 且生成长度随训练变化，不能直接当作无 instrumentation 的固定负载性能。各阶段 median 不能简单相加解释总 median；旧 recipe 的 20-step 数字也不能并入本次质量 parity 结论。

**返回材料：** 每个 seed 的实际 launch command、完整 resolved config、代码/环境/模型身份；全部训练和 guard 日志；TensorBoard/scalars；31 份逐题 eval JSONL，每份 2,319 行；逐步 rollout 数据；step-300 checkpoint、性能摘要和校验和。评估记录保留 `input / output / gts / score / acc / fmt / length_penalty`、数据来源和稳定题目身份；格式不同则附转换器。

### 8.3 通过标准

**质量逐 seed、逐数据集验收。** 先确认三个 seed 均完成 300 步，逐步 batch 成员、分组和冻结语义正确；全部 31 次评估各覆盖 1,319 个 GSM8K、1,000 个 OMI2 唯一题目，无漏题或重复。

对每个 TPU seed、每个数据集，在同一 eval step 使用 GPU 三 seed 的 min/max：

```text
允许范围 = [GPU_min(step) − 0.015, GPU_max(step) + 0.015]
```

- **31 个评估点最多允许 2 个超出该范围。** 上下界都检查，明显高于参考带也需排查语义。
- **Step 0** 仍须单独满足第三步的 GPU 精确均值 ±1.0 个百分点，不能被更宽的曲线范围代替。
- **Step 300 必须在下表内，没有例外。**

| 数据集 | Step-300 允许范围 |
| --- | --- |
| GSM8K | [0.773, 0.823] |
| OMI2 | [0.529, 0.569] |

三个 TPU seed 的两个数据集都满足以上条件，才记为 `STEP 6 QUALITY: PASS`。不能用三 seed 平均值掩盖单个 seed 失败。已执行但越界超限或 collapse 的 run 不通过；缺少评估、未完成训练等记录为 `INCOMPLETE`，不能挑剩余结果宣告通过。

GPU 三 seed 的 min–max band 是经验范围，不是置信区间。逐题 paired bootstrap 是补充分析，不替代上述规则；单项诊断超出 GPU seed spread 也不自动判失败，但需解释。

**性能单独判断测量是否有效。** Recipe、资源口径、事前窗口、计时边界一致，排除项、dump 开销、tokens/长度分布及原始证据完整时，记为 `PERF MEASUREMENT: VALID`。尚未约定 TPU 必须达到的 step time 或速度比，不新增速度门槛；窗口或测量证据未齐时记为 `PERF: PENDING`，单独报告质量结论。

最终报告展示全部 GPU/TPU seed 曲线、各 seed 的 step-0→300 增益、逐数据集越界点及最终点判定，并附诊断差异和同口径性能结果。
