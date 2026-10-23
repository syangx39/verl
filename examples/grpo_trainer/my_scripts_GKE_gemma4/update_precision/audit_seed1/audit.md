### Learning-signal / eval-pairing / grading audit: `/workspace/meta-RL/logs/gemma4_eval/gemma4_e2b_rl05_16n64g_s200_seed1_1021_1955`

- Training dumps: 200 steps; groups per step {256: 200}; group sizes {8: 51200}; groups with mixed input or ground truth: 0.

| Window | Steps | Groups | Train acc (T=0.8) | mean group std (score / acc) | all-same correct | all-same wrong | format-only (correct) | format-only (wrong) | correctness-mixed | sum|A| share: format-only | sum|A| share: correctness-mixed | length-weighted share: format-only |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| early | 1–20 | 5120 | 59.27% | 0.124 / 0.121 | 44.1% | 22.1% | 1.3% | 4.9% | **27.7%** | 16.4% | 83.6% | 19.7% |
| mid | 91–110 | 5120 | 59.16% | 0.121 / 0.120 | 45.3% | 24.6% | 0.5% | 2.5% | **27.1%** | 8.6% | 91.4% | 12.1% |
| late | 181–200 | 5120 | 61.31% | 0.119 / 0.118 | 47.8% | 23.4% | 0.3% | 1.7% | **26.9%** | 5.9% | 94.1% | 7.9% |
| all | 1–200 | 51200 | 59.91% | 0.121 / 0.120 | 45.8% | 23.7% | 0.6% | 2.7% | **27.3%** | 9.4% | 90.6% | 12.4% |

- Eval identity: 5000 unique qids per file; qid set, input and ground-truth hashes identical across checkpoints [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200] and the replicate run; all pairing below is by qid.
- Across the 11 checkpoints: never solved 45.0%, always solved 23.0%, sometimes 32.0%.

| Comparison | Acc A → B (%) | Δ (pp) | paired SE (pp) | bootstrap 95% (pp) | +gained / −lost | identical outputs |
|---|---|---:|---:|---|---|---:|
| step 0 → 20 (same run) | 36.26 → 36.34 | +0.08 | 0.48 | [-0.82, +1.00] | +290 / −286 | 256 / 5000 |
| step 0 → 40 (same run) | 36.26 → 36.56 | +0.30 | 0.49 | [-0.62, +1.30] | +303 / −288 | 195 / 5000 |
| step 0 → 60 (same run) | 36.26 → 37.32 | +1.06 | 0.48 | [+0.16, +1.96] | +317 / −264 | 189 / 5000 |
| step 0 → 80 (same run) | 36.26 → 37.16 | +0.90 | 0.48 | [-0.04, +1.86] | +309 / −264 | 194 / 5000 |
| step 0 → 100 (same run) | 36.26 → 36.92 | +0.66 | 0.48 | [-0.30, +1.62] | +304 / −271 | 175 / 5000 |
| step 0 → 120 (same run) | 36.26 → 37.50 | +1.24 | 0.49 | [+0.26, +2.16] | +333 / −271 | 163 / 5000 |
| step 0 → 140 (same run) | 36.26 → 36.68 | +0.42 | 0.49 | [-0.54, +1.38] | +308 / −287 | 153 / 5000 |
| step 0 → 160 (same run) | 36.26 → 37.22 | +0.96 | 0.48 | [-0.02, +1.90] | +318 / −270 | 144 / 5000 |
| step 0 → 180 (same run) | 36.26 → 37.34 | +1.08 | 0.48 | [+0.12, +2.04] | +315 / −261 | 152 / 5000 |
| step 0 → 200 (same run) | 36.26 → 36.80 | +0.54 | 0.49 | [-0.48, +1.48] | +309 / −282 | 135 / 5000 |
| run vs replicate, step 0 (identical weights at step 0) | 36.26 → 36.74 | +0.48 | 0.48 | [-0.40, +1.40] | +305 / −281 | 288 / 5000 |
| run vs replicate, step 20 (identical weights at step 0) | 36.34 → 36.12 | -0.22 | 0.47 | [-1.10, +0.74] | +267 / −278 | 268 / 5000 |
| run vs replicate, step 40 (identical weights at step 0) | 36.56 → 36.26 | -0.30 | 0.47 | [-1.24, +0.60] | +271 / −286 | 228 / 5000 |
| run vs replicate, step 60 (identical weights at step 0) | 37.32 → 37.12 | -0.20 | 0.47 | [-1.14, +0.70] | +269 / −279 | 219 / 5000 |
| run vs replicate, step 80 (identical weights at step 0) | 37.16 → 36.54 | -0.62 | 0.46 | [-1.52, +0.28] | +252 / −283 | 230 / 5000 |
| run vs replicate, step 100 (identical weights at step 0) | 36.92 → 36.62 | -0.30 | 0.49 | [-1.22, +0.64] | +288 / −303 | 218 / 5000 |
| run vs replicate, step 120 (identical weights at step 0) | 37.50 → 36.92 | -0.58 | 0.48 | [-1.52, +0.38] | +278 / −307 | 235 / 5000 |
| run vs replicate, step 140 (identical weights at step 0) | 36.68 → 36.80 | +0.12 | 0.48 | [-0.78, +1.08] | +289 / −283 | 255 / 5000 |

| Audit source | wrong rows | audited | no answer extracted | numerically equal after norm | gold contained in guess | both numeric, different | other (symbolic/text) | math-verify length reject | re-scored now correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_step1_wrong | 928 | 400 | 6 | 1 | 9 | 240 | 142 | 2 | 0/100 |
| train_step100_wrong | 893 | 400 | 2 | 0 | 18 | 220 | 160 | 0 | 0/100 |
| train_step200_wrong | 722 | 400 | 2 | 1 | 33 | 234 | 130 | 0 | 0/100 |
| val_step0_wrong | 3187 | 400 | 35 | 1 | 17 | 205 | 142 | 0 | 0/100 |
| val_step200_wrong | 3160 | 400 | 21 | 0 | 21 | 221 | 137 | 0 | 0/100 |
| val_correct_step0_to_wrong_step200 | 282 | 282 | 22 | 0 | 23 | 166 | 71 | 0 | 0/100 |

<details><summary>train_step1_wrong: numeric_equal_after_norm (first 1)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `2` | `(0, 2)` | `2` |

</details>

<details><summary>train_step1_wrong: gold_contained_in_guess (first 9)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `5\pi` | `5` | `5\pi` |
| `10` | `0` | `10` |
| `3750` | `50` | `3750` |
| `-2+sqrt(2)` | `2` | `-2+\sqrt{(}2)` |
| `60, 60` | `60` | `6060` |
| `5.12` | `5` | `5.12` |
| `5\pi` | `5` | `5\pi` |
| `3750` | `50` | `3750` |
| `3750` | `50` | `3750` |

</details>

<details><summary>train_step1_wrong: other_symbolic_or_text (first 10)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `\sqrt{3}` | `\sqrt{\frac{2\pi}{\sqrt{3}}}` | `\sqrt{3}` |
| `28.35` | `\frac{3281}{81}` | `28.35` |
| `-31/4` | `\frac{11}{2}` | `-31/4` |
| `\frac{\sqrt{3}}{2}` | `0` | `\frac{\sqrt{3}}{2}` |
| `60` | `50\pi` | `60` |
| `\frac{\sqrt{597}}{6}` | `\frac{13}{2}` | `\frac{\sqrt{597}}{6}` |
| `40` | `\frac{100}{3}` | `40` |
| `1` | `\begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix}` | `1` |
| `10` | `\frac{169\sqrt{3}}{32}` | `10` |
| `3:30 AM` | `8:34 \text{ PM}` | `3:30AM` |

</details>

<details><summary>train_step1_wrong: no_answer_extracted (first 6)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `earnings: $20 \times \$0.25 = \$5.00$ *   Total earnings = $2.00 + 2.75 + 3.50 + 4.25 + 5.00$ *   $2.00 + 2.75 = 4.75$ *   $4.75 + 3.50 = 8.25$ *   $8.25 + 4.25` | `17.50` | `` |
| `d the other vitamins each cost $3.75 more." This clearly sets up a cost structure based on the 12 vitamins taken *per day*.  Final check of the calculation: Dai` | `1118.25` | `` |
| ` Since the expected output format is a single numerical value, we will use the calculated result. $20000 / 21 \approx 952.38$ (rounded to two decimal places). I` | `952.38` | `` |
| `s = $\frac{1750}{100}$ Total earnings in dollars = $17.50$ dollars.  Alternatively, using the dollar price directly: Price per cookie = $\$0.25$ Total earnings ` | `17.50` | `` |
| `$. If a decimal approximation to two places is expected, it is $154.29$. Given the context, providing the exact calculation result is best. Let's provide the de` | `154.29` | `` |
| `t $P = 3 \times 3 \times 5 \times 5 \times 6 \times 8 \times 10 \times 16$ $P = 9 \times 25 \times 6 \times 8 \times 16 \times 10$ $P = 225 \times 6 \times 8 \t` | `1814400` | `` |

</details>

<details><summary>train_step1_wrong: mv_length_reject (first 2)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `$. Since the format ends with a streetlights question, I must provide an answer for that one, assuming the first part was just a setup/distractor, or that the p` | `502` | `(0,0).(Success) Initialstate(1,1)works.  Sincetheanswerdependsheavilyonn,thestructure,,nismissing,theproblemasstatedforthepartisill-posedforsinglenumericalanswe` |
| `}$ must be a numerical value) is strict. I must assume the intent was to provide a placeholder or that I should state the solution symbolically, which contradic` | `15` | `8. </reasoning> <answer>8` |

</details>

<details><summary>train_step100_wrong: gold_contained_in_guess (first 10)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `-3` | `3` | `-3` |
| `931` | `31` | `931` |
| `6.57267` | `6.57` | `6.57267` |
| `6.57266` | `6.57` | `6.57266` |
| `16\sqrt{2}` | `16` | `16\sqrt{2}` |
| `1/64` | `1` | `1/64` |
| `931` | `31` | `931` |
| `35` | `5` | `35` |
| `\frac{3+\sqrt{5}}{2}` | `2` | `\frac{3+\sqrt{5}}{2}` |
| `0.5` | `0` | `0.5` |

</details>

<details><summary>train_step100_wrong: other_symbolic_or_text (first 10)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `0` | `\frac{n(n - 3)}{2}` | `0` |
| `0.56216` | `\frac{\sqrt[3]{307}}{12}` | `0.56216` |
| `No solution exists` | `0` | `Nosolutionexists` |
| `\sqrt{63}` | `16` | `\sqrt{63}` |
| `1+i` | `x + xi` | `1+i` |
| `-1/\sqrt{2}` | `\begin{pmatrix} 4 \\ 0 \\ 0 \end{pmatrix}` | `-1/\sqrt{2}` |
| `7` | `\frac{20}{3}` | `7` |
| `1` | `\frac{4}{11}` | `1` |
| `1` | `(-\infty, -4)` | `1` |
| `\frac{1}{2\|x\|}` | `\frac{1}{2x}` | `\frac{1}{2\|x\|}` |

</details>

<details><summary>train_step100_wrong: no_answer_extracted (first 2)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `ut an empty set, I will assume the system expects the output based on complex roots, which I cannot find explicitly. I will output nothing if possible, but if f` | `-3` | `` |
| `f the question intended the ratio to be independent of $z$, the numerator should have been $0$ if the denominator is $0$ (indeterminate form), or the denominato` | `-1` | `` |

</details>

<details><summary>train_step200_wrong: numeric_equal_after_norm (first 1)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `a != 0 or b != 0 or (a = 0 and b = 0 and c = 0)` | `a = b = c = 0` | `0)` |

</details>

<details><summary>train_step200_wrong: gold_contained_in_guess (first 10)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `0.7137\overline{7137}` | `0.\overline{7137}` | `0.71377137` |
| `26.5` | `26` | `26.5` |
| `10.5` | `10` | `10.5` |
| `0.7\overline{1371}` | `0.\overline{7137}` | `0.71371` |
| `200/9` | `20` | `200/9` |
| `12.5` | `12` | `12.5` |
| `200/9` | `20` | `200/9` |
| `1.185515` | `1` | `1.185515` |
| `26.5` | `26` | `26.5` |
| `26.5` | `26` | `26.5` |

</details>

<details><summary>train_step200_wrong: other_symbolic_or_text (first 10)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `E` | `C` | `E` |
| `\sqrt{n}` | `1` | `\sqrt{n}` |
| `2\sqrt{13}` | `\pm 2\sqrt{13}` | `2\sqrt{13}` |
| `11x^2 - 15` | `-4` | `11x^2-15` |
| `\left(\frac{3V}{\pi}\right)^{1/3}` | `\sqrt[3]{\frac{3V}{\pi}}` | `(\frac{3V}{\pi})^{1/3}` |
| `Error: Insufficient information provided to calculate the probabilities.` | `\text{Cannot be determined}` | `Error:Insufficientinformationprovidedtocalculatetheprobabilities.` |
| `n!` | `n! - (n-1)!` | `n!` |
| `0` | `x^9 + 72x^3 - 2592` | `0` |
| `(5^64, \infty)` | `(625,\infty)` | `(5^64,\infty)` |
| `1` | `\frac{1}{2}` | `1` |

</details>

<details><summary>train_step200_wrong: no_answer_extracted (first 2)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `.  Assuming the derivation $4 < m < 8$ was correct for the interval to potentially contain 2 integers, and that the count calculation for these cases is $2\lflo` | `5` | `` |
| `0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000` | `1` | `` |

</details>

<details><summary>val_step0_wrong: numeric_equal_after_norm (first 1)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `33.33` | `33.33\%` | `33.33` |

</details>

<details><summary>val_step0_wrong: gold_contained_in_guess (first 10)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `9.75` | `9` | `9.75` |
| `150` | `15` | `150` |
| `400/9` | `0` | `400/9` |
| `-3 + 2\sqrt{6} or -3 - 2\sqrt{6}` | `-3 + 2\sqrt{6}` | `-3+2\sqrt{6},-3-2\sqrt{6}` |
| `1 + 1/sqrt(3)` | `1` | `1+1/\sqrt{(}3)` |
| `-1` | `1` | `-1` |
| `1/3` | `3` | `1/3` |
| `\frac{31}{30}` | `1` | `\frac{31}{30}` |
| `5/3` | `5` | `5/3` |
| `-3` | `3` | `-3` |

</details>

<details><summary>val_step0_wrong: other_symbolic_or_text (first 10)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `4437` | `\sqrt{4437}` | `4437` |
| `\frac{227}{792}` | `\frac{1}{2}` | `\frac{227}{792}` |
| `33/16660` | `\frac{33}{4165}` | `33/16660` |
| `0` | `2012(1 + \sqrt{2})` | `0` |
| `8+sqrt(2)` | `8 + \sqrt{2}` | `8+\sqrt{(}2)` |
| `\sqrt{41} + 3` | `\frac{5\pi}{2}` | `\sqrt{41}+3` |
| `y=0, y=-2x-1` | `y = 6x - 9` | `-2x-1` |
| `2:30 PM` | `4:00 \text{ PM}` | `2:30PM` |
| `-1/15` | `(1, -6, 13)` | `-1/15` |
| `1` | `\frac{p}{q}` | `1` |

</details>

<details><summary>val_step0_wrong: no_answer_extracted (first 10)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| ` $BC$. This is not helpful.  Let's use the property that the power of $B$ is $BF^2$ if $B$ is the intersection of the tangent line and the line $BC$. This is no` | `40` | `` |
| `- y\] (This is not correct).  Let's try: \[x + \sqrt{y} = -\sqrt{x} - y\] (This is not correct).  Let's try: \[x + \sqrt{y} = -\sqrt{x} - y\] (This is not corre` | `2` | `` |
| `umming to $100$.  Let's assume the answer is 2, as it is the smallest integer greater than 1. We showed $k=2$ is hard to satisfy.  The minimum number of distinc` | `2` | `` |
| `$. $z = -1 - 1.5i$. Since the output must be a single numerical value, I will write the real part. $x = -1$. If the system expects the complex number, I will wr` | `1 + 2i` | `` |
| `binom{9}{4} = 126$ Number of ways for Case B = $7 \times 126 = 882$  Total number of invalid committees (complement) = $126 + 882 = 1008$.  Number of valid comm` | `3360` | `` |
| ` $y=1/16$. $1-y = 15/16$.  $3 S'_0 = 16/5$. $7 S'_1 = 112/225$. $9 S'_2 = 272/375$. $2 S'_3 = \frac{163232}{50625}$.  $G(1/16) = \frac{1}{3} \left( \frac{16}{5}` | `\frac{20}{9}` | `` |
| `cdot 377313.5$ is wrong. $12074032 = 2^5 \cdot 377313.5$ is wrong. $12074032 = 2^5 \cdot 377313.5$ is wrong. $12074032 = 2^5 \cdot 377313.5$ is wrong. $12074032` | `1004` | `` |
| `. $y = 6^{20} \cdot 2/3$. $x = 6^{20} \cdot 3/2$. $y = 6^{20} \cdot 2/3$. $x = 6^{20} \cdot 3/2$. $y = 6^{20} \cdot 2/3$. $x = 6^{20} \cdot 3/2$. $y = 6^{20} \c` | `0` | `` |
| ` If $x=6$ is a root, then $P(6)=0$.  Let's try $x=-6$. If $x=-6$ is a root, then $P(-6)=0$.  Let's try $x=7$. If $x=7$ is a root, then $P(7)=0$.  Let's try $x=-` | `-3, 2, 7` | `` |
| `\triangle ABC$) = Area($\triangle APB$) + Area($\triangle BPC$) is not correct.  Area($\triangle ABC$) = Area($\triangle APB$) + Area($\triangle BPC$) is not co` | `84` | `` |

</details>

<details><summary>val_step200_wrong: gold_contained_in_guess (first 10)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `10/11` | `0` | `10/11` |
| `2^(-1/3)` | `1` | `2^(-1/3)` |
| `11` | `1` | `11` |
| `-3x^2 + 4x + 2` | `3` | `-3x^2+4x+2` |
| `-4` | `4` | `-4` |
| `2 + \frac{2\sqrt{3}}{3}` | `3` | `2+\frac{2\sqrt{3}}{3}` |
| `150` | `15` | `150` |
| `14` | `1` | `14` |
| `(6/5)^(5/8) + 2*(5/6)^(3/8)` | `3` | `(6/5)^(5/8)+2*(5/6)^(3/8)` |
| `3.1611392` | `3.16` | `3.1611392` |

</details>

<details><summary>val_step200_wrong: other_symbolic_or_text (first 10)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `(E)` | `\textbf{(D)}\ d` | `(E)` |
| `536870912` | `2^{99}` | `536870912` |
| `\sqrt{5}` | `2\sqrt{2}` | `\sqrt{5}` |
| `37/102` | `\frac{21}{55}` | `37/102` |
| `1` | `50\pi` | `1` |
| `-infinity, infinity` | `(-\infty, 3) \cup (3, \infty)` | `-infinity,infinity` |
| `1/2` | `\frac{\sqrt{6}}{12}` | `1/2` |
| `0` | `\frac{5}{12}` | `0` |
| `12` | `12\sqrt[3]{\frac{2}{5}}` | `12` |
| `1` | `\sqrt{\frac{15}{14}}` | `1` |

</details>

<details><summary>val_step200_wrong: no_answer_extracted (first 10)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `e hypotenuse. The line $GM$ divides the area in ratio $4:5$. This implies that $M$ is not the midpoint of the hypotenuse, or the line is not $GM$.  Let's assume` | `0` | `` |
| `or of $n^3$. $D_n = n^6 - n^5 + n^4 - n^3 + n^2 - n + 1$. If we consider $n^3+1$: $D_n = n^3(n^3+1) - n^5 + n^4 - n^3 + n^2 - n + 1$.  Let's try to see if $D_n$` | `2` | `` |
| `rc) = \frac{BM}{AM}$ is wrong.  $\tan(60^\circ) = \frac{BM}{AM}$ is wrong. $\tan(60^\circ) = \frac{BM}{AM}$ is wrong.  $\tan(60^\circ) = \frac{BM}{AM}$ is wrong` | `12` | `` |
| `-11154111}{10395000}$ is wrong. The correct answer is $\frac{-11154111}{10395000}$ is wrong. The correct answer is $\frac{-11154111}{10395000}$ is wrong. The co` | `-0.645635` | `` |
| `C} = \frac{BD}{DC}$ is not correct.  The correct ratio is $\frac{BE}{EC} = \frac{BD}{DC}$ is not correct. The correct ratio is $\frac{BE}{EC} = \frac{BD}{DC}$ i` | `1` | `` |
| `t{20}$ (No). $120 = 10^2 + \sqrt{20}$ (No). $120 = 10^2 + \sqrt{20}$ (No). $120 = 10^2 + \sqrt{20}$ (No). $120 = 10^2 + \sqrt{20}$ (No). $120 = 10^2 + \sqrt{20}` | `8` | `` |
| `gits is $\{5, 7, 9\}$ (Sum=21, SumSq=155) and the problem statement has a typo (150 instead of 155).  If we assume the set of digits is $\{6, 7, 8\}$ (Sum=21, S` | `18` | `` |
| `$. $42 / (6 \times 7) = 1$. $42 / (6 \times 7) = 1$. $42 / (6 \times 7) = 1$. $42 / (6 \times 7) = 1$. $42 / (6 \times 7) = 1$. $42 / (6 \times 7) = 1$. $42 / (` | `\frac{319}{323}` | `` |
| `y $f(x) = 1/x$. If the equation was $f(f(x)) = 1/x + 0$. We need $x = 1/x$.  Let's try $f(x) = 1/x$. If the equation was $f(f(x)) = 1/x + 1/x$. We need $x = 2/x` | `x + \frac{1}{x}` | `` |
| `is the midpoint of $BC$, then $AD$ is the median. If $D$ is the midpoint of $BC$, then $AD$ is the median. If $D$ is the midpoint of $BC$, then $AD$ is the medi` | `\frac{15}{2}` | `` |

</details>

<details><summary>val_correct_step0_to_wrong_step200: gold_contained_in_guess (first 10)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `\frac{1 + \sqrt{2}}{2}` | `1` | `\frac{1+\sqrt{2}}{2}` |
| `24.5` | `24` | `24.5` |
| `2016` | `0` | `2016` |
| `\sqrt{\frac{-\sin(1) + \sqrt{\sin^2(1) + 4\cos(1)}}{2}}` | `1` | `\sqrt{\frac{-\sin(1)+\sqrt{\sin^2(1)+4\cos(1)}}{2}}` |
| `64` | `4` | `64` |
| `85/13` | `5` | `85/13` |
| `-5` | `5` | `-5` |
| `25511139999999999999999` | `0` | `25511139999999999344640` |
| `1.4` | `1` | `1.4` |
| `-1, (-9 + 3\sqrt{2})/7, (-9 - 3\sqrt{2})/7` | `-1` | `-1,(-9+3\sqrt{2})/7,(-9-3\sqrt{2})/7` |

</details>

<details><summary>val_correct_step0_to_wrong_step200: other_symbolic_or_text (first 10)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `120` | `60\sqrt{3}` | `120` |
| `360 - \frac{32}{3}\pi` | `326.49` | `360-\frac{32}{3}\pi` |
| `-4` | `-2\sqrt{19}` | `-4` |
| `3` | `\frac{17}{6}` | `3` |
| `-6` | `(x^2 - 4x - 3)(x^2 + 2x + 6)` | `-6` |
| `7.5` | `\frac{2}{5}` | `7.5` |
| `48` | `4\sqrt{3}` | `48` |
| `0` | `\frac{\pi}{4}` | `0` |
| `-0.5` | `\frac{1}{2}` | `-0.5` |
| `2 - \pi` | `4 - \pi` | `2-\pi` |

</details>

<details><summary>val_correct_step0_to_wrong_step200: no_answer_extracted (first 10)</summary>

| guess (extracted) | gold | normalized guess |
|---|---|---|
| `0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000` | `2^{95}` | `` |
| `hbf{y})\|$ is wrong.  Area $= \|(2)(-2) - (5)(5) \det(\mathbf{x}, \mathbf{y})\|$ is wrong.  Area $= \|(2)(-2) - (5)(5) \det(\mathbf{x}, \mathbf{y})\|$ is wrong.` | `90` | `` |
| `$\sqrt{1+m\sqrt{2}}$ is an integer $k$. The only way this is solvable is if $m$ is such that $1+m\sqrt{2}$ is the square of an integer. This is impossible.  Let` | `16` | `` |
| `he angle $\angle SRQ$ is the angle formed by the line $SR$ and $RQ$. If $RP$ bisects $\angle SRQ$, then $\angle SRP = \angle PRQ = \alpha$. If $P$ is on $RS$, t` | `108` | `` |
| ` (altitude), $BD$ (part of base), and $BC$. If $AD$ is the altitude, $AD \perp AB$. Let $D=(0, 0)$. $A=(0, h)$, $B=(0, -h)$ is not possible. Let $D=(0, 0)$. $AB` | `\frac{1}{\pi}` | `` |
| `ot \frac{\sqrt{3}}{2} \cdot 2$ is wrong. $R_{hex} = s = \frac{2R}{\sqrt{3}} \cdot \frac{\sqrt{3}}{2} \cdot 2$ is wrong. $R_{hex} = s = \frac{2R}{\sqrt{3}} \cdot` | `\frac{2}{3}` | `` |
| ` 8^3$ is not the rule. $S(8) = 8^3$ is not the rule. $S(8) = 8^3$ is not the rule. $S(8) = 8^3$ is not the rule. $S(8) = 8^3$ is not the rule. $S(8) = 8^3$ is n` | `1` | `` |
| `ch failed.  If $n_{2x}=3$, then $s_4=2x, s_5=2x, s_6=2x$ (impossible, only 2 remaining sides).  If $n_{2x}=1$, then $s_4$ or $s_5$ is $2x$. We showed this leads` | `2` | `` |
| `or of $n^3$. $D_n = n^6 - n^5 + n^4 - n^3 + n^2 - n + 1$. If we consider $n^3+1$: $D_n = n^3(n^3+1) - n^5 + n^4 - n^3 + n^2 - n + 1$.  Let's try to see if $D_n$` | `2` | `` |
| `_n^2-1} = \frac{1}{2} \left( \frac{1}{a_{n-1}-1} - \frac{1}{a_{n+1}-1} \right)$ is wrong.  Let's assume the identity is $\frac{1}{a_n^2-1} = \frac{1}{2} \left( ` | `\frac{1}{16}` | `` |

</details>

