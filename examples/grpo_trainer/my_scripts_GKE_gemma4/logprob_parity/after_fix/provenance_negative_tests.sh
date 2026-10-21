#!/usr/bin/env bash
# Negative tests for gemma4_logprob_parity.py provenance checks (run on a worker pod; CPU only).
cd /tmp
H=/tmp/gemma4_logprob_parity.py
FIXA=/workspace/meta-RL/logprob_parity/fixture_fixed_1021_0813/fixture_step1.npz
FIXB=/workspace/meta-RL/logprob_parity/fixture_t1_notrunc_1021_0723/fixture_step1.npz
POS=/tmp/parity_v2_fixed   # positive run (select/hf/vllm on FIXA)
NEG=/tmp/parity_v2_neg; rm -rf $NEG; mkdir -p $NEG
python3 $H select --fixture $FIXA --out $NEG > /dev/null 2>&1
run() { echo "--- $1"; shift; "$@" > /tmp/neg_out.txt 2>&1; rc=$?; grep -E "^\[(report|select|hf|fixture)\]|refusing|^  " /tmp/neg_out.txt | grep -v "lengths=" | head -4; echo "exit=$rc"; }

cp /workspace/meta-RL/logprob_parity/analysis_fixed_1021_0813/hf__bf16_sdpa_unpacked.npz /workspace/meta-RL/logprob_parity/analysis_fixed_1021_0813/hf__bf16_sdpa_unpacked.json $NEG/
run "(a) legacy result without provenance (old tool output) -> report must refuse" python3 $H report --fixture $FIXA --out $NEG
rm -f $NEG/hf__*

cp $POS/hf__bf16_sdpa_unpacked.npz $NEG/ && python3 - "$POS" "$NEG" <<'EOF'
import json, sys
m = json.load(open(f"{sys.argv[1]}/hf__bf16_sdpa_unpacked.json")); m["fixture_sha256"] = "0" * 64
json.dump(m, open(f"{sys.argv[2]}/hf__bf16_sdpa_unpacked.json", "w"))
EOF
run "(b) result tagged with another fixture's sha256 -> report must refuse" python3 $H report --fixture $FIXA --out $NEG

python3 - "$POS" "$NEG" <<'EOF'
import json, sys
m = json.load(open(f"{sys.argv[1]}/hf__bf16_sdpa_unpacked.json")); m["selected"] = m["selected"][:-1] + [0]
json.dump(m, open(f"{sys.argv[2]}/hf__bf16_sdpa_unpacked.json", "w"))
EOF
run "(c) result computed for a different selection -> report must refuse" python3 $H report --fixture $FIXA --out $NEG

cp $POS/hf__bf16_sdpa_unpacked.json $NEG/ && python3 - "$POS" "$NEG" <<'EOF'
import numpy as np, sys
z = np.load(f"{sys.argv[1]}/hf__bf16_sdpa_unpacked.npz"); d = {k: z[k] for k in z.files}
k0 = sorted(d)[0]; d[k0] = d[k0][:-3]
np.savez(f"{sys.argv[2]}/hf__bf16_sdpa_unpacked.npz", **d)
EOF
run "(d) result with a wrong-length sequence -> report must refuse" python3 $H report --fixture $FIXA --out $NEG

run "(e) selection.json from fixture A used with fixture B -> report must refuse" python3 $H report --fixture $FIXB --out $NEG
run "(f) re-select with another seed while results exist -> select must refuse" python3 $H select --fixture $FIXA --out $NEG --seed 1

FAKE=/tmp/fakefix; rm -rf $FAKE /tmp/parity_v2_fake; mkdir -p $FAKE; ln -s $FIXA $FAKE/fixture_step1.npz
python3 - <<'EOF'
import json
m = json.load(open("/workspace/meta-RL/logprob_parity/fixture_fixed_1021_0813/fixture_step1.json"))
m["rollout"].update(temperature=0.8, top_k=50, top_p=0.95)
json.dump(m, open("/tmp/fakefix/fixture_step1.json", "w"))
EOF
run "(g) T=0.8 / top-k 50 / top-p 0.95 fixture -> select must refuse" python3 $H select --fixture $FAKE/fixture_step1.npz --out /tmp/parity_v2_fake
run "(h) same with --allow_nonparity_fixture -> warning, proceeds" python3 $H select --fixture $FAKE/fixture_step1.npz --out /tmp/parity_v2_fake --allow_nonparity_fixture
