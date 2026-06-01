#!/bin/bash
# 5 seeds in 5 SEPARATE jobs (NO job array) — one job per seed, each running all
# 5 envs sequentially. Seeds 42..46 -> run_0..run_4. Chains a final cross-seed
# aggregate after all 5 finish.  bash submit_5seeds_5jobs.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p slurm_logs results logs

SEED_BASE="${SEED_BASE:-42}"
N_SEEDS="${N_SEEDS:-5}"
dep=""
for i in $(seq 0 $(( N_SEEDS - 1 ))); do
    SEED=$(( SEED_BASE + i ))
    JID=$(sbatch --parsable --job-name="prr_seed${SEED}" \
                 --export=ALL,SEED=${SEED},RUN_ID=${i} submit_one_seed.sh)
    echo "submitted seed ${SEED} -> job ${JID} (run_${i}, 5 envs sequential)"
    dep="${dep}:${JID}"
done
AID=$(sbatch --parsable --dependency="afterany${dep}" submit_aggregate.sh)
echo "chained cross-seed aggregate ${AID} (after${dep})"
