#!/usr/bin/env bash
# One ESMFold2 design worker, one GPU.
# Does not clear freeze locks, does not kill other jobs, and does not
# launch the legacy generator. Clone this branch into a new directory.
# Leave /run/ti/src/germinal untouched.
set -euo pipefail

GPU="${1:?single gpu index}"
CONFIG="${2:?esmfold2 yaml config}"
GERM="${3:-/run/ti/src/germinal-esmfold2}"
ENVPY="${ESMFOLD2_PYTHON:-/run/ti/envs/esmfold2/bin/python}"
ROOT="${MMDESIGN_RUNTIME_ROOT:-/run/ti/mmdesign_like_vhh}"
PROJ="${MMDESIGN_PROJECT_ROOT:-/AIRvePFS/dair/sulixian-data/mmdesign_like_vhh}"

if [[ "$GPU" == *","* || "$GPU" == *" "* || ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "REFUSING gpu=$GPU. One worker per GPU."
  exit 2
fi

for lock in \
  "$PROJ/locks/SCIENCE_FIX_FREEZE" \
  "$ROOT/locks/SCIENCE_FIX_FREEZE" \
  "$PROJ/locks/PRODUCTION_DISABLED" \
  "$ROOT/locks/PRODUCTION_DISABLED"
do
  if [[ -f "$lock" && "${ESMFOLD2_CAMPAIGN_UNLOCK:-0}" != "1" ]]; then
    echo "$(date -Is) freeze lock present ($lock). ESMFold2 worker not started."
    echo "Set ESMFOLD2_CAMPAIGN_UNLOCK=1 to run this worker only. Do not delete the lock and do not start the legacy supervisor."
    exit 0
  fi
done

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" == *","* ]]; then
  echo "REFUSING CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES. One device only."
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$ROOT"/{locks,logs,runs}
LOCK="$ROOT/locks/esmfold2_${HOSTNAME}_gpu${GPU}.lock"
LOG="$ROOT/logs/esmfold2_${HOSTNAME}_gpu${GPU}.log"
exec 8>"$LOCK"
if ! flock -n 8; then
  echo "$(date -Is) esmfold2 lock held for gpu=$GPU; another worker owns this card"
  exit 0
fi
exec >>"$LOG" 2>&1

if command -v nvidia-smi >/dev/null 2>&1; then
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null | sed '/^$/d' || true)"
  if [[ -n "$busy" ]]; then
    echo "$(date -Is) REFUSING gpu=$GPU: compute process present ($busy). Will not preempt."
    exit 4
  fi
fi

cd "$GERM"
echo "$(date -Is) esmfold2 worker gpu=$GPU config=$CONFIG germ=$GERM"
set +e
"$ENVPY" -u run_germinal_esmfold2.py --config "$CONFIG" --gpu "$GPU"
rc=$?
set -e
echo "$(date -Is) esmfold2 worker done rc=$rc"
exit "$rc"
