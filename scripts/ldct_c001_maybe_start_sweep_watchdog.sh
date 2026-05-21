#!/usr/bin/env bash
# If the LDCT-C001 real ntrain sweep watchdog is not running, start it (for cron).
# Example crontab (every 10 minutes):
#   */10 * * * * bash /home/xielei/3dgs/r2_gaussian/scripts/ldct_c001_maybe_start_sweep_watchdog.sh

set -euo pipefail
R2_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$R2_ROOT/output/by_experiment/re_ldctc001_spiral_different_methods_ntrains"
PIDFILE="${SWEEP_PIDFILE:-$OUT/.sweep_watchdog.pid}"
PY_R2="${PY_R2:-/home/xielei/miniconda3/envs/r2_gaussian_n/bin/python}"
LOG_APPEND="${SWEEP_WATCHDOG_LOG:-$OUT/sweep_watchdog_supervised_$(date +%Y%m%d).log}"
mkdir -p "$OUT"

alive_watchdog() {
  if [[ ! -f "$PIDFILE" ]]; then
    return 1
  fi
  local pid
  pid="$(tr -d ' \n' <"$PIDFILE")"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  if [[ -r "/proc/$pid/cmdline" ]] && grep -q "ldct_c001_real_ntrain_sweep_watchdog" "/proc/$pid/cmdline" 2>/dev/null; then
    return 0
  fi
  return 1
}

if alive_watchdog; then
  echo "$(date -Is) [supervisor] watchdog already running (pid $(tr -d '\0' <"$PIDFILE"))" >>"$OUT/supervisor_spool.log"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Match your GPU if JIT fails; see scripts/run_ldct_c001_real_spiral_ntrain_methods_sweep.py (SAX env).
export SAX_TORCH_CUDA_ARCH_LIST="${SAX_TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;12.0}"

cd "$R2_ROOT"
echo "$(date -Is) [supervisor] starting watchdog -> $LOG_APPEND" >>"$OUT/supervisor_spool.log"
nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" SAX_TORCH_CUDA_ARCH_LIST="$SAX_TORCH_CUDA_ARCH_LIST" \
  "$PY_R2" scripts/ldct_c001_real_ntrain_sweep_watchdog.py \
  --interval_sec "${SWEEP_WATCHDOG_INTERVAL:-180}" \
  --max_attempts 0 \
  --pidfile "$PIDFILE" \
  -- \
  --ntrains 50 100 200 400 800 1000 \
  --only all \
  --sax_methods naf intratomo lineformer \
  >>"$LOG_APPEND" 2>&1 &
echo "$(date -Is) [supervisor] started pid=$!" >>"$OUT/supervisor_spool.log"
