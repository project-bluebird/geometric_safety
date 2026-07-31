#!/usr/bin/env bash
# Generation-2 campaign: every experiment behind the manuscript's numbers,
# regenerated in one session so the archive set is internally consistent.
set -euo pipefail

run_root="${1:-paper_eval_outputs/paper_g2_$(date +%Y%m%d_%H%M%S)}"
n_jobs="${N_JOBS:-128}"
repeat_seeds="${REPEAT_SEEDS:-5}"
uv_bin="${UV:-uv}"

mkdir -p "$run_root/logs" paper_eval_outputs
printf '%s\n' "$run_root" > paper_eval_outputs/latest_g2_run.txt

run_and_log() {
    local name="$1"
    shift
    local log_path="$run_root/logs/${name}.log"
    printf '[%s] starting %s\n' "$(date -Is)" "$name" | tee "$log_path"
    "$@" 2>&1 | tee -a "$log_path"
    printf '[%s] finished %s\n' "$(date -Is)" "$name" | tee -a "$log_path"
}

run_and_log main_evaluation \
    "$uv_bin" run python -m paper.run_evaluation \
    --n-jobs "$n_jobs" \
    --output-dir "$run_root/main_evaluation" \
    --row-output-mode none \
    --skip-plots \
    --profile large100k \
    --repeat-seeds "$repeat_seeds"

run_and_log projection_sweep \
    "$uv_bin" run python -m paper.run_projection_sweep \
    --n-jobs "$n_jobs" \
    --output-dir "$run_root/projection_full" \
    --combined-rows \
    --compress-rows

run_and_log robustness \
    "$uv_bin" run python -m paper.run_robustness \
    --n-jobs "$n_jobs" \
    --output-dir "$run_root/robustness" \
    --repeat-seeds "$repeat_seeds"

run_and_log sampling_sweep \
    "$uv_bin" run python -m paper.run_sampling_sweep \
    --n-jobs "$n_jobs" \
    --output-dir "$run_root/sampling_sweep" \
    --repeat-seeds "$repeat_seeds"

run_and_log schedule_enumeration \
    "$uv_bin" run python -m paper.run_schedule_enumeration \
    --output "$run_root/schedule_enumeration/schedule_enumeration.json"

run_and_log threshold_3nmi \
    "$uv_bin" run python -m paper.run_evaluation \
    --n-jobs "$n_jobs" \
    --output-dir "$run_root/threshold_3nmi" \
    --row-output-mode none \
    --skip-plots \
    --profile paper \
    --repeat-seeds 1 \
    --threshold-nmi 3

run_and_log slack_band_02 \
    "$uv_bin" run python -m paper.run_robustness \
    --n-jobs "$n_jobs" \
    --output-dir "$run_root/slack_band_02" \
    --repeat-seeds 1 \
    --campaign exceedance \
    --boundary-band-nmi 0.2

run_and_log slack_band_04 \
    "$uv_bin" run python -m paper.run_robustness \
    --n-jobs "$n_jobs" \
    --output-dir "$run_root/slack_band_04" \
    --repeat-seeds 1 \
    --campaign exceedance \
    --boundary-band-nmi 0.4

run_and_log envelope_budget \
    "$uv_bin" run python -m paper.render_envelope_budget_figure \
    --output "$run_root/envelope_budget/fig_envelope_budget.pdf"

printf 'Saved generation-2 run to %s\n' "$run_root"
