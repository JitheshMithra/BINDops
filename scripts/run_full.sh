#!/bin/bash
# The arm with the missing ingredients actually connected.
#
#   run_full.sh msa   -> v5 trainer (validation loop + best-checkpoint selection)
#                        + real UniRef90 MSA features (24-dim adapter input)
#                        + real site score (adapter UNFROZEN: BINDOPS_MODE=flexible)
#   run_full.sh all   -> all of the above + triangle pair stack + recycling
#
# "msa" exists so the bundle is not a single uninterpretable blob: with the pair-stack-only arm
# already measured, msa vs all vs pairstack separates the MSA/site contribution from the pair
# stack's. Recycling is still bundled inside "all" -- two cards and ~12 h per arm do not stretch
# to a 4-way factorial, and that ablation is named in the writeup rather than pretended away.
#
# Steps are 6000, not 10000: the 2026-10-13 budget test found step_18000 vs step_10000 n.s., and
# the first val curve was flat from step 4000 on, so the last 4000 steps bought nothing. With
# best.pth chosen on held-out loss there is no longer any reason to pay for them.
set -u
ARM=${1:?usage: run_full.sh msa|all}
W=/8TBDrive1/jithesh/docking_model          # DATA root (unchanged)
# CODE root: this repo. Override BINDOPS_HOME to run the code from elsewhere.
BINDOPS_HOME=${BINDOPS_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
V=$W/verified_v2
PYD=/8TBDrive2/DiffPepBuilder/env/bin/python
PYA=~/anaconda3/bin/python
STEPS=6000
VAL=$V/data/metadata_val_real_A.csv
LOG=$V/full_$ARM.log
CKPT=$V/runs/full_$ARM
mkdir -p "$V/stages" "$CKPT"

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export FLOW_TRAINER=$BINDOPS_HOME/bindops/training/train_flow_v5.py
export MSA_PROFILE_PKL=$V/msa/msa_profiles.pkl
export SITE_SCORE_PKL=$V/msa/site_scores.pkl
export BINDOPS_PROFILE=1
# flexible, NOT control: control freezes the adapter at its zero init, which is exactly how the
# "BINDops-conditioned" arms came to contain no BINDops. flexible trains it.
export BINDOPS_MODE=flexible
export BLIND_PRIOR=surface

if [ "$ARM" = "all" ]; then
  TRAINER=$BINDOPS_HOME/bindops/training/train_pair_flow.py
  GEN=$BINDOPS_HOME/bindops/inference/gen_poses_pair.py
  export PAIR_C=64 PAIR_C_HIDDEN=32 PAIR_LAYERS=4 PAIR_STACK=1
  RECYCLE=1
  VALFLAG="--pair-stack"
else
  TRAINER=$BINDOPS_HOME/bindops/training/train_bindops_flow.py
  GEN=$BINDOPS_HOME/bindops/inference/gen_poses_flow.py
  RECYCLE=0
  VALFLAG=""
fi

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
cd "$W" || exit 1
say "=== arm '$ARM': $STEPS steps, recycle=$RECYCLE, MSA+site on, val every 1000 ==="

# ---- preflight: never spend GPU time on features that do not cover the data -----------------
if ! ~/bin/capped_run.sh /8TBDrive2/SpeedDocker/env_symfix/bin/python3 -I \
      "$BINDOPS_HOME/bindops/features/preflight_features.py" --min-cov 0.97 --splits train,val_real_A,test >> "$LOG" 2>&1; then
  say "PREFLIGHT FAILED -- features do not cover the splits. Not training. See $LOG"
  exit 1
fi
say "preflight passed"

# ---- scheduling: give the 30-min val curves a card, without idling one for hours ------------
# Only the SECOND arm waits. anchorfix frees a card ~3.5 h before pairstack does (both measured
# at 750 steps/h on 2026-10-15), so making `msa` wait too would leave that card idle for 3.5 h.
# `all` still waits, which guarantees the val curves get a card as soon as pairstack ends rather
# than queueing behind a 12 h arm -- the reason the wait exists.
if [ "$ARM" = "all" ] && [ ! -f "$V/stages/valcurve_real.done" ]; then
  say "waiting for the queued val curves to finish before taking a card"
  until [ -f "$V/stages/valcurve_real.done" ]; do sleep 120; done
elif [ "$ARM" != "all" ]; then
  say "not waiting on the val curves: this arm takes the first card that frees (see patch note)"
fi

if [ ! -f "$V/stages/train_full_$ARM.done" ]; then
  G=$(~/bin/gpu_lease.sh full$ARM 13000 $$) || { say "no GPU lease"; exit 1; }
  say "train: GPU $G"
  CUDA_VISIBLE_DEVICES=$G \
    ~/bin/capped_run.sh $PYD "$TRAINER" "$CKPT" "$V/data" \
      --max-steps $STEPS --batch 64 --lr 1e-4 --max-sq-res 150000 --seed 0 \
      --val-csv "$VAL" --val-every 1000 --val-reps 4 --recycle $RECYCLE \
      >> "$V/train_full_$ARM.log" 2>&1
  rc=$?
  ~/bin/gpu_lease.sh --release "$G"
  # The start-pose leak is the DEFAULT path in the vendored speeddock/dataset.py; only
  # BLIND_PRIOR switches the runtime patch on, and a missing export would train against
  # native-derived starts whose only symptom is a suspiciously GOOD DockQ. Assert the
  # trainer actually printed that the patch went live -- the export alone proves nothing.
  if ! grep -q "native peptide pose never read" "$V/train_full_$ARM.log"; then
    say "BLIND PRIOR NOT ACTIVE -- start poses may be native-derived. Discarding."; exit 1
  fi
  if ! grep -q "DONE $STEPS steps" "$V/train_full_$ARM.log"; then
    say "TRAINING DID NOT REACH $STEPS STEPS (rc=$rc) -- stopping"; exit 1
  fi
  touch "$V/stages/train_full_$ARM.done"
  say "train done; selection: $(grep SELECTED "$V/train_full_$ARM.log" | tail -1)"
fi

# ---- generate from the VAL-SELECTED checkpoint, not from the last one -----------------------
CK=$CKPT/best.pth
[ -f "$CK" ] || CK=$CKPT/step_$STEPS.pth
say "generating from $(basename $CK)"

# ---- reload gate: the pair arm loads non-pair tensors with strict=False and discards the
# missing-key list, so a silent reload failure is possible and would only surface as bad DockQ
# three hours later. Check before spending the card. CPU, no lease needed.
if ! CUDA_VISIBLE_DEVICES= ~/bin/capped_run.sh $PYD "$BINDOPS_HOME/bindops/gates/gate_reload.py" "$CK" "$ARM" \
      > "$V/gate_reload_$ARM.log" 2>&1; then
  say "RELOAD GATE FAILED -- not generating. See $V/gate_reload_$ARM.log"
  tail -20 "$V/gate_reload_$ARM.log" | tee -a "$LOG"
  exit 1
fi
say "reload gate passed: $(grep -c '^PASS' "$V/gate_reload_$ARM.log") checks"
if [ ! -f "$V/stages/gen_full_$ARM.done" ]; then
  G=$(~/bin/gpu_lease.sh full$ARM 13000 $$) || { say "no lease for generate"; exit 1; }
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  CUDA_VISIBLE_DEVICES=$G BINDOPS_CENTER_NOISE=0 \
    ~/bin/capped_run.sh $PYD "$GEN" "$CK" "$V/proc_test/metadata_test.csv" \
      "$V/runs/poses_test_full_$ARM" 32 20 --gpu "$G" >> "$V/gen_full_$ARM.log" 2>&1
  ~/bin/gpu_lease.sh --release "$G"
  touch "$V/stages/gen_full_$ARM.done"; say "generate done"
fi

say "scoring DockQ"
~/bin/capped_run.sh $PYA "$BINDOPS_HOME/bindops/inference/rescore_dockq.py" "$V/runs/poses_test_full_$ARM" 6 \
  > "$V/runs/poses_test_full_$ARM.score" 2>&1
say "=== DONE arm '$ARM': $V/runs/poses_test_full_$ARM.score ==="
tail -8 "$V/runs/poses_test_full_$ARM.score" >> "$LOG"
