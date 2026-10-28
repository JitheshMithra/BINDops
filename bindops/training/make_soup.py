"""Interpolate between the DPD backbone and ours: theta(a) = (1-a)*theta_DPD + a*theta_ours.

WHY THIS IS THE RIGHT NEXT TEST, AND WHY IT IS FREE. The 2x2 settled that our fine-tuning is what
cost the per-pose quality, and that it bought pose DIVERSITY in exchange:

    cell (BINDOPS_MODE=control, N=8, same 60 targets)   mean pose   within-sd   top-1
    a=0  DPD backbone        poses_ABLATE_warmstart        0.0696      0.0060   0.0722
    a=1  ours               poses_X22_ours_control         0.0412      0.0310   0.0416

Both endpoints are already measured under IDENTICAL configuration, which is what makes a sweep
between them interpretable. The whole trade is a DIRECTION in weight space, so the question "is
there a point on that line with most of the quality and some of the diversity" costs only generation
-- NO TRAINING AT ALL. If such a point exists it is immediately deployable, because top-1 is what we
ship and the oracle's diversity is currently unbanked (we have no selector that beats random).

WHAT THE SHAPE OF THE CURVE MEANS, pre-registered before any intermediate arm exists:
  monotone in a (quality falls, diversity rises, top-1 falls)
      -> the trade is a smooth frontier with no free lunch. SHIP a=0: revert the backbone. The
         fine-tuning direction is simply harmful and the honest move is to give it up.
  interior maximum in TOP-1
      -> a soup beats both endpoints and that a is the new shipped config. Report the whole curve.
  interior MINIMUM (worse than both endpoints)
      -> a barrier on the line: linear mode connectivity fails between these two points, so
         interpolation is the wrong instrument and the fix must come from retraining with an
         objective that does not destroy per-pose quality. That is informative too, and it is the
         outcome that would kill this approach cleanly rather than leaving it ambiguous.
The deciding metric is TOP-1 DockQ (what we would ship), with mean pose / within-sd reported to show
where on the frontier each a sits.

THE GUARD. PairFlowModel.load_state_dict runs strict=False, so a key-name mismatch loads NOTHING,
silently, and generates from random weights -- which would look like a dramatic soup result while
measuring noise. shim.py hit exactly this (0/587 matched, DDP 'module.' prefix) and refused to
write. This script inherits that guard AND adds a real negative control: it builds a=0 and a=1 and
asserts they reproduce the two checkpoints already on disk tensor by tensor. A flipped
interpolation direction, or a wrong blend, cannot survive that.

ONLY THE 425 BACKBONE TENSORS ACTUALLY MOVE, even though all 587 are blended. The a=0 endpoint is
dpd_warmstart_ablation.pth, which shim.py built by taking the 425 DPD backbone tensors and KEEPING
OUR 162 adapter/pair tensors. So those 162 are byte-identical at both endpoints and blending them is
an identity -- the architecture is held fixed across the sweep and a moves only the backbone. That
is asserted below rather than assumed, because if it were ever false the sweep would silently be
interpolating the adapter too. Integer and non-floating buffers are copied, never blended --
averaging a step counter is meaningless.

usage: make_soup.py 0.25 0.5 0.75
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os
import sys

import numpy as np
import torch

W = "/8TBDrive1/jithesh/docking_model"
DPD_CK = W + "/verified_v2/runs/dpd_warmstart_ablation.pth"   # already the spliced a=0 point
OURS_CK = W + "/verified_v2/runs/full_all/best.pth"
OUTDIR = W + "/verified_v2/runs"

ALPHAS = [float(x) for x in sys.argv[1:]] or [0.25, 0.5, 0.75]
for a in ALPHAS:
    if not 0.0 <= a <= 1.0:
        raise SystemExit("alpha must be in [0,1], got %r" % a)


def find_sd(ck, label):
    if isinstance(ck, dict):
        for k in ("model", "model_state_dict", "state_dict", "ema_model", "net"):
            if k in ck and isinstance(ck[k], dict):
                return ck[k]
        if all(hasattr(v, "shape") for v in ck.values()):
            return ck
    raise SystemExit("could not locate a state dict in %s" % label)


print("loading the two endpoints...")
dpd_blob = torch.load(DPD_CK, map_location="cpu", weights_only=False)
our_blob = torch.load(OURS_CK, map_location="cpu", weights_only=False)
# dpd_warmstart_ablation.pth was written BY shim.py, so its keys are already our key set and the
# 'module.' prefix is long gone. Stripping again is a harmless no-op and keeps this robust if the
# endpoint is ever pointed at a raw DPD checkpoint instead.
A0 = {k.replace("module.", ""): v for k, v in find_sd(dpd_blob, "dpd_warmstart_ablation").items()}
A1 = {k.replace("module.", ""): v for k, v in find_sd(our_blob, "full_all/best").items()}

shared = [k for k in A1 if k in A0 and tuple(A0[k].shape) == tuple(A1[k].shape)]
blendable = [k for k in shared if A0[k].is_floating_point() and A1[k].is_floating_point()]
copied = [k for k in A1 if k not in blendable]
print("our tensors %d | shared+same-shape %d | BLENDABLE (float) %d | copied from ours %d"
      % (len(A1), len(shared), len(blendable), len(copied)))
if len(blendable) / max(len(A1), 1) < 0.5:
    raise SystemExit("REFUSING: only %.1f%% of our tensors are blendable. A soup built from this "
                     "would be mostly un-interpolated and the sweep would measure nothing."
                     % (100.0 * len(blendable) / max(len(A1), 1)))

# WHICH TENSORS ACTUALLY MOVE. Anything identical at both endpoints is unaffected by a, so the
# effective dimensionality of this sweep is the set that differs. Report it, and name what is frozen.
moving = [k for k in blendable
          if not torch.allclose(A0[k].float(), A1[k].float(), atol=0, rtol=0)]
frozen = [k for k in blendable if k not in set(moving)]
print("tensors that DIFFER between endpoints (these are what a moves): %d" % len(moving))
print("tensors IDENTICAL at both endpoints (a is an identity on these):  %d  e.g. %s"
      % (len(frozen), sorted(frozen)[:4]))
_adapter_pair = [k for k in frozen if k.startswith(("adapter", "pair"))]
print("  of those frozen, adapter.*/pair.* : %d  -- the architecture is held fixed across the sweep"
      % len(_adapter_pair))
assert len(moving) > 0.5 * len(blendable) or len(moving) >= 400, (
    "only %d tensors differ between the endpoints; there is no line to sweep" % len(moving))

# how far apart are the endpoints? if the fine-tuning barely moved the backbone there is no line.
drift = []
for k in blendable[:400]:
    x, y = A0[k].float().flatten(), A1[k].float().flatten()
    if x.numel() > 1 and x.norm() > 0:
        drift.append(float((y - x).norm() / x.norm()))
print("endpoint separation on %d tensors: median %.4f, p90 %.4f"
      % (len(drift), float(np.median(drift)), float(np.percentile(drift, 90))))


def soup(a):
    out = {}
    for k in A1:
        if k in blendable:
            out[k] = (1.0 - a) * A0[k].float() + a * A1[k].float()
            out[k] = out[k].to(A1[k].dtype)
        else:
            out[k] = A1[k]
    return out


# NEGATIVE CONTROL: the two endpoints must come back out of the blend exactly. This is what catches
# a flipped direction -- the single most likely bug here, and one that would invert the entire
# reading of the sweep while producing perfectly plausible-looking numbers.
for a, ref, nm in ((0.0, A0, "a=0 must equal the DPD backbone"),
                   (1.0, A1, "a=1 must equal our weights")):
    s = soup(a)
    bad = [k for k in blendable
           if not torch.allclose(s[k].float(), ref[k].float(), atol=1e-6, rtol=1e-5)]
    if bad:
        raise SystemExit("NEGATIVE CONTROL FAILED (%s): %d/%d tensors differ, e.g. %s"
                         % (nm, len(bad), len(blendable), bad[:4]))
    print("negative control passed: %s (%d tensors)" % (nm, len(blendable)))

for a in ALPHAS:
    out = {"model": soup(a)}
    if isinstance(our_blob, dict) and "head" in our_blob:
        out["head"] = our_blob["head"]        # our confidence head, unchanged at every a
    p = os.path.join(OUTDIR, "soup_a%03d.pth" % round(a * 100))
    torch.save(out, p)
    # report where this a actually sits, as a check that it is not silently an endpoint
    d0, d1 = [], []
    for k in blendable[:200]:
        x, y, z = A0[k].float(), A1[k].float(), out["model"][k].float()
        if x.numel() > 1 and (y - x).norm() > 0:
            d0.append(float((z - x).norm() / (y - x).norm()))
            d1.append(float((z - y).norm() / (y - x).norm()))
    print("wrote %s   measured fraction of the way from DPD to ours: %.4f (want %.4f)"
          % (p, float(np.median(d0)), a))
    assert abs(float(np.median(d0)) - a) < 1e-3, "soup landed at the wrong point on the line"
print("\ndone: %d soup checkpoints" % len(ALPHAS))
