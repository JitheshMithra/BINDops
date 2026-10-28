"""Localisation objective for the global placement head. Drop-in wrapper around flow_loss.

TWO TERMS, and the division of labour between them is the whole point.

 1. POINTER CROSS-ENTROPY -- teaches WHICH PATCH. The head's softmax over receptor residues is
    scored against a soft target centred on the true peptide centroid,
    q_i  proportional to  exp(-||CA_i - gt_peptide_centroid||^2 / 2 sigma^2),  sigma = 6 A.
    This is a classification over the receptor's own residues, which is well conditioned at any
    error magnitude: a pointer aiming 40 A off gets exactly as informative a gradient as one
    aiming 8 A off. Every previous placement fix was a coordinate regression and inherited that
    metric's pathology -- a quadratic explodes at 40 A, and torch.clamp has ZERO gradient past
    its cutoff, so a clamped term gives a 40 A miss no gradient at all (the measured reason the
    iface-FAPE dose-response ran backwards).

 2. CENTROID SMOOTH-L1 -- teaches HOW FAR TO COMMIT, i.e. it is the gate's teacher. Huber on the
    centroid distance with beta = 10 A: quadratic where the pose is nearly placed, LINEAR beyond
    10 A. Linear is the property that matters -- bounded gradient magnitude at a 40 A miss, but
    still nonzero, which is exactly what a clamp destroys and a bare quadratic makes explosive.

THE TWO TERMS DISAGREE ABOUT THE DESTINATION, DELIBERATELY, AND THE WEIGHTS ENCODE WHICH WINS.
This is the one thing to understand before changing W_POINTER. The pointer CE pulls w toward q,
and a pointer with w = q lands a median 8.35 A from the true centroid, because q is a Gaussian
over receptor CAs and their weighted mean is pulled inward (reach_place.py, n=86). The centroid
term pulls toward the REACHABLE optimum, median 0.13 A. So term 1's fixed point is NOT term 2's,
and term 1 must not be allowed to win at convergence.

The budget, measured rather than guessed: CE starts at ~5.9 nats and bottoms at ~3.57 for a
perfect q-pointer, so it can buy 2.3 nats; the centroid term starts at ~20 (Huber of a 25.7 A
error at beta=10) and bottoms near 0.2. At W_POINTER=10 the CE could buy 23 loss units against
the centroid term's 20 -- it would dominate, and the arm would train the peptide to a destination
we have already measured as 8 A too deep. At W_POINTER=2 it buys 4.6 against 20, which is what we
want: the CE is a WELL-CONDITIONED EARLY GUIDE into the right patch (a classification gradient is
just as informative at a 40 A error as at an 8 A one, where every coordinate regression this
project has tried either exploded or clamped to zero gradient), and the centroid term owns the
final placement. Treat W_POINTER as an auxiliary-loss weight, not an objective weight.

WHY THE GATE NEEDS ITS OWN TERM AT ALL. The stock loss does already supervise placement, through
an unclamped backbone L2 that is 95-99.9% of the total. That term would eventually move the gate.
But it is a quadratic on 5 atoms in global coordinates with no scale control, and the arm's entire
risk is that the gate sits at its zero init and the head stays a no-op -- the exact way the pair
stack came out null. A small, well-scaled, placement-shaped term is cheap insurance against
repeating that, and unlike the bb L2 its value is READABLE in the step log, so "the gate never
moved" and "the gate moved and did not help" are distinguishable outcomes rather than one null.

LABEL PROVENANCE. The target centroid comes from batch['rigids_0'], ground truth, which is never
an input: the peptide reaches the network noised, through rigids_t and sc_ca_t. The RECEPTOR
coordinates used for the pointer's support are the ones the model already sees -- the receptor is
fixed, not noised -- which is correct, since the receptor's geometry is given at inference too.

WHAT IS REPORTED, and why these and not the losses. `ptr_d` is the distance from the pointed-at
target to the true peptide centroid, in angstroms, and `ptr_hit` the fraction of the batch under
10 A. Those are the localisation metric itself, on the same 10 A convention as the frozen-feature
gate run (48.1% hit<10 A for the standalone head, 30.6% for the full docking model). `gate` is the
mean |tanh| output: if it stays at 0 the head is inert and the arm is null for a MECHANICAL
reason, not a scientific one, and that must be visible at step 25 rather than at step 6000.

knobs: W_POINTER=2.0  W_CENTROID=1.0  POINTER_SIGMA=6.0  CENTROID_BETA=10.0
"""
import os

import torch
import torch.nn.functional as F
from openfold.utils.rigid_utils import Rigid

W_POINTER = float(os.environ.get("W_POINTER", "2.0"))
W_CENTROID = float(os.environ.get("W_CENTROID", "1.0"))
SIGMA = float(os.environ.get("POINTER_SIGMA", "6.0"))
BETA = float(os.environ.get("CENTROID_BETA", "10.0"))


def _pool(x, m):
    return (x * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1.0)


def place_term(out, batch):
    """(loss, parts). Returns (None, {}) when the model emitted no placement stash."""
    st = out.get("place", None)
    if st is None:
        return None, {}
    res = batch["res_mask"].float()
    lig = batch["ligand_mask"].float()
    pep_mask, rec_mask = lig * res, (1.0 - lig) * res
    gt_trans = Rigid.from_tensor_7(batch["rigids_0"], normalize_quats=True).get_trans().float()
    gt_cent = _pool(gt_trans, pep_mask)                               # [B,3] true centroid

    # --- 1. pointer cross-entropy ------------------------------------------------------------
    d2 = ((gt_trans - gt_cent[:, None, :]) ** 2).sum(-1)              # [B,N]
    ql = (-d2 / (2.0 * SIGMA * SIGMA)).masked_fill(rec_mask < 0.5, float("-inf"))
    q = torch.softmax(ql, dim=-1) * rec_mask                          # [B,N], 0 off-receptor
    logits = st["ptr_logits"].float()
    logp = torch.log_softmax(logits, dim=-1)
    # q is exactly zero off the receptor and logp is -inf there, so the product is 0 * -inf = NaN.
    # Zeroing logp on those positions is numerically identical to the intended sum and finite.
    logp = logp.masked_fill(rec_mask < 0.5, 0.0)
    ce = -(q * logp).sum(-1)                                          # [B]
    ent = -(q * torch.where(q > 0, q.clamp_min(1e-12).log(),
                            torch.zeros_like(q))).sum(-1)             # [B] H(q)
    l_ptr = ce.mean()

    # --- 2. centroid smooth-L1 (the gate's teacher) ------------------------------------------
    pred_trans = Rigid.from_tensor_7(out["rigids"], normalize_quats=True).get_trans().float()
    pred_cent = _pool(pred_trans, pep_mask)
    dc = (pred_cent - gt_cent).norm(dim=-1)                           # [B] angstroms
    l_cent = F.smooth_l1_loss(dc, torch.zeros_like(dc), beta=BETA)

    loss = W_POINTER * l_ptr + W_CENTROID * l_cent
    with torch.no_grad():
        # the localisation metric itself: how far the POINTER aims from the true site. This is
        # the number the arm lives or dies by, on the same 10 A convention as the gate run.
        pd = (st["target"].detach().float() - gt_cent).norm(dim=-1)
        parts = {"ptr_ce": float(l_ptr.detach()), "ptr_kl": float((ce - ent).mean().detach()),
                 "ptr_d": float(pd.mean()), "ptr_hit": float((pd < 10.0).float().mean()),
                 "cent_d": float(dc.mean().detach()),
                 "gate": float(st["gate"].detach().abs().mean()),
                 # signed, in angstroms: positive = the head is pushing the peptide OUTWARD of
                 # the pointed patch, which is the direction that corrects the pointer's known
                 # inward bias. A radial that stays at 0 means that correction never engaged.
                 "radial": float(st["radial"].detach().mean())}
    return loss, parts


def wrap_flow_loss(base):
    """base(out, batch, ...) -> (total, parts). Same signature out, placement terms added."""

    def flow_loss(out, batch, *a, **kw):
        total, parts = base(out, batch, *a, **kw)
        l_p, p_p = place_term(out, batch)
        if l_p is not None:
            total = total + l_p
            parts = dict(parts, **p_p)
        return total, parts

    flow_loss.base = base
    return flow_loss
