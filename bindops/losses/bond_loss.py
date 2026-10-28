"""BOND_LOSS: give the flow a reason not to contract the peptide chain.

THE DEFECT THIS EXISTS FOR (analysis_ablation/grow.py, results/28_grow.txt). Our fine-tuning
COMPRESSES the peptide. Median consecutive CA-CA distance against the 3.81 A peptide bond:

    DiffPepDock warm start (alpha 0.00)   3.99 A  = 1.05x   -- arrives CORRECT
    our fine-tuned weights (alpha 1.00)   2.55 A  = 0.67x   -- fine-tuning broke it
    weight soup            (alpha 0.20)   3.01 A  = 0.79x   -- our best arm, partway back
    RAPiDock                              3.82 A  = 1.00x
    native                                3.81 A

A chain at 0.67x scale cannot be within 2 A of a native at 1.00x wherever it is placed, which is a
candidate account of 0 of 2432 poses under 2 A.

THE CAUSE, MEASURED from runs/full_all/val_curve.jsonl, which logs both terms:
    step 2000 (SHIPPED, best.pth):  val_fape 0.4658  val_bb 723.5  val_loss 362.196
    step 4000:                      val_fape 0.4174  val_bb 834.7  val_loss 417.765
    step 6000:                      val_fape 0.3967  val_bb 772.3  val_loss 386.568
    recomputed 1.0*fape + 0.5*bb matches the logged val_loss to 1.14e-13.
    => At the SHIPPED checkpoint FAPE is 0.129% of the loss and the backbone L2 is 99.87%.
    (0.100-0.129% across the three steps. The 0.397/772.3/0.103% triple quoted elsewhere in
    this project is step_6000, NOT the checkpoint we ship.)
FAPE is the only shipped term that constrains internal geometry and it carries one part in a
thousand of the gradient. The backbone term is an UNNORMALISED, UNCLAMPED mean of squared
displacements, and under directional uncertainty a squared point loss is minimised by contracting
the chain toward its centroid.

TWO CHANGES, each justified by its own gate in analysis_ablation/lossfix.py (5/5 passed,
results/31_lossfix.txt):

  1. RESHAPE the backbone term: Huber on displacement / 10 A instead of the raw mean square.
     GATE 1: geometry's share of the gradient goes ~0.1% -> 25.8%, because the squared term
     stops swamping it. GATE 2: the gradient is bounded but NEVER zero (0.1 at 20, 40 and 80 A),
     unlike FAPE which is clamped at 10 A and has exactly zero gradient past it -- the failure mode
     place_loss.py already recorded for coordinate regressions.
     THIS DOES NOT STOP THE CONTRACTION ON ITS OWN. GATE 3 showed the old AND the reshaped point
     loss are both minimised by a fully COLLAPSED chain (s=0.00) under rotational ambiguity. That is
     a general property of per-residue point losses, not a quirk of the square, so the reshape is
     justified by gates 1-2 only and must not be sold as the contraction fix.

  2. ADD a rotation- and translation-INVARIANT bond term. GATE 5: it is the only term tested that is
     minimised at the correct chain scale (exactly s=1.00), because a quantity built from
     |CA_{i+1} - CA_i| cannot be touched by the directional ambiguity that defeats point losses. It
     scores 0.109 at the observed 0.67x contraction with a nonzero gradient (0.0223) there.

HOW IT COMPOSES. Same convention as contact_loss.py and place_loss.py: `wrap_flow_loss(base)` wraps
whichever flow_loss was selected, so it stacks with IFACE_FAPE / CONTACT_LOSS / PLACE_LOSS. The
reshape is done by RECOMPUTING the original backbone term differentiably and subtracting that TENSOR
-- not the float in `parts` -- so the old term's gradient is cancelled exactly rather than left in
place. SELF_TEST below asserts that cancellation numerically; a drifted formula would silently leave
a residue, which is precisely the kind of silent no-op this project has been bitten by before.

OFF BY DEFAULT. Every earlier arm reproduces exactly unless BOND_LOSS=1.

usage: BOND_LOSS=1 [W_BOND=1.0] via train_flow_v5.py ; `python bond_loss.py` runs the self-test.
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os
import sys

import torch

BOND = float(os.environ.get("BOND_LEN", "3.81"))
W_BOND = float(os.environ.get("W_BOND", "1.0"))
W_BB = float(os.environ.get("W_BB", "0.5"))          # must match what the trainer passes
L_SCALE = float(os.environ.get("BB_LENGTH_SCALE", "10.0"))
RESHAPE_BB = os.environ.get("BOND_RESHAPE_BB", "1") == "1"
EPS = 1e-8
CA_IDX = 1


def _huber(x, delta=1.0):
    """Quadratic for |x| < delta, linear beyond; continuous in value and first derivative."""
    a = x.abs()
    return torch.where(a < delta, 0.5 * x * x, delta * (a - 0.5 * delta))


def _masks(batch):
    loss_mask = (batch['ligand_mask'] * batch['res_mask']).float()
    am = batch['atom37_mask'][:, :, :5] * loss_mask[..., None]
    return loss_mask, am


def bb_square(out, batch):
    """The SHIPPED backbone term, recomputed differentiably so it can be cancelled exactly."""
    _, am = _masks(batch)
    pa, ga = out['atom37'][:, :, :5], batch['atom37_pos'][:, :, :5]
    return (((pa - ga) ** 2).sum(-1) * am).sum() / am.sum().clamp_min(1.0)


def bb_huber(out, batch):
    """The reshaped backbone term: Huber on displacement / L_SCALE. O(1) instead of O(hundreds)."""
    _, am = _masks(batch)
    pa, ga = out['atom37'][:, :, :5], batch['atom37_pos'][:, :, :5]
    d = torch.sqrt(((pa - ga) ** 2).sum(-1) + EPS)
    return (_huber(d / L_SCALE) * am).sum() / am.sum().clamp_min(1.0)


def bond_term(out, batch):
    """Mean squared RELATIVE deviation of consecutive peptide CA-CA from the peptide bond.

    Rotation- and translation-invariant by construction. Normalised by BOND so the term is
    dimensionless and O(1): a chain contracted to 0.67x scores (0.67-1)^2 = 0.109. Only pairs where
    BOTH residues are peptide are counted, and where seq_idx is available only pairs that are truly
    consecutive, so a chain break is never charged as a stretched bond.
    """
    loss_mask, _ = _masks(batch)
    ca = out['atom37'][:, :, CA_IDX, :]
    cm = loss_mask * batch['atom37_mask'][:, :, CA_IDX]
    pair = cm[:, 1:] * cm[:, :-1]
    if 'seq_idx' in batch:
        si = batch['seq_idx']
        pair = pair * ((si[:, 1:] - si[:, :-1]) == 1).float()
    d = torch.sqrt(((ca[:, 1:, :] - ca[:, :-1, :]) ** 2).sum(-1) + EPS)
    r = (d - BOND) / BOND
    return ((r * r) * pair).sum() / pair.sum().clamp_min(1.0), d, pair


def wrap_flow_loss(base):
    """Wrap whichever flow_loss is active, same convention as contact_loss / place_loss."""

    def wrapped(out, batch, *a, **kw):
        total, parts = base(out, batch, *a, **kw)
        l_bond, d, pair = bond_term(out, batch)
        total = total + W_BOND * l_bond
        if RESHAPE_BB:
            old = bb_square(out, batch)
            new = bb_huber(out, batch)
            # cancel the shipped term WITH its gradient, then add the reshaped one
            total = total - W_BB * old + W_BB * new
            parts = dict(parts)
            parts['bb_sq'] = old.item()
            parts['bb'] = new.item()
        else:
            parts = dict(parts)
        parts['bond'] = l_bond.item()
        if pair.sum() > 0:
            parts['ca_ca'] = float((d * pair).sum() / pair.sum())
        return total, parts

    return wrapped


# --------------------------------------------------------------------------- self-test
def _fake(n_res=16, n_pep=12, scale=1.0, err=20.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, (n_pep - 1) * BOND, n_pep)
    chain = torch.stack([t, torch.zeros(n_pep), torch.zeros(n_pep)], 1)
    gt = torch.zeros(1, n_res, 37, 3)
    gt[0, :n_pep, CA_IDX, :] = chain
    for a in range(37):
        gt[0, :n_pep, a, :] = chain + 0.3 * torch.randn(n_pep, 3, generator=g)
    cen = chain.mean(0)
    pred_chain = cen + (chain - cen) * scale + torch.tensor([err, 0.0, 0.0])
    pr = torch.zeros(1, n_res, 37, 3)
    pr[0, :n_pep, CA_IDX, :] = pred_chain
    for a in range(37):
        pr[0, :n_pep, a, :] = pred_chain + 0.3 * torch.randn(n_pep, 3, generator=g)
    pr.requires_grad_(True)
    lig = torch.zeros(1, n_res)
    lig[0, :n_pep] = 1.0
    batch = dict(ligand_mask=lig, res_mask=torch.ones(1, n_res),
                 atom37_pos=gt, atom37_mask=torch.ones(1, n_res, 37),
                 seq_idx=torch.arange(n_res).unsqueeze(0).float())
    return dict(atom37=pr), batch, pr


if __name__ == "__main__":
    print("bond_loss self-test  (BOND=%.2f  W_BOND=%.2f  W_BB=%.2f  reshape=%s)"
          % (BOND, W_BOND, W_BB, RESHAPE_BB))
    ok = []

    # 1. exact cancellation: the recomputed shipped term must equal what a stub base reports
    out, batch, pr = _fake()
    def stub(o, b, w_fape=1.0, w_bb=W_BB):
        return w_bb * bb_square(o, b), {'fape': 0.0, 'bb': bb_square(o, b).item()}
    tot_b, parts_b = stub(out, batch)
    w = wrap_flow_loss(stub)
    tot_w, parts_w = w(out, batch)
    recomputed = parts_w['bb_sq']
    d1 = abs(recomputed - parts_b['bb'])
    print("  1 cancellation: base bb %.6f vs recomputed %.6f  |d| %.2e -> %s"
          % (parts_b['bb'], recomputed, d1, "PASS" if d1 < 1e-5 else "FAIL"))
    ok.append(d1 < 1e-5)
    expect = W_BB * parts_w['bb'] + W_BOND * parts_w['bond']
    d2 = abs(float(tot_w) - expect)
    print("  2 total reconstructs as W_BB*bb_huber + W_BOND*bond: %.6f vs %.6f  |d| %.2e -> %s"
          % (float(tot_w), expect, d2, "PASS" if d2 < 1e-4 else "FAIL"))
    ok.append(d2 < 1e-4)

    # 3. the bond term must be minimised at the correct scale and detect the observed contraction
    print("  3 bond term vs chain scale (and the CA-CA it implies):")
    best = (9e18, None)
    for sc in (0.4, 0.67, 0.8, 1.0, 1.2):
        o, b, _ = _fake(scale=sc)
        v, d, p = bond_term(o, b)
        print("      s=%.2f  bond %.5f   mean CA-CA %.2f A" % (sc, float(v),
                                                               float((d * p).sum() / p.sum())))
        if float(v) < best[0]:
            best = (float(v), sc)
    print("      minimised at s=%.2f -> %s" % (best[1], "PASS" if abs(best[1] - 1.0) < 1e-9
                                               else "FAIL"))
    ok.append(abs(best[1] - 1.0) < 1e-9)

    # 4. gradient must flow to the prediction at the observed contraction
    out, batch, pr = _fake(scale=0.67)
    tot, parts = wrap_flow_loss(stub)(out, batch)
    tot.backward()
    gn = float(pr.grad.norm())
    print("  4 gradient reaches atom37 at s=0.67: norm %.5f -> %s"
          % (gn, "PASS" if gn > 1e-6 else "FAIL"))
    ok.append(gn > 1e-6)

    # 5. scale sanity: reshaped bb must be O(1) where the square is O(hundreds)
    out, batch, _ = _fake(err=27.8)
    sq, hu = float(bb_square(out, batch)), float(bb_huber(out, batch))
    print("  5 at 27.8 A error: bb_square %.1f  bb_huber %.3f  ratio %.0fx -> %s"
          % (sq, hu, sq / max(hu, 1e-9), "PASS" if hu < 10.0 < sq else "FAIL"))
    ok.append(hu < 10.0 < sq)

    # 6. a chain break must not be charged as a stretched bond
    o, b, _ = _fake()
    b['seq_idx'] = b['seq_idx'].clone()
    b['seq_idx'][0, 6:] += 50          # inject a break after residue 5
    v_break, _, p_break = bond_term(o, b)
    o2, b2, _ = _fake()
    v_ok, _, p_ok = bond_term(o2, b2)
    print("  6 chain break excluded: pairs %d -> %d, bond %.5f -> %.5f -> %s"
          % (int(p_ok.sum()), int(p_break.sum()), float(v_ok), float(v_break),
             "PASS" if int(p_break.sum()) < int(p_ok.sum()) else "FAIL"))
    ok.append(int(p_break.sum()) < int(p_ok.sum()))

    print("\n  SELF-TEST %d/%d passed" % (sum(ok), len(ok)))
    sys.exit(0 if all(ok) else 1)
