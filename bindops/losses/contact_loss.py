"""Inter-chain contact loss on the pair stack. Drop-in wrapper around flow_loss.

WHY THIS EXISTS. The triangle-multiplicative pair stack (pair_stack.py) was added on
2026-10-15 and gated 8/8, and the arm came back null. The reason is not the mechanism, it is
that NOTHING READ IT. The stack's only consumer is `out_proj`, which is zero-initialised so the
warm start stays exact -- meaning at step 0 the gradient arriving at every triangle layer is
identically zero. And the loss it would eventually have to earn its way into is the 24-line
peptide-masked flow loss, which contains no term whose value depends on which receptor residue
pairs with which peptide residue. A pair representation with no pair-shaped target is dead
weight; triangle multiplication computes pairwise consistency and we never asked it for any.

WHAT THIS ADDS. A supervised target of exactly the pair shape: for every (peptide i, receptor j)
pair, does i contact j in the native complex. BCE on the pair stack's own contact head, summed
over the inter-chain block only.

CONVENTION, chosen to match cmap_cache.py exactly so this is the SAME task the localisation
head was measured on (0.697 AUROC) rather than a near-miss of it:
  pseudo-CB = CB where present, CA for glycine;  contact = pseudo-CB distance < 8.0 A.

WHAT IT IS NOT. It is not a geometry term. It never touches the predicted coordinates, so it
cannot move a pose by itself; it shapes the edge representation that the IPA trunk then reads.
That is deliberate -- the three previous attempts to fix placement by adding a coordinate term
(iface FAPE, dose-response x3) are on record as null, and the architecture audit says the gap is
representational.

LABEL PROVENANCE. Targets come from batch['atom37_pos'], the ground-truth coordinates, which are
never an input to the network (the peptide arrives noised through rigids_t / sc_ca_t). So this
is supervision, not leakage. The receptor half of atom37_pos IS the receptor the model sees --
it is fixed, not noised -- which is correct: the receptor's own geometry is given at inference.

knobs: W_CONTACT=1.0  CONTACT_CUTOFF=8.0  CONTACT_POS_W=0 (0 = auto from the batch, clamped)
"""
import os

import torch
import torch.nn.functional as F

W_CONTACT = float(os.environ.get("W_CONTACT", "1.0"))
CUTOFF = float(os.environ.get("CONTACT_CUTOFF", "8.0"))
POS_W = float(os.environ.get("CONTACT_POS_W", "0"))      # 0 = per-batch auto
CB, CA = 3, 1                                            # atom37 order, as in cmap_cache.py


def pseudo_cb(atom37_pos, atom37_mask):
    """CB where present, CA for glycine. Returns coords [B,N,3] and a validity mask [B,N]."""
    xyz = atom37_pos[..., CB, :].clone()
    miss = atom37_mask[..., CB] < 0.5
    xyz = torch.where(miss[..., None], atom37_pos[..., CA, :], xyz)
    ok = (atom37_mask[..., CB] > 0.5) | (atom37_mask[..., CA] > 0.5)
    return xyz, ok.to(atom37_pos.dtype)


def contact_target(batch):
    """Native inter-chain contact map and the mask of pairs the loss is allowed to see.

    The mask is the peptide x receptor block AND its transpose (the head is symmetrised, so
    both triangles carry the same prediction and both are supervised), intersected with
    res_mask on both axes -- so padded residues contribute nothing and a padded batch scores
    identically to an unpadded one.
    """
    xyz, ok = pseudo_cb(batch["atom37_pos"], batch["atom37_mask"])
    D = torch.cdist(xyz.float(), xyz.float())
    y = (D < CUTOFF).to(xyz.dtype)
    res = batch["res_mask"].to(xyz.dtype)
    lig = (batch["ligand_mask"].to(xyz.dtype) * res)
    rec = ((1.0 - batch["ligand_mask"].to(xyz.dtype)) * res)
    inter = lig[..., None] * rec[..., None, :]
    inter = inter + inter.transpose(-1, -2)              # both triangles, never double-counted
    valid = ok * res
    return y, inter * valid[..., None] * valid[..., None, :]


def contact_term(out, batch):
    """(loss, parts). Returns (None, {}) when the model emitted no contact logits."""
    lg = out.get("contact_logits", None)
    if lg is None:
        return None, {}
    y, m = contact_target(batch)
    lg = lg.to(y.dtype)
    npos = (y * m).sum()
    ntot = m.sum()
    if POS_W > 0:
        pw = torch.as_tensor(POS_W, dtype=y.dtype, device=y.device)
    else:
        # inter-chain contacts are ~1-3% of the block; without reweighting the all-negative
        # solution is already at 98% and the head never leaves it.
        pw = ((ntot - npos) / npos.clamp_min(1.0)).clamp(1.0, 200.0).detach()
    bce = F.binary_cross_entropy_with_logits(lg, y, pos_weight=pw, reduction="none")
    loss = (bce * m).sum() / ntot.clamp_min(1.0)
    with torch.no_grad():
        p = torch.sigmoid(lg.detach())
        mp, mn = m * y, m * (1.0 - y)
        sep = ((p * mp).sum() / mp.sum().clamp_min(1.0)
               - (p * mn).sum() / mn.sum().clamp_min(1.0))
    return loss, {"contact": float(loss.detach()), "c_sep": float(sep),
                  "c_pos": float(npos), "c_frac": float(npos / ntot.clamp_min(1.0))}


def wrap_flow_loss(base):
    """base(out, batch, ...) -> (total, parts). Same signature out, contact term added."""

    def flow_loss(out, batch, *a, **kw):
        total, parts = base(out, batch, *a, **kw)
        l_c, p_c = contact_term(out, batch)
        if l_c is not None:
            total = total + W_CONTACT * l_c
            parts = dict(parts, **p_c)
        return total, parts

    flow_loss.base = base
    return flow_loss
