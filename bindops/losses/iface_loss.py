"""Docking-aware flow loss. Drop-in replacement for speeddock.losses.flow_loss.

THE BUG THIS FIXES, measured 2026-10-16. speeddock/losses.py calls compute_fape with
frames_mask = ligand_mask*res_mask and positions masked to peptide atoms -- BOTH the frames and
the points come from the peptide alone. FAPE is invariant under a rigid motion of the body that
supplies both, so that term is mathematically incapable of seeing where the peptide sits on the
receptor. Measured on random frames: FAPE = 0.822370 with the peptide in place, and 0.822370
after moving it 5 A, 20 A, 60 A, or rotating it 115 degrees, while backbone L2 went 0.8 -> 3597.

So placement was supervised ONLY by that raw backbone L2 -- which is 95-99.9% of the total loss
across training (train_full_msa.log) and is an unclamped quadratic on 5 atoms in global
coordinates. The frame-aware, well-behaved term contributed nothing to docking, and the "fape"
number used for checkpoint selection is a CONFORMATION metric, which is why it disagreed with
val_loss in every arm.

THE FIX is AlphaFold-Multimer's: a second FAPE whose FRAMES are the receptor's and whose POINTS
are the peptide's, so the peptide's position is read in each receptor residue's local frame.
AF-Multimer clamps within-chain FAPE at 10 A and between-chain at 30 A, the looser clamp
existing to give gradient on badly wrong interfaces -- our exact regime, median top-1 ligand
RMSD ~37 A. torch.clamp has ZERO gradient past its cutoff, so at a 10 A clamp a peptide 40 A
away receives no interface gradient at all.

knobs: W_IFACE=1.0  IFACE_CLAMP=30.0  W_BB_IFACE=0.05  (bb is downweighted because the interface
term now carries placement; left at 0.5 it would still be ~95% of the loss and swamp it)
"""
import os

import torch
from openfold.utils.rigid_utils import Rigid
from openfold.utils.loss import compute_fape

W_IFACE = float(os.environ.get("W_IFACE", "1.0"))
IFACE_CLAMP = float(os.environ.get("IFACE_CLAMP", "30.0"))
W_BB_IFACE = float(os.environ.get("W_BB_IFACE", "0.05"))


def flow_loss(out, batch, w_fape=1.0, w_bb=None):
    w_bb = W_BB_IFACE if w_bb is None else w_bb
    loss_mask = (batch['ligand_mask'] * batch['res_mask']).float()      # [B,N] peptide frames
    B, N = loss_mask.shape
    pred_rigid = Rigid.from_tensor_7(out['rigids'], normalize_quats=True)
    gt_rigid = Rigid.from_tensor_7(batch['rigids_0'], normalize_quats=True)
    pred_pts = out['atom37'].reshape(B, N * 37, 3)
    gt_pts = batch['atom37_pos'].reshape(B, N * 37, 3)
    pts_mask = (loss_mask[:, :, None] * batch['atom37_mask']).reshape(B, N * 37)

    # conformation: peptide frames, peptide points. Invariant to placement BY CONSTRUCTION --
    # that is correct for what it measures, and is why it cannot be the only FAPE term.
    l_fape = compute_fape(pred_rigid, gt_rigid, loss_mask, pred_pts, gt_pts, pts_mask,
                          length_scale=10.0, l1_clamp_distance=10.0, eps=1e-4).mean()

    # placement: RECEPTOR frames, peptide points. This is the term that can see docking.
    rec_mask = (batch['res_mask'] * (1.0 - batch['ligand_mask'])).float()
    l_iface = compute_fape(pred_rigid, gt_rigid, rec_mask, pred_pts, gt_pts, pts_mask,
                           length_scale=10.0, l1_clamp_distance=IFACE_CLAMP, eps=1e-4).mean()

    pa, ga = out['atom37'][:, :, :5], batch['atom37_pos'][:, :, :5]
    am = batch['atom37_mask'][:, :, :5] * loss_mask[..., None]
    l_bb = (((pa - ga) ** 2).sum(-1) * am).sum() / am.sum().clamp_min(1.0)

    total = w_fape * l_fape + W_IFACE * l_iface + w_bb * l_bb
    return total, {'fape': l_fape.item(), 'iface': l_iface.item(), 'bb': l_bb.item()}
