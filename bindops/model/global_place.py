"""GLOBAL PLACEMENT HEAD -- the architectural variable this model has never had.

THE GAP, read from both sources rather than guessed. RAPiDock (models/diffusion.py) predicts, per
complex, ONE global translation and ONE global rotation, with the translation's magnitude scaled by
a learned function of the noise level:

    global_pred = self.final_conv(pep_node_attr, center_edge_index, ..., out_nodes=num_graphs)
    tr_pred = tr_pred / tr_norm * self.tr_final_layer(cat([tr_norm, graph_sigma_emb]))

Our IpaScore has no such variable. Every block does

    rigid_update = bb_update_b(node_embed * diffuse_mask)
    curr_rigids  = curr_rigids.compose_q_update_vec(rigid_update, diffuse_mask)

-- a PER-RESIDUE update composed in each residue's OWN local frame. To move a 12-mer 20 A to a
different patch, twelve residues must each independently emit a large update that happens to
agree, while those same twelve updates are also the only thing maintaining the peptide bond. The
model is structurally better at refining a pose than at placing one, which is exactly the measured
failure: right depth (centroid 1.35x receptor Rg vs native 1.39x), wrong patch (2.6% of poses land
within 6 A of the true site, and poses that DO land there are 81.7% acceptable).

WHY NOT COPY RAPiDock LITERALLY. Their 3-vector is produced by an equivariant tensor-product
convolution, so it rotates with the input. Our node embeddings are INVARIANT scalars; an MLP from
them to a 3-vector would emit a translation in whatever arbitrary global frame the coordinates
happen to arrive in, and would have to memorise that frame. That is not a head, it is a bug.

WHAT THIS DOES INSTEAD -- a POINTER plus a RADIAL OFFSET.

    target = sum_i softmax(logit)_i * CA_i          over RECEPTOR residues only
    n_hat  = (target - receptor_centroid) / |...|   outward direction at the pointed patch
    delta  = tanh(gate) * (target - peptide_centroid)  +  R_MAX * tanh(radial) * n_hat
    peptide translations += delta

Equivariance is then free and exact: `target`, `receptor_centroid` and `peptide_centroid` are all
built from input coordinates, so `delta` rotates and translates with the complex, while `gate` and
`radial` are scalars read off invariant features. Nothing has to memorise a frame.

WHY THE RADIAL TERM EXISTS -- it is a gate-run finding, not decoration. With the pointer alone,
`target` is a weighted mean of receptor CAs, so every destination it can name lies inside the
CONVEX HULL of those CAs -- and the true peptide centroid does not: the peptide sits ON the
surface, outside the hull. Measured over 86 val complexes (verify_place.py CHECK 7,
reach_place.py): the pointer the loss's own soft target implies lands a median 8.35 A from the
true centroid and a one-hot pointer on the nearest receptor CA still lands 6.48 A away, in both
cases systematically INWARD -- 96% of cases have a positive outward component, median +5.60 A.
Shipping the pointer alone would have trained the peptide to sit ~6-8 A too deep, on a model that
already interpenetrates (1.39 A min approach, 3x native contacts). With the radial term the
reachable optimum is a median 0.13 A, 79% of complexes under 2 A and 99% under 6 A, so the
parameterisation imposes no meaningful ceiling of its own. One scalar, bounded at R_MAX, buys a
median +7.05 A.

THREE DELIBERATE CHOICES:

 1. NO GLOBAL ROTATION. Our diagnosis is a placement failure, not an orientation failure, and the
    per-residue IPA updates already set orientation. A global rotation also needs an equivariant
    construction of its own (the pointer trick does not extend to SO(3) for free), so it is a
    separate arm rather than a confound bundled into this one.

 2. EXACT NO-OP AT INIT, WITH A LIVE GRADIENT IN EVERY BRANCH. gate and radial are both
    zero-initialised and tanh(0) = 0, so at step 0 delta is identically zero and the model is
    bit-for-bit the pretrained network -- the discipline that keeps the 240k-step DiffPepDock warm
    start from being silently discarded. Crucially the two terms are SUMMED, not multiplied: each
    is a zero scalar times a NONZERO vector, so tanh'(0) = 1 makes both gradients nonzero at step
    0. This is the trap the pair-stack arm fell into -- its only consumer was a zero-init out_proj
    whose gradient at step 0 was identically zero, so it could never begin to learn. A no-op must
    be arranged so that exactly one factor of each product is zero, never both.

 3. CONDITIONED ON t. The flow's t runs 1 -> 0 over sampling, and the displacement needed at t=1
    (pure prior) is far larger than at t=0.1 (nearly placed). This is the one idea taken straight
    from RAPiDock's noise-scaled magnitude, and it costs two input features.

A NOTE ON PRECISION, because it cost a gate failure. openfold's Rigid.__init__ hard-casts its
translation to float32 ("Force full precision", rigid_utils.py:901), so anything routed through a
Rigid cannot be checked at float64 tolerance -- the end-to-end equivariance error is float32
round-off times the COORDINATE magnitude (~1e-6 A at 40 A coordinates), not times |delta|. The
geometry therefore lives in `displace()`, a pure dtype-preserving function, so the construction
can be verified in exact arithmetic separately from the float32 wiring.

knobs: PLACE_C_HIDDEN=128  PLACE_TEMP=1.0  PLACE_R_MAX=10.0
"""
import os

import torch
from torch import nn
from openfold.utils.rigid_utils import Rigid

# R_MAX = 20 A, set by measurement (reach_place.py, n=86 val complexes). The radial offset the
# reachable optimum actually wants has median 9.75 A, and at R_MAX=10 that sat ON the boundary:
# tanh would have had to saturate at 1, where its gradient vanishes, so the one scalar the design
# leans on would have been hardest to train exactly where it matters. At 20 A the typical optimum
# lands at tanh ~ 0.49, in the responsive part of the curve, and the bound still cannot become a
# second uncontrolled translation (the gate's own reach is already tens of angstroms).
R_MAX = float(os.environ.get("PLACE_R_MAX", "20.0"))

# --- HARD POINTER (see patch_place_hard.py for the measurement that forced this) --------------
# A convex combination of receptor CAs lands inside their hull, and the weighted mean of a
# near-uniform softmax IS the receptor centroid -- 28 A from a surface site, and averaging also
# destroys multimodality (a pointer torn between patches lands between them, in the core). The
# first real run measured the pointer FLAT (step coef +0.000099, CI straddling zero, after
# controlling for the batch-size confound r=-0.996) with the gate correctly collapsing to ~0.01.
# Hard selection makes every destination a REAL receptor residue position, so committing is safe.
# PLACE_HARD=0 restores the original convex combination; PLACE_TOPK>1 averages the top k (not the
# default: scattered top-k means a central mean again, the failure being fixed).
HARD = os.environ.get("PLACE_HARD", "1") != "0"
TOPK = int(os.environ.get("PLACE_TOPK", "1"))


def harden(w):
    """Straight-through: forward = hard top-k mean weights, backward = the softmax."""
    if not HARD:
        return w
    k = max(1, TOPK)
    idx = w.topk(k, dim=-1).indices
    hard = torch.zeros_like(w).scatter_(-1, idx, 1.0 / k)
    return (hard - w).detach() + w


def _pool(x, m):
    """Masked mean over residues -> [B, C]. Dtype follows x."""
    return (x * m[..., None].to(x.dtype)).sum(1) / m.sum(1, keepdim=True).clamp_min(1.0).to(x.dtype)


def displace(trans, w, gate, radial, pep_mask, rec_mask, eps=1e-6):
    """The geometry, as a pure function so it can be tested in exact arithmetic.

    trans [B,N,3] coordinates; w [B,N] pointer weights (zero off the receptor, summing to 1);
    gate, radial [B,1] invariant scalars in [-1,1]; masks [B,N].

    Every term is built from `trans` by weighted means and differences, so for any rigid motion
    x -> Rx + u the outputs satisfy target -> R.target + u and delta -> R.delta exactly.
    """
    w = w.to(trans.dtype)
    target = (w[..., None] * trans).sum(1)                       # [B,3] convex comb of CAs
    pep_cent = _pool(trans, pep_mask)                            # [B,3]
    rec_cent = _pool(trans, rec_mask)                            # [B,3]
    v = target - rec_cent
    # Outward unit direction at the pointed-at patch. Divided by (|v| + eps) rather than |v|:
    # when the pointer is diffuse -- as it is at init, where `target` sits ~0.1 A from the
    # receptor centroid -- there is no identified patch and no meaningful outward direction, and
    # this makes n_hat shrink to zero instead of becoming an arbitrary unit vector.
    n_hat = v / (v.norm(dim=-1, keepdim=True) + eps)
    delta = gate.to(trans.dtype) * (target - pep_cent) \
        + R_MAX * radial.to(trans.dtype) * n_hat
    return delta, target, pep_cent


class GlobalPlacement(nn.Module):
    """One pointer-based global translation of the peptide per complex. No-op at init."""

    def __init__(self, c_s, c_hidden=None):
        super().__init__()
        c_hidden = int(os.environ.get("PLACE_C_HIDDEN", "128")) if c_hidden is None else c_hidden
        self.temp = float(os.environ.get("PLACE_TEMP", "1.0"))
        self.c_s = c_s
        # pointer: [ node_i ; pooled peptide ; t ] -> one logit for receptor residue i.
        # Conditioning on the pooled peptide is what makes it a DOCKING pointer rather than a
        # generic "is this residue interface-like" predictor -- the same receptor should be
        # pointed at differently for different peptides.
        self.ptr = nn.Sequential(
            nn.LayerNorm(2 * c_s + 1), nn.Linear(2 * c_s + 1, c_hidden), nn.ReLU(),
            nn.Linear(c_hidden, c_hidden), nn.ReLU(), nn.Linear(c_hidden, 1))
        # two invariant scalars: how far along (target - centroid) to travel, and how far to sit
        # outward of the pointed patch along its own surface normal.
        self.scal = nn.Sequential(
            nn.LayerNorm(2 * c_s + 1), nn.Linear(2 * c_s + 1, c_hidden), nn.ReLU(),
            nn.Linear(c_hidden, 2))
        # the no-op: zero-init the LAST scalar layer only. tanh(0) = 0 -> delta == 0 exactly,
        # while tanh'(0) = 1 and both multiplicands are nonzero -> the gradient into these
        # weights is alive. The pointer is NOT zero-initialised: it has its own loss term, and
        # zeroing both factors of a product is how the pair stack arm came out null.
        nn.init.zeros_(self.scal[-1].weight)
        nn.init.zeros_(self.scal[-1].bias)
        self.stash = None

    def forward(self, node_embed, rigids, input_feats):
        node_mask = input_feats["res_mask"].type(torch.float32)
        lig = input_feats["ligand_mask"].type(torch.float32)
        pep_mask = lig * node_mask
        rec_mask = (1.0 - lig) * node_mask
        trans = rigids.get_trans()                                   # [B,N,3], angstroms
        t = input_feats["t"].reshape(-1, 1).to(trans.dtype)           # [B,1]

        # The head is built from model_conf.node_embed_size, but what arrives here is the IPA
        # trunk's output width (ipa.c_s). Those are the same number in every config we train,
        # and ConfidenceHead already relies on it -- but a silent mismatch would surface as a
        # shape error several frames deep inside nn.Sequential, so it is named here instead.
        if node_embed.shape[-1] != self.c_s:
            raise RuntimeError(
                f"[place] node_embed width {node_embed.shape[-1]} != head c_s {self.c_s}: "
                f"model_conf.node_embed_size and ipa.c_s have diverged in this config")

        pooled_pep = _pool(node_embed, pep_mask)                      # [B,C]
        B, N, _ = node_embed.shape
        feat = torch.cat([node_embed,
                          pooled_pep[:, None, :].expand(B, N, self.c_s),
                          t[:, None, :].to(node_embed.dtype).expand(B, N, 1)], dim=-1)
        logits = self.ptr(feat).squeeze(-1) / self.temp               # [B,N]
        # softmax over RECEPTOR residues only. A float mask would leave padded and peptide
        # residues with a small but nonzero weight, which would silently let the pointer aim at
        # the peptide itself; -inf makes them contribute exactly zero.
        logits = logits.masked_fill(rec_mask < 0.5, float("-inf"))
        # a degenerate example with no receptor residue at all would make the whole row -inf and
        # the softmax NaN. Cannot happen in this corpus (every case is two real chains) but a NaN
        # here would poison the whole batch's gradient, so it is guarded rather than assumed.
        has_rec = rec_mask.sum(1) > 0.5                               # [B]
        logits = torch.where(has_rec[:, None], logits, torch.zeros_like(logits))
        # Softmax in fp16 under AMP loses the tail of a 200-way distribution, so half precision
        # is upcast -- but NOT downcast: forcing .float() here would silently cap the module at
        # single precision and make an exact-arithmetic equivariance check vacuous.
        sdt = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
        w = torch.softmax(logits.to(sdt), dim=-1) * rec_mask.to(sdt)  # [B,N]
        # the pointer the LOSS sees stays soft (the CE needs a distribution); only the pointer the
        # GEOMETRY sees is hardened, so the classification gradient is unchanged.
        w_geo = harden(w)

        s = torch.tanh(self.scal(torch.cat([_pool(node_embed, rec_mask), pooled_pep,
                                            t.to(node_embed.dtype)], dim=-1)))   # [B,2]
        gate, radial = s[:, :1], s[:, 1:]
        delta, target, pep_cent = displace(trans, w_geo, gate, radial, pep_mask, rec_mask)
        delta = torch.where(has_rec[:, None], delta, torch.zeros_like(delta))

        new_trans = trans + delta[:, None, :] * pep_mask[..., None].to(delta.dtype)
        # stash for the loss and the step log. Rebuilt every forward and cleared by the caller
        # immediately after use: holding these on the module would pin the graph past opt.step().
        self.stash = {"ptr_w": w, "ptr_w_geo": w_geo, "ptr_logits": logits,
                      "gate": gate.squeeze(-1),
                      "radial": (R_MAX * radial).squeeze(-1), "delta": delta,
                      "target": target, "pep_cent": pep_cent,
                      "rec_mask": rec_mask, "pep_mask": pep_mask}
        return Rigid(rigids.get_rots(), new_trans)
