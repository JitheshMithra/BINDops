"""Triangle-multiplicative pair stack -- the representational gap the 2026-10-13 audit found.

THE PROBLEM. DiffPepBuilder's edge representation is built ONCE, by an MLP over
cross-concatenated 1D features plus the self-conditioning distogram (model/embedding.py), and
is thereafter updated only by `EdgeTransition` (model/ipa_module.py:51), which recomputes edge
(i,j) from its two ENDPOINT node embeddings and nothing else. So edge (i,j) can never see edge
(i,k) and edge (k,j). The network therefore cannot represent, let alone enforce, the
constraint "i contacts k and k contacts j, so i and j are close" -- which is the whole content
of a contact map. That is exactly the machinery Boltz-2 has and this model does not, and it is
the standing explanation for why the model emits a well-formed peptide at an arbitrary site.

WHAT THIS ADDS. Triangle multiplicative update (AlphaFold2 Alg. 11/12), outgoing and incoming,
plus a pair transition, repeated N_LAYERS times over the edge tensor before it reaches the IPA
trunk. This is the minimum mechanism that propagates pairwise consistency.

THREE DELIBERATE CHOICES, each with its reason:

 1. LOW RANK. The stack runs in `c_pair` channels (default 64), down-projected from the model's
    256-channel edge embedding and projected back. Full-width triangle ops on 256 channels buy
    little here and cost 4x the activation memory on a 16 GB P100 that is already sharing the
    box with a live training run.

 2. EXACT NO-OP AT INIT. `out_proj` is zero-initialised, so at step 0 the returned edge tensor
    is bit-for-bit the input and the model is EXACTLY the pretrained DiffPepDock network. This
    is the same discipline the BINDops adapter uses, and it is what makes the 240k-step warm
    start survive an architecture change. verify_pair_stack.py asserts it numerically; without
    it, adding the stack would silently discard the pretrained weights' calibration.

 3. NO TRIANGLE ATTENTION. Its logits are [B, N, H, N, N]; at N=384, H=4 that is 226M entries
    (~0.45 GB in fp16) per tensor with several live at once, on top of the 12.6 GB the trainer
    already peaks at. Triangle multiplication carries the pairwise-consistency mechanism on its
    own, so it goes in first and alone. Adding attention is a separate, later, memory-budgeted
    change -- not a confound bundled into this one.

MEMORY IS BOUNDED BY THE EXISTING BATCHER, not by luck: `length_batching` caps
B * N^2 <= max_squared_res (150000), so every pair tensor here is at most 150000 * c_pair
elements (~19 MB in fp16) whatever the shape of the batch.
"""
import os
import torch
from torch import nn
import torch.utils.checkpoint as _cp

# Gradient checkpointing on each pair layer. MEASURED, not precautionary: without it the
# 4-layer stack peaked at 16.2 GB of this P100's 17.2 GB (94%) against a 13.4 GB baseline,
# i.e. under 1 GB of headroom for an 11 h run on a box that has already lost runs to an OOM
# hours in, and throughput fell to 1.8 cx/s from 2.7. Recomputing each block in the backward
# pass trades one extra forward of this (small) module for most of its stored activations,
# and changes the math not at all -- so the no-op-at-init and mechanism checks still hold.
_USE_CKPT = os.environ.get("PAIR_CKPT", "1") == "1"


class TriangleMultiplication(nn.Module):
    """AlphaFold2 Alg. 11 (outgoing) / Alg. 12 (incoming), in c_hidden channels.

    outgoing:  out_ij = sum_k  a_ik * b_jk
    incoming:  out_ij = sum_k  a_ki * b_kj
    """

    def __init__(self, c_pair, c_hidden, outgoing=True):
        super().__init__()
        self.outgoing = outgoing
        self.norm_in = nn.LayerNorm(c_pair)
        self.a_lin = nn.Linear(c_pair, c_hidden)
        self.a_gate = nn.Linear(c_pair, c_hidden)
        self.b_lin = nn.Linear(c_pair, c_hidden)
        self.b_gate = nn.Linear(c_pair, c_hidden)
        self.norm_out = nn.LayerNorm(c_hidden)
        self.out_lin = nn.Linear(c_hidden, c_pair)
        self.out_gate = nn.Linear(c_pair, c_pair)

    def forward(self, z, pair_mask):
        # pair_mask [B,N,N] -> every padded (i,j) contributes nothing to the sum over k, which
        # is what keeps a padded batch numerically identical to an unpadded one.
        m = pair_mask[..., None]
        s = self.norm_in(z)
        a = torch.sigmoid(self.a_gate(s)) * self.a_lin(s) * m
        b = torch.sigmoid(self.b_gate(s)) * self.b_lin(s) * m
        # bmm over channels is much cheaper than einsum here: [B,c,N,N] @ [B,c,N,N]^T
        a = a.permute(0, 3, 1, 2)
        b = b.permute(0, 3, 1, 2)
        if self.outgoing:                       # sum_k a_ik b_jk
            o = torch.matmul(a, b.transpose(-1, -2))
        else:                                   # sum_k a_ki b_kj
            o = torch.matmul(a.transpose(-1, -2), b)
        o = o.permute(0, 2, 3, 1)
        return torch.sigmoid(self.out_gate(s)) * self.out_lin(self.norm_out(o))


class PairTransition(nn.Module):
    def __init__(self, c_pair, expansion=2):
        super().__init__()
        self.norm = nn.LayerNorm(c_pair)
        self.net = nn.Sequential(
            nn.Linear(c_pair, c_pair * expansion), nn.ReLU(),
            nn.Linear(c_pair * expansion, c_pair))

    def forward(self, z):
        return self.net(self.norm(z))


class PairStack(nn.Module):
    """Low-rank triangle-multiplicative stack over the edge embedding. Exact no-op at init."""

    def __init__(self, c_edge, c_pair=None, c_hidden=None, n_layers=None):
        super().__init__()
        c_pair = int(os.environ.get("PAIR_C", "64")) if c_pair is None else c_pair
        c_hidden = int(os.environ.get("PAIR_C_HIDDEN", "32")) if c_hidden is None else c_hidden
        n_layers = int(os.environ.get("PAIR_LAYERS", "4")) if n_layers is None else n_layers
        self.c_pair, self.n_layers = c_pair, n_layers
        self.in_proj = nn.Linear(c_edge, c_pair)
        self.tri = nn.ModuleList()
        self.trans = nn.ModuleList()
        for _ in range(n_layers):
            self.tri.append(nn.ModuleList([
                TriangleMultiplication(c_pair, c_hidden, outgoing=True),
                TriangleMultiplication(c_pair, c_hidden, outgoing=False)]))
            self.trans.append(PairTransition(c_pair))
        self.norm_final = nn.LayerNorm(c_pair)
        self.out_proj = nn.Linear(c_pair, c_edge)
        # THE no-op guarantee: zero output projection -> z_out == z_in exactly at step 0, so the
        # 240k-step DiffPepDock warm start is preserved rather than silently invalidated.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # --- inter-chain contact head (CONTACT_LOSS=1) --------------------------------
        # The zero-init above is exactly why this arm came back null: out_proj is the
        # stack's ONLY consumer, so at step 0 the gradient reaching every triangle layer
        # is identically zero, and the flow loss it would have to earn its way into has no
        # term that depends on which receptor residue pairs with which peptide residue.
        # This head gives the pair representation a target of its own pair shape. It does
        # NOT touch out_proj, so the no-op-at-init guarantee on the edge path is unchanged
        # and the 240k-step warm start is still exact.
        self.contact_head = nn.Linear(c_pair, 1) if os.environ.get("CONTACT_LOSS") == "1" else None
        self.contact_logits = None

    @staticmethod
    def _block(z, pair_mask, tri_out, tri_in, trans):
        z = z + tri_out(z, pair_mask)
        z = z + tri_in(z, pair_mask)
        z = z + trans(z)
        return z * pair_mask[..., None]

    def forward(self, edge_embed, pair_mask):
        z = self.in_proj(edge_embed) * pair_mask[..., None]
        ckpt = _USE_CKPT and self.training and torch.is_grad_enabled()
        for (tri_out, tri_in), trans in zip(self.tri, self.trans):
            if ckpt:
                z = _cp.checkpoint(self._block, z, pair_mask, tri_out, tri_in, trans,
                                   use_reentrant=False)
            else:
                z = self._block(z, pair_mask, tri_out, tri_in, trans)
        zf = self.norm_final(z)
        if self.contact_head is not None:
            lg = self.contact_head(zf).squeeze(-1)
            # contacts are symmetric; symmetrising halves the effective parameters the head
            # has to spend on learning that i-j and j-i are the same question.
            self.contact_logits = 0.5 * (lg + lg.transpose(-1, -2))
        return edge_embed + self.out_proj(zf) * pair_mask[..., None]
