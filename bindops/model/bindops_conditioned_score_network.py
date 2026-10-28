"""Wraps DiffPepBuilder's ScoreNetwork with a BINDops-conditioned adapter, without modifying
the original (vendored, read-only) source. Everything from the original ScoreNetwork is
frozen; the ONLY trainable parameters are the new adapter (a small MLP projecting BINDops'
per-residue interface score into the model's node-embedding space, added right where
init_node_embed is first computed -- before the main IPA network runs).

Both the DiffPepBuilder backbone and BINDops v4 itself are frozen in this first run, per the
agreed plan: get a real frozen-weights baseline first, then compare against a second run
that allows some weights to become trainable, once the frozen result is in hand.
"""
import sys, os
sys.path.insert(0, os.environ.get('DPB_BASE', '/8TBDrive2/DiffPepBuilder'))
import torch
from torch import nn
from model.score_network import ScoreNetwork
from data import all_atom, residue_constants
from openfold.utils.rigid_utils import Rigid


class BindopsAdapter(nn.Module):
    """Projects a per-residue BINDops interface score (scalar in [0,1]) into the model's
    node-embedding space. Small 2-layer MLP, the only trainable module in the frozen-backbone
    run."""
    def __init__(self, node_embed_size, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, node_embed_size),
        )
        # zero-init the final layer so the adapter starts as a true no-op (identical to the
        # unconditioned frozen backbone at step 0) -- standard adapter-init practice, lets
        # training show a clean, honest "does conditioning help at all" curve from a known
        # baseline rather than an arbitrary random perturbation.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, bindops_score):
        # bindops_score: [batch, N_res] -> [batch, N_res, node_embed_size]
        return self.net(bindops_score.unsqueeze(-1))


class BindopsConditionedScoreNetwork(nn.Module):
    """Wraps a real, loaded ScoreNetwork. Freezes it entirely, adds the trainable adapter.
    Forward signature matches the original ScoreNetwork.forward (same input_feats dict),
    plus expects input_feats['bindops_score'] (real per-residue BINDops v4 predictions,
    precomputed once per structure, not backpropagated through)."""
    def __init__(self, score_network: ScoreNetwork, freeze_backbone=True):
        super().__init__()
        self.score_network = score_network
        self.adapter = BindopsAdapter(score_network._model_conf.node_embed_size)
        if freeze_backbone:
            for p in self.score_network.parameters():
                p.requires_grad_(False)
            self.score_network.eval()  # frozen backbone stays in eval mode (no dropout/BN drift)

    def forward(self, input_feats):
        bb_mask = input_feats['res_mask'].type(torch.float32)
        fixed_mask = input_feats['fixed_mask'].type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        sn = self.score_network
        init_node_embed, init_edge_embed = sn.embedding_layer(
            aatype=input_feats['aatype'],
            seq_idx=input_feats['seq_idx'],
            t=input_feats['t'],
            fixed_mask=fixed_mask,
            self_conditioning_ca=input_feats['sc_ca_t'],
            esm_embed=input_feats.get('esm_embed', None),
        )
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]

        # === the only new part: real BINDops conditioning ===
        bindops_score = input_feats['bindops_score']  # [batch, N_res], real precomputed BINDops v4 scores
        node_embed = node_embed + self.adapter(bindops_score) * bb_mask[..., None]
        # ======================================================

        model_out = sn.score_model(node_embed, edge_embed, input_feats)
        node_embed = model_out['node_embed']
        edge_embed = model_out['edge_embed']

        torsion_norm, _, torsion_pred = sn.torsion_pred(node_embed)
        psi_pred = torsion_pred[..., 0, :]
        gt_psi = input_feats['torsion_angles_sin_cos'][..., 2, :]
        psi_pred = sn._apply_mask(psi_pred, gt_psi, 1 - fixed_mask[..., None])

        gt_chi = input_feats['torsion_angles_sin_cos'][..., 3:7, :]
        from model.score_network import NUM_TORSIONS
        if NUM_TORSIONS >= 5:
            chi_pred = torsion_pred[..., 1:5, :]
            chi_pred = sn._apply_mask(chi_pred, gt_chi, 1 - fixed_mask[..., None, None])
        else:
            chi_pred = gt_chi

        rigids_pred = model_out['final_rigids']
        atom37_pred, _, atom14_pred = all_atom.compute_allatom(
            rigids_pred, input_feats['aatype'], psi_pred, chi_pred)
        atom37_pred = atom37_pred.to(rigids_pred.device)
        atom14_pred = atom14_pred.to(rigids_pred.device)

        CA_IDX = residue_constants.atom_order['CA']
        ca_pos_pred = atom37_pred[..., CA_IDX, :]
        ca_dist_pred = torch.linalg.norm(
            ca_pos_pred[:, :, None, :] - ca_pos_pred[:, None, :, :], dim=-1)

        aa_logits_pred = sn.aa_type_pred(node_embed, edge_embed, ca_dist_pred)

        return {
            'psi': psi_pred, 'chi': chi_pred, 'torsion_norm': torsion_norm,
            'aa_logits': aa_logits_pred,
            'rot_score': model_out['rot_score'], 'trans_score': model_out['trans_score'],
            'rigids': rigids_pred.to_tensor_7(),
            'atom37': atom37_pred, 'atom14': atom14_pred, 'ca_dist': ca_dist_pred,
            'node_embed': node_embed,
        }
