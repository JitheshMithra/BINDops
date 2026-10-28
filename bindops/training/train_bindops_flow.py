"""Option A -- BINDops-conditioned FLOW generator.

Stage 1 measured flow matching at oracle DockQ 0.536-0.583 against full diffusion's 0.312 on this
project's own data. Every BINDops conditioning experiment so far was built on the diffusion arm --
the one that lost. This moves the same zero-init adapter onto train_flow_v4.py.

Three runtime patches, no file under /8TBDrive2 is modified:

 1. MODEL. score_network.ScoreNetwork -> BindopsFlowModel: the real network plus an adapter MLP
    that injects BINDops v4's per-residue interface probability into the node embeddings.
    state_dict() deliberately exposes the BACKBONE's keys unprefixed. train_flow_v4 filters the
    warm-start checkpoint with `k in model.state_dict()` BEFORE calling load_state_dict, so a
    wrapper whose keys all read "score_network.*" would silently drop every pretrained weight
    (and hard-exit under --resume). Adapter weights ride under "adapter.*", absent from any stock
    checkpoint, so they stay zero-init on a warm start and reload on a resume.
 2. DATA. PlaceDataset subclasses PdbDataset and calls _process_csv_row, so patching that one
    method covers both trainers. It adds bindops_score and keeps the pkl's BINDops-centred frame
    (the stock loader re-centres on the TRUE peptide centroid, which leaks the answer).
    _assemble then tree.map_structures the whole dict and du.pad_feats pads every key along the
    residue dimension, so the score reaches the batch unshaped-by-hand -- verified, not assumed.
 3. STOP. train_flow_v4 already has an exact --max-steps, so no step patch is needed.

usage: BINDOPS_MODE=fixed|flexible train_bindops_flow.py <out_dir> <data_dir> [flow args...]
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os, sys, importlib.util, functools as fn
import numpy as np
import torch

MODE = os.environ["BINDOPS_MODE"]
# "control" = the unconditioned baseline: adapter stays zero and frozen, so it is a verified no-op,
# while data, steps, lr, seed and batching match the conditioned arms exactly. This is the arm that
# makes "conditioning helped" a testable claim rather than an assertion.
assert MODE in ("fixed", "flexible", "control"), MODE
BASE = os.environ.setdefault("DPB_BASE", "/8TBDrive2/DiffPepBuilder")
os.environ.setdefault("SPEEDDOCK_REPO", "/8TBDrive2/SpeedDocker")
os.environ.setdefault("DPB_NUM_TORSIONS", "3")   # every flow_v4 ckpt has the pre-v5 3-torsion head
os.environ["BASE_PATH"] = BASE
sys.path.insert(0, BASE)
sys.path.insert(0, os.environ["SPEEDDOCK_REPO"])

from bindops.model.bindops_conditioned_score_network import BindopsAdapter
from model import score_network as _sn
from data import pdb_data_loader, utils as du
import tree
from openfold.data import data_transforms

_RealScoreNetwork = _sn.ScoreNetwork


# === MSA/PSSM PROFILE CONDITIONING (2026-10-05) ==================================================
# Boltz gets a full MSA; BINDops so far got that signal compressed to ONE number per residue by the
# binding model. PROFILE=1 widens the adapter input to [bindops_score, 20-dim PSI-BLAST log-odds],
# so the docking network itself sees the evolutionary profile. Zero-init final layer is preserved,
# so step 0 is still an exact no-op and the DiffPepDock warm start is untouched.
# === REAL MSA FEATURES + REAL SITE SCORE (2026-10-15) ===========================================
# MSA_PROFILE_PKL: {gid: float16 [N, 23]} from build_profiles.py -- 20 UniRef90 log-odds, entropy,
#   gap fraction, log1p(depth), peptide rows zero. This is the "no MSA" hole: the model's only
#   sequence signal was ESM, a single-sequence LM that never sees this receptor's own alignment.
# SITE_SCORE_PKL: {gid: float32 [N]} from build_site_scores.py -- replaces the literal-zero
#   bindops_score with 1 - coord_num percentile, which the 2026-10-13 benchmark measured as the
#   whole content of the v4 field on peptide sites (ties 185-dim v4, p=0.23).
# Both are looked up by pdb_name and HARD-FAIL on a miss; see the module docstring of
# patch_features.py for why silence is the dangerous option here.
MSA_PKL = os.environ.get("MSA_PROFILE_PKL", "")
SITE_PKL = os.environ.get("SITE_SCORE_PKL", "")
MSA_NFEAT = 23
# per-feature divisors so every channel reaches the adapter at roughly unit scale:
# log-odds are clipped to +-8, entropy <= log(20) ~ 3.0, gap fraction <= 1, log1p(depth) <= ~9
_MSA_SCALE = None
_MSA, _SITE = None, None


def _msa_scale():
    global _MSA_SCALE
    if _MSA_SCALE is None:
        _MSA_SCALE = np.concatenate([np.full(20, 8.0), [3.0, 1.0, 9.0]]).astype(np.float32)
    return _MSA_SCALE


def _msa():
    global _MSA
    if _MSA is None:
        import pickle
        with open(MSA_PKL, "rb") as f:
            _MSA = pickle.load(f)
        print(f"[bindops-flow] loaded {len(_MSA)} MSA profiles ({MSA_NFEAT} feats) from "
              f"{MSA_PKL}", flush=True)
    return _MSA


def _site():
    global _SITE
    if _SITE is None:
        import pickle
        with open(SITE_PKL, "rb") as f:
            _SITE = pickle.load(f)
        print(f"[bindops-flow] loaded {len(_SITE)} site scores from {SITE_PKL}", flush=True)
    return _SITE


PROFILE = os.environ.get("BINDOPS_PROFILE", "0") == "1"
PROFILE_SCALE = float(os.environ.get("BINDOPS_PROFILE_SCALE", "10.0"))   # log-odds are ~[-13,13]
_PROFILES = None


def _profiles():
    global _PROFILES
    if _PROFILES is None:
        import pickle
        with open(os.environ.get("BINDOPS_PROFILE_PKL",
                  "/8TBDrive1/jithesh/docking_model/bindops_dock_data_full/pssm_profiles.pkl"), "rb") as f:
            _PROFILES = pickle.load(f)
        print(f"[bindops-flow] loaded {len(_PROFILES)} PSSM profiles", flush=True)
    return _PROFILES


class ProfileAdapter(torch.nn.Module):
    """Same shape as BindopsAdapter but over a per-residue VECTOR (score + 20-dim profile)."""

    def __init__(self, node_embed_size, in_dim, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden, node_embed_size))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, vec):
        return self.net(vec)


# === TRAINABLE POCKET MODEL (2026-10-05) =========================================================
# v4 is gradient-boosted trees: no gradients, so it could never be tuned by a docking objective.
# distill_bindops.py produced a differentiable student (held-out r=0.989 vs v4, same contact AUROC).
# FINETUNE=1 runs that student in-graph over [ScanNet, ISIS, TriGlue_const, PSSM window 180, PeSTo,
# coord_num] and feeds ITS output to the adapter, so the docking loss trains the site predictor.
# Peptide rows are masked to 0, matching the precomputed bindops_score convention.
FINETUNE = os.environ.get("BINDOPS_FINETUNE", "0") == "1"
FREEZE_STUDENT = os.environ.get("BINDOPS_FREEZE_STUDENT", "0") == "1"
STUDENT_LR_SCALE = float(os.environ.get("BINDOPS_STUDENT_LR_SCALE", "0.1"))
if FINETUNE:
    PROFILE = True          # the student needs the profile + components
_COMPS = None


def _comps():
    global _COMPS
    if _COMPS is None:
        import pickle
        with open(os.environ.get("BINDOPS_COMP_PKL",
                  "/8TBDrive1/jithesh/docking_model/bindops_dock_data_full/bindops_components.pkl"), "rb") as f:
            _COMPS = pickle.load(f)
        print(f"[bindops-flow] loaded {len(_COMPS)} component arrays", flush=True)
    return _COMPS


class StudentHead(torch.nn.Module):
    """Differentiable stand-in for BINDops v4, warm-started from the distilled weights."""

    def __init__(self, ckpt="/8TBDrive1/jithesh/docking_model/models/bindops_student.pt"):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(185, 256), torch.nn.GELU(),
                                       torch.nn.Linear(256, 128), torch.nn.GELU(),
                                       torch.nn.Linear(128, 1))
        d = torch.load(ckpt, map_location="cpu")
        self.net.load_state_dict({k.replace("net.", "", 1): v for k, v in d["state"].items()})
        self.register_buffer("mu", torch.tensor(d["mu"], dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(d["sd"], dtype=torch.float32))
        print(f"[bindops-flow] StudentHead warm start: r={d['r']:.4f} vs v4, "
              f"AUROC {d['auroc_student']:.4f} vs {d['auroc_v4']:.4f}", flush=True)

    def forward(self, profile, comp, rec_mask):
        # profile [B,N,20] -> centred 9-window [B,N,180]. Peptide rows are zero in `profile`, so
        # windowing the whole padded array gives receptor residues exactly the zero context the
        # receptor-only windowing used at distillation time.
        x = profile.transpose(1, 2)                                  # [B,20,N]
        x = torch.nn.functional.pad(x, (4, 4))
        x = x.unfold(-1, 9, 1)                                       # [B,20,N,9]
        win = x.permute(0, 2, 3, 1).reshape(profile.shape[0], profile.shape[1], 180)
        f = torch.cat([comp[..., 0:1], comp[..., 1:2], comp[..., 2:3], win,
                       comp[..., 3:4], comp[..., 4:5]], dim=-1)
        f = (f - self.mu) / self.sd
        return torch.sigmoid(self.net(f).squeeze(-1)) * rec_mask


class BindopsFlowModel(torch.nn.Module):
    """ScoreNetwork + zero-init BINDops adapter, with a backbone-transparent state_dict."""

    def __init__(self, model_conf, diffuser):
        super().__init__()
        self.score_network = _RealScoreNetwork(model_conf, diffuser)
        # adapter input = [site score] + per-residue feature vector.
        #   MSA on : 1 + 23 = 24      PSSM on: 1 + 20 = 21      neither: scalar score
        _in = 1 + (MSA_NFEAT if MSA_PKL else 20)
        self.adapter = (ProfileAdapter(model_conf.node_embed_size, _in) if PROFILE
                        else BindopsAdapter(model_conf.node_embed_size))
        if PROFILE:
            print(f"[bindops-flow] ProfileAdapter in_dim={_in} "
                  f"({'MSA' if MSA_PKL else 'PSSM'} features)", flush=True)
        self.student = StudentHead() if FINETUNE else None
        if self.student is not None:
            if FREEZE_STUDENT:
                for q in self.student.parameters():
                    q.requires_grad_(False)
                print("[bindops-flow] student FROZEN (control arm)", flush=True)
            elif STUDENT_LR_SCALE != 1.0:
                for q in self.student.parameters():
                    q.register_hook(lambda g: g * STUDENT_LR_SCALE)
                print(f"[bindops-flow] student trainable, grad scale {STUDENT_LR_SCALE}", flush=True)
        self._model_conf = model_conf
        if MODE == "fixed":
            for p in self.score_network.parameters():
                p.requires_grad_(False)
        if MODE == "control":
            for p in self.adapter.parameters():
                p.requires_grad_(False)   # frozen at zero-init -> exact no-op, backbone trains normally

    # --- keys the trainer's warm-start filter expects -------------------------------------
    def state_dict(self, *a, **kw):
        sd = dict(self.score_network.state_dict(*a, **kw))
        for k, v in self.adapter.state_dict().items():
            sd[f"adapter.{k}"] = v
        if self.student is not None:
            # without this the fine-tuned pocket model is trained and then thrown away at every save
            for k, v in self.student.state_dict().items():
                sd[f"student.{k}"] = v
        return sd

    def load_state_dict(self, sd, strict=False):
        ad = {k[len("adapter."):]: v for k, v in sd.items() if k.startswith("adapter.")}
        stu = {k[len("student."):]: v for k, v in sd.items() if k.startswith("student.")}
        bb = {k: v for k, v in sd.items()
              if not k.startswith("adapter.") and not k.startswith("student.")}
        res = self.score_network.load_state_dict(bb, strict=False)
        if ad:
            self.adapter.load_state_dict(ad, strict=True)
        if stu and self.student is not None:
            self.student.load_state_dict(stu, strict=True)
        print(f"[bindops-flow] loaded backbone: missing={len(res.missing_keys)} "
              f"unexpected={len(res.unexpected_keys)}; adapter tensors={len(ad)}; "
              f"student tensors={len(stu)}", flush=True)
        return res

    def train(self, mode=True):
        super().train(mode)
        if MODE == "fixed":
            self.score_network.eval()   # frozen backbone: no dropout/BN drift
        return self

    def forward(self, input_feats):
        sn = self.score_network
        bb_mask = input_feats["res_mask"].type(torch.float32)
        fixed_mask = input_feats["fixed_mask"].type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]
        init_node_embed, init_edge_embed = sn.embedding_layer(
            aatype=input_feats["aatype"], seq_idx=input_feats["seq_idx"], t=input_feats["t"],
            fixed_mask=fixed_mask, self_conditioning_ca=input_feats["sc_ca_t"],
            esm_embed=input_feats.get("esm_embed", None))
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]
        # === the conditioning ===
        if FINETUNE:
            _score = self.student(input_feats["pssm_profile"].to(node_embed.dtype),
                                  input_feats["bindops_comp"].to(node_embed.dtype),
                                  fixed_mask)   # fixed_mask == 1 - ligand_mask == receptor mask
        else:
            _score = input_feats["bindops_score"]
        if PROFILE:
            # MSA features arrive pre-scaled by the loader; the legacy PSSM path keeps its single
            # PROFILE_SCALE divisor so old arms stay reproducible.
            _feat = (input_feats["msa_profile"].to(node_embed.dtype) if MSA_PKL
                     else input_feats["pssm_profile"].to(node_embed.dtype) / PROFILE_SCALE)
            _v = torch.cat([_score.unsqueeze(-1), _feat], dim=-1)
            node_embed = node_embed + self.adapter(_v) * bb_mask[..., None]
        else:
            node_embed = node_embed + self.adapter(_score) * bb_mask[..., None]
        # ========================
        model_out = sn.score_model(node_embed, edge_embed, input_feats)
        node_embed, edge_embed = model_out["node_embed"], model_out["edge_embed"]
        from model.score_network import NUM_TORSIONS
        from data import all_atom, residue_constants
        torsion_norm, _, torsion_pred = sn.torsion_pred(node_embed)
        psi_pred = torsion_pred[..., 0, :]
        gt_psi = input_feats["torsion_angles_sin_cos"][..., 2, :]
        psi_pred = sn._apply_mask(psi_pred, gt_psi, 1 - fixed_mask[..., None])
        gt_chi = input_feats["torsion_angles_sin_cos"][..., 3:7, :]
        if NUM_TORSIONS >= 5:
            chi_pred = sn._apply_mask(torsion_pred[..., 1:5, :], gt_chi, 1 - fixed_mask[..., None, None])
        else:
            chi_pred = gt_chi
        rigids_pred = model_out["final_rigids"]
        atom37_pred, _, atom14_pred = all_atom.compute_allatom(
            rigids_pred, input_feats["aatype"], psi_pred, chi_pred)
        atom37_pred = atom37_pred.to(rigids_pred.device); atom14_pred = atom14_pred.to(rigids_pred.device)
        CA = residue_constants.atom_order["CA"]
        ca = atom37_pred[..., CA, :]
        ca_dist = torch.linalg.norm(ca[:, :, None, :] - ca[:, None, :, :], dim=-1)
        return {"psi": psi_pred, "chi": chi_pred, "torsion_norm": torsion_norm,
                "aa_logits": sn.aa_type_pred(node_embed, edge_embed, ca_dist),
                "rot_score": model_out["rot_score"], "trans_score": model_out["trans_score"],
                "rigids": rigids_pred.to_tensor_7(), "atom37": atom37_pred, "atom14": atom14_pred,
                "ca_dist": ca_dist, "node_embed": node_embed}


class _ScoreNetworkProxy:
    """Swap ScoreNetwork only in the trainer's view. Rebinding the attribute on the real module
    breaks ScoreNetwork.__init__'s own super(ScoreNetwork, self) call."""
    ScoreNetwork = BindopsFlowModel

    def __getattr__(self, k):
        return getattr(_sn, k)


@fn.lru_cache(maxsize=100)
def _process_csv_row(self, processed_file_path):
    pf = du.read_pkl(processed_file_path)
    bindops_score = pf.pop("bindops_score").astype(np.float32)
    pf.pop("bindops_center", None)
    noise = os.environ.get("BINDOPS_CENTER_NOISE", "1") == "1"
    std = float(os.environ.get("BINDOPS_CENTER_NOISE_STD", "5.0"))
    # pkl is already centred on the BINDops pocket, so center_pos=0 keeps that frame; the jitter is
    # DiffPepBuilder's own augmentation and teaches tolerance to an imprecise centre.
    pf = du.parse_chain_feats(pf, center_pos=np.zeros(3), center_pos_noise=noise,
                              center_pos_noise_std=std)
    del pf["center_pos"]
    modeled_idx = pf.pop("modeled_idx")
    pf = tree.map_structure(lambda x: x[modeled_idx], pf)
    bindops_score = bindops_score[modeled_idx]
    _nm = os.path.basename(processed_file_path)[:-4]
    if SITE_PKL:
        # Replace the literal-zero bindops_score with the real per-residue site score.
        _s = _site().get(_nm)
        if _s is None:
            raise RuntimeError(
                f"SITE_SCORE_PKL is set but has no entry for {_nm}. Not substituting zeros: that "
                f"is how bindops_score came to be all-zero for every training case. Run "
                f"preflight_features.py to see split coverage, or rebuild with build_site_scores.py.")
        _s = np.asarray(_s, np.float32)
        if _s.shape[0] != len(bindops_score):
            raise RuntimeError(f"site score length {_s.shape[0]} != {len(bindops_score)} for {_nm}")
        bindops_score = _s[modeled_idx] if _s.shape[0] != len(modeled_idx) else _s
    if MSA_PKL:
        _m = _msa().get(_nm)
        if _m is None:
            raise RuntimeError(
                f"MSA_PROFILE_PKL is set but has no entry for {_nm}. Not substituting zeros -- see "
                f"preflight_features.py for split coverage.")
        _m = np.asarray(_m, np.float32)
        if _m.shape[0] != len(modeled_idx) and _m.shape[0] != len(bindops_score):
            raise RuntimeError(f"msa profile length {_m.shape[0]} unexpected for {_nm}")
        _m = _m[modeled_idx] if _m.shape[0] == len(modeled_idx) else _m
        _m = _m / _msa_scale()[None, :]
    if PROFILE and not MSA_PKL:
        _p = _profiles().get(_nm)
        _p = (np.zeros((len(bindops_score), 20), np.float32) if _p is None
              else np.asarray(_p, np.float32)[modeled_idx])
    if FINETUNE:
        _c = _comps().get(_nm)
        _c = (np.zeros((len(bindops_score), 5), np.float32) if _c is None
              else np.asarray(_c, np.float32)[modeled_idx])
    if os.environ.get("LEAK_SCRAMBLE") == "1":
        # LEAK TEST ONLY: wreck every native peptide coordinate (random rotation, +40 A shift, 3 A
        # per-atom noise). Under BLIND_PRIOR the generated poses must be bit-for-bit unchanged.
        from scipy.spatial.transform import Rotation as _R
        lig = pf["ligand_mask"].astype(bool)
        r = np.random.default_rng(12345)
        ap = pf["atom_positions"].copy()
        ap[lig] = ap[lig] @ _R.random(random_state=7).as_matrix().T + 40.0 + r.normal(0, 3.0, ap[lig].shape)
        pf["atom_positions"] = ap * pf["atom_mask"][..., None]
    cf = {"aatype": torch.tensor(pf["aatype"]).long(),
          "all_atom_positions": torch.tensor(pf["atom_positions"]).double(),
          "all_atom_mask": torch.tensor(pf["atom_mask"]).double(),
          "coordinate_bias": torch.tensor(pf["coordinate_bias"]).double()}
    cf = data_transforms.atom37_to_frames(cf)
    cf = data_transforms.make_atom14_masks(cf)
    cf = data_transforms.make_atom14_positions(cf)
    cf = data_transforms.atom37_to_torsion_angles()(cf)
    chain_idx, res_idx = pf["chain_index"], pf["residue_index"]
    mapping = {o: n for n, o in enumerate(np.unique(chain_idx))}
    new_chain_idx = np.vectorize(mapping.get)(chain_idx)
    new_res_idx = np.zeros_like(res_idx); start = 1
    for cid in np.unique(new_chain_idx).tolist():
        m = (new_chain_idx == cid).astype(int); cr = res_idx[m == 1]
        shift = start - np.min(cr); new_res_idx += (res_idx + shift) * m
        start = np.max(cr) + shift + 100
    return {"aatype": cf["aatype"], "seq_idx": new_res_idx, "chain_idx": new_chain_idx,
            "residx_atom14_to_atom37": cf["residx_atom14_to_atom37"],
            "residue_index": pf["residue_index"], "res_mask": pf["bb_mask"],
            "ligand_mask": pf["ligand_mask"], "atom37_pos": cf["all_atom_positions"],
            "atom37_mask": cf["all_atom_mask"], "atom14_pos": cf["atom14_gt_positions"],
            "coordinate_bias": cf["coordinate_bias"], "esm_embed": pf["esm_embed"],
            "rigidgroups_0": cf["rigidgroups_gt_frames"],
            "torsion_angles_sin_cos": cf["torsion_angles_sin_cos"],
            "alt_torsion_angles_sin_cos": cf["alt_torsion_angles_sin_cos"],
            "bindops_score": torch.tensor(bindops_score),
            **({"msa_profile": torch.tensor(_m)} if MSA_PKL else {}),
          **({"pssm_profile": torch.tensor(_p)} if (PROFILE and not MSA_PKL) else {}),
            **({"bindops_comp": torch.tensor(_c)} if FINETUNE else {})}


pdb_data_loader.PdbDataset._process_csv_row = _process_csv_row

# === BLIND PRIOR (2026-10-04 audit) ===================================================================
# speeddock's _placed_frames starts the peptide at its NATIVE centroid in its NATIVE bound conformation
# (rigidly rotated) -- the binding site and the bound shape are handed over, so DockQ measured
# "undo a rotation", not docking. BLIND_PRIOR replaces it, for training (flow interpolates from it)
# and inference (make_poseset starts from it) alike, with a prior that reads NOTHING of the peptide:
# every peptide residue gets an independent uniform random rotation and a translation drawn from
# N(c, BLIND_SIGMA^2 I). c is the only site information and is chosen by the arm:
#   "bindops"  -> c = frame origin = the BINDops-predicted pocket centre (receptor-only, leak-free)
#   "receptor" -> c = receptor CA centroid (no site information at all: fully blind)
# Unset BLIND_PRIOR keeps the legacy leaky placement, so the old runs stay reproducible.
BLIND = os.environ.get("BLIND_PRIOR", "")
BLIND_SIGMA = float(os.environ.get("BLIND_SIGMA", "10.0"))
if BLIND:
    assert BLIND in ("bindops", "receptor", "surface", "native", "head"), BLIND
    NATIVE_OFFSET = float(os.environ.get("BLIND_NATIVE_OFFSET", "0.0"))
    SURF_DEPTH = float(os.environ.get("SURF_DEPTH", "6.0"))
    SURF_JITTER = float(os.environ.get("SURF_JITTER", "4.0"))
    # HEAD PRIOR (2026-10-17). Same draw as the surface prior -- one anchor per POSE,
    # SURF_JITTER spread -- but the anchor is a PREDICTED patch centre instead of an
    # arbitrary exposed CA, so this is a one-variable A/B against BLIND_PRIOR=surface.
    # Centres come from head_centres.py (greedy-NMS patch centres + the TRAIN-measured
    # +3.05 A surface push), keyed by pdb_name, and are chosen per pose from the pose
    # seed so chunked generation picks the SAME centre as a single-process run.
    HEAD_CENTRES = os.environ.get("HEAD_CENTRES", "")
    _HEADC, _HEADMISS = {}, [0, 0]
    if BLIND == "head":
        assert HEAD_CENTRES, "BLIND_PRIOR=head requires HEAD_CENTRES=<pkl>"
        import pickle as _pk
        with open(HEAD_CENTRES, "rb") as _fh:
            _HEADC = _pk.load(_fh)
        print(f"[bindops-flow] head prior: {len(_HEADC)} targets from {HEAD_CENTRES}", flush=True)
    from speeddock import dataset as _sd
    from scipy.spatial.transform import Rotation as _Rot

    def _blind_frames(self, gt_bb_rigid, diffused_mask, rng):
        R = gt_bb_rigid.get_rots().get_rot_mats().numpy().copy()
        t = gt_bb_rigid.get_trans().numpy().copy()
        m = np.asarray(diffused_mask).astype(bool)
        P = int(m.sum())
        R2, t2 = R.copy(), t.copy()
        R2[m] = _Rot.random(P, random_state=int(rng.integers(2**31))).as_matrix()
        if BLIND == "head":
            C = _HEADC.get(getattr(self, "_cur_name", None))
            si, ii = getattr(self, "_cur_seed", None), getattr(self, "_cur_idx", None)
            if C is not None and len(C) and si is not None and ii is not None:
                base = np.asarray(C, dtype=np.float64)[int(si - 1000 * ii) % len(C)]
                t2[m] = base + rng.normal(0.0, SURF_JITTER, (P, 3))
                _HEADMISS[0] += 1
                return R2, t2, m
            _HEADMISS[1] += 1   # no centre for this target: fall back to the surface prior
        if BLIND in ("surface", "head"):
            # SURFACE PRIOR (2026-10-10). The centroid prior is mis-specified for real
            # docking: native peptide centroids sit a median 23.4 A from the receptor CA
            # centroid on genuine complexes vs 11.4 A on the artificial same-chain splits
            # that were 87% of the old training set, so only 39% of genuine sites fell
            # within 2 sigma of it (provenance_audit/objective_alignment.py). Real
            # peptides bind the SURFACE, so draw the start centre just outside an exposed
            # receptor CA. Measured, 400 genuine cases at N=32 (prior_design.py):
            # hit rate within 6 A 22.0% -> 29.0%, E[min dist] 12.40 -> 8.45 A. Widening
            # the centroid sigma instead makes it WORSE (s15 16.5%, s20 12.2%).
            rec = t[~m]
            if len(rec) >= 10:
                cen = rec.mean(0)
                outw = rec - cen
                nrm = np.linalg.norm(outw, axis=1, keepdims=True)
                nrm[nrm < 1e-6] = 1e-6
                outw = outw / nrm
                dd = np.linalg.norm(rec[:, None, :] - rec[None, :, :], axis=-1)
                nbr = (dd < 10.0).sum(1)
                keep = np.where(nbr <= np.percentile(nbr, 40))[0]   # convex / exposed
                idx = keep if len(keep) >= 5 else np.arange(len(rec))
                # ONE anchor per POSE, not one per residue (2026-10-15 fix). prior_design.py
                # validated this prior with n = number of DRAWS (candidate pocket centres, one
                # per pose) and reported hit 29% / E[min d] 8.45 A. The call below was written
                # with n = P = number of PEPTIDE RESIDUES, so every residue of a single peptide
                # was anchored at an INDEPENDENT surface site: measured start consecutive CA-CA
                # 40.5 A against a real peptide bond at 3.8 A, peptide span 66 A against a
                # native 12 A, and -- because the mean of P independent surface points is the
                # receptor CENTRE -- a start centroid at 0.73x the receptor radius, i.e.
                # BURIED, inverting the one fact ("peptides bind the surface") that this prior
                # exists to encode. Re-measured on 150 train targets at N=32:
                #   shipped (per-residue)  hit  6.0% / E[min d] 14.48 A
                #   per-pose (this fix)    hit 24.7% / E[min d]  9.02 A
                #   old centroid_s10       hit  4.0% / E[min d] 21.96 A
                # So the shipped prior was no better than the one it replaced AND it destroyed
                # the peptide's chain geometry. SURF_JITTER stays at 4.0 so this is a
                # one-variable change against the validated design.
                pick = rng.choice(idx, 1)[0]
                base = rec[pick] + outw[pick] * SURF_DEPTH
                t2[m] = base + rng.normal(0.0, SURF_JITTER, (P, 3))
            else:                       # tiny receptor: fall back to the centroid prior
                t2[m] = rec.mean(0) + rng.normal(0.0, BLIND_SIGMA, (P, 3))
        elif BLIND == "native":
            # ORACLE-SITE ARM -- DIAGNOSTIC ONLY, DELIBERATE LEAK, NEVER SHIP A NUMBER FROM IT.
            # t[m] is the GROUND-TRUTH peptide translation, so its mean is the true peptide
            # centroid: perfect localization, handed over for free. R2[m] above is still a
            # uniform random rotation, so the BOUND CONFORMATION is NOT leaked -- only the site.
            # Run with BLIND_SIGMA=4.0 to match the surface arm's SURF_JITTER, which makes this
            # a one-variable A/B against it: same spread, right place vs guessed place.
            c = t[m].mean(0)
            if NATIVE_OFFSET > 0.0:
                # Displace by a fixed distance in a per-target-deterministic direction. Seeded
                # from the geometry, NOT from rng, because the 4 seed-aligned pose chunks run as
                # separate processes and must all use the SAME wrong centre -- otherwise the 32
                # poses scatter around 4 different centres and the sweep measures nothing.
                import hashlib as _hl
                _key = (np.round(c, 3).tobytes()
                        + np.round(t[~m].mean(0), 3).tobytes()
                        + b"%d" % len(t))
                _seed = int(_hl.md5(_key).hexdigest()[:8], 16)
                _d = np.random.default_rng(_seed).normal(size=3)
                _d = _d / max(float(np.linalg.norm(_d)), 1e-6)
                c = c + _d * NATIVE_OFFSET
            t2[m] = c + rng.normal(0.0, BLIND_SIGMA, (P, 3))
        else:
            c = np.zeros(3) if BLIND == "bindops" else t[~m].mean(0)  # receptor only
            t2[m] = c + rng.normal(0.0, BLIND_SIGMA, (P, 3))
        return R2, t2, m

    _sd.PlaceDataset._placed_frames = _blind_frames
    if BLIND == "head":
        _orig_assemble = _sd.PlaceDataset._assemble
        def _assemble_named(self, idx, rng):
            try:
                row = self.csv.iloc[idx]
                self._cur_name = (row["pdb_name"] if "pdb_name" in row.index
                                  else os.path.basename(row["processed_path"])[:-4])
            except Exception:
                self._cur_name = None
            self._cur_idx = int(idx)
            return _orig_assemble(self, idx, rng)
        _sd.PlaceDataset._assemble = _assemble_named
        _orig_eval = _sd.PlaceDataset.eval_item
        def _eval_seeded(self, idx, seed):
            self._cur_seed = int(seed)
            return _orig_eval(self, idx, seed)
        _sd.PlaceDataset.eval_item = _eval_seeded
        import atexit as _ax
        _ax.register(lambda: print(f"[bindops-flow] head prior: {_HEADMISS[0]} draws from a "
                                   f"predicted centre, {_HEADMISS[1]} fell back to surface",
                                   flush=True))
    if BLIND == "native":
        print("[bindops-flow] *** ORACLE-SITE ARM: the TRUE peptide centroid is being handed "
              "to the prior (offset %.1f A). DIAGNOSTIC ONLY -- never publish a number from "
              "this arm. ***" % NATIVE_OFFSET, flush=True)
    print(f"[bindops-flow] BLIND_PRIOR={BLIND} sigma={BLIND_SIGMA}"
          + (f" surf_depth={SURF_DEPTH} surf_jitter={SURF_JITTER}"
             if BLIND == "surface" else "")
          + (" (TRUE peptide centroid READ -- oracle arm)" if BLIND == "native"
             else " (native peptide pose never read)"), flush=True)

# Which trainer to drive. Defaults to v4 so every existing caller is unchanged; set
# FLOW_TRAINER to .../docking_model/train_flow_v5.py for the arms that want the validation
# loop and recycling.
_TRAINER = os.environ.get("FLOW_TRAINER", _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "train_flow_v5.py"))
print(f"[bindops-flow] trainer: {_TRAINER}", flush=True)
spec = importlib.util.spec_from_file_location("flow_trainer", _TRAINER)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
T.score_network = _ScoreNetworkProxy()

if __name__ == "__main__":
    print(f"[bindops-flow] MODE={MODE}", flush=True)
    T.main()
