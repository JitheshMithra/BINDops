"""Confirm the BINDops-conditioned flow model is wired correctly, rather than inferring it from
'the smoke test did not crash'. Four checks:

 1. warm start really loaded (a wrapper whose state_dict keys are prefixed would silently drop
    every pretrained weight and the model would be random)
 2. trainable parameter counts match the mode (fixed = adapter only)
 3. bindops_score reaches the batch with real per-residue values, not zeros
 4. the zero-init adapter is an EXACT no-op at step 0 -- the conditioned model must reproduce the
    unconditioned model bit for bit, which proves the injection point is correct and harmless
    before training, and that any later difference is learned rather than a wiring artefact
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os, sys
os.environ.setdefault("BINDOPS_MODE", "fixed")
os.environ.setdefault("DPB_BASE", "/8TBDrive2/DiffPepBuilder")
os.environ.setdefault("SPEEDDOCK_REPO", "/8TBDrive2/SpeedDocker")
os.environ.setdefault("DPB_NUM_TORSIONS", "3")
import torch
from bindops.training import train_bindops_flow as B
from hydra import initialize_config_dir, compose
from data import se3_diffuser, utils as du, pdb_data_loader
from speeddock.dataset import PlaceDataset

BASE = os.environ["DPB_BASE"]
D = "/8TBDrive1/jithesh/docking_model/bindops_dock_data"
with initialize_config_dir(version_base=None, config_dir=f"{BASE}/config"):
    conf = compose(config_name="base", overrides=[
        f"data.train_csv_path={D}/metadata_train.csv", f"data.val_csv_path={D}/metadata_val.csv",
        "experiment.use_ddp=False", "experiment.batch_size=2", "experiment.num_loader_workers=0"])
diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)

print("1) WARM START")
m = B.BindopsFlowModel(conf.model, diffuser)
blob = du.read_pkl(f"{BASE}/experiments/checkpoints/diffpepdock_v1.pth", use_torch=True, map_location="cpu")
ck = {k.replace("module.", ""): v for k, v in blob["model"].items()}
msd = m.state_dict()
match = [k for k in ck if k in msd and ck[k].shape == msd[k].shape]
print(f"   checkpoint tensors {len(ck)}, matching this model's keys {len(match)} "
      f"({'OK' if len(match) > 0.9 * len(ck) else 'BROKEN -- keys would be dropped'})")
m.load_state_dict({k: ck[k] for k in match})

print("2) TRAINABLE PARAMETERS")
for mode in ("fixed", "flexible"):
    B.MODE = mode
    mm = B.BindopsFlowModel(conf.model, diffuser)
    tr = sum(p.numel() for p in mm.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in mm.parameters())
    ad = sum(p.numel() for p in mm.adapter.parameters())
    print(f"   {mode:9s} trainable {tr:,} / {tot:,}   (adapter alone = {ad:,})")
B.MODE = "fixed"

print("3) bindops_score IN THE BATCH")
ds = PlaceDataset(data_conf=conf.data, diffuser=diffuser, is_training=True, flow=True,
                  mask_pep_esm=False, seed=0)
# is_training=True returns the dict alone; the (dict, name) tuple is the eval path
feats = ds[0]
name = ds.csv.iloc[0]["pdb_name"]
bs = feats["bindops_score"]
import numpy as _np
lig = _np.asarray(feats["ligand_mask"]).astype(bool)
bs = _np.asarray(bs)
print(f"   target {name}: shape {tuple(bs.shape)}, res_mask len {tuple(feats['res_mask'].shape)}")
print(f"   receptor scores min {bs[~lig].min():.3f} max {bs[~lig].max():.3f} mean {bs[~lig].mean():.3f}")
print(f"   peptide slots all zero: {bool((bs[lig] == 0).all())}")

print("4) ZERO-INIT ADAPTER IS AN EXACT NO-OP")
def _b(v):
    t = v if torch.is_tensor(v) else torch.tensor(_np.asarray(v))
    return t.unsqueeze(0) if t.ndim > 0 else t.reshape(1)
batch = {k: _b(v) for k, v in feats.items()}
batch["t"] = batch["t"].float()
m.eval()
with torch.no_grad():
    out_cond = m(batch)["rigids"]
    inner = m.score_network
    saved = batch["bindops_score"]
    batch["bindops_score"] = torch.zeros_like(saved)
    out_zero = m(batch)["rigids"]
    batch["bindops_score"] = saved
    out_plain = inner(batch)["rigids"]
d_plain = (out_cond - out_plain).abs().max().item()
d_zero = (out_cond - out_zero).abs().max().item()
print(f"   max|conditioned - unconditioned backbone| = {d_plain:.3e}  ({'no-op OK' if d_plain < 1e-5 else 'DIFFERS'})")
print(f"   max|score vs zeroed score|                = {d_zero:.3e}  (also ~0 before training)")
print("   -> any post-training difference is learned, not a wiring artefact")
