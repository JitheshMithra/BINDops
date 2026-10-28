"""Held-out validation for the flow arms -- the thing train_flow_v4.py never had.

`train_flow_v4.py` passes `data.val_csv_path` into the hydra config and then never reads it:
PdbDataset only opens that path when `is_training=False`, and the trainer builds the training
dataset only. So metadata_val.csv's 30 rows were dead, checkpoints were saved on step count
alone, and there was no signal for overfitting, underfitting or a broken objective.

This runs out-of-band instead of inside the trainer, deliberately: the trainer is shared with
the stage1 arms and a live 11 h run, and editing it would invalidate their reproducibility.

Determinism matters more than speed here -- a val curve is only comparable across checkpoints if
every checkpoint sees the SAME interpolation times and the SAME prior draws. Each (example,
repeat) gets a fixed seed, so the tau schedule and the surface-prior draw are identical at every
step. Batch-of-one avoids length_batching entirely (the collate that crashed the 2026-10-10 run).

usage: BLIND_PRIOR=surface validate_ckpt.py <ckpt> <val_csv> [--reps R] [--gpu G] [--json OUT]
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os, sys, json, argparse
import numpy as np
import torch

os.environ.setdefault("BINDOPS_MODE", "control")
os.environ.setdefault("DPB_BASE", "/8TBDrive2/DiffPepBuilder")
os.environ.setdefault("SPEEDDOCK_REPO", "/8TBDrive2/SpeedDocker")
os.environ.setdefault("DPB_NUM_TORSIONS", "3")

from bindops.training import train_bindops_flow as B          # applies model + loader + blind-prior patches
import tree
from hydra import initialize_config_dir, compose
from data import se3_diffuser
from speeddock.dataset import PlaceDataset
from speeddock.losses import flow_loss
from openfold.utils.rigid_utils import Rigid

BASE = os.environ["DPB_BASE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("val_csv")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--pair-stack", action="store_true",
                    help="checkpoint carries a pair stack (sets PAIR_STACK=1 before model build)")
    a = ap.parse_args()
    if a.pair_stack:
        os.environ["PAIR_STACK"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with initialize_config_dir(version_base=None, config_dir=f"{BASE}/config"):
        conf = compose(config_name="base", overrides=[
            f"data.train_csv_path={a.val_csv}", f"data.val_csv_path={a.val_csv}",
            "data.num_eval_samples=null", "data.center_pos_noise=False",
            "experiment.use_ddp=False"])
    diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)

    # Must match the class the checkpoint was TRAINED with. Building BindopsFlowModel for a
    # pair-stack checkpoint would send every pair.* tensor into the strict=False no-man's-land
    # and validate a model with the stack silently absent -- a quietly wrong val curve.
    if a.pair_stack:
        from bindops.training.train_pair_flow import PairFlowModel
        model = PairFlowModel(conf.model, diffuser).to(device)
    else:
        model = B.BindopsFlowModel(conf.model, diffuser).to(device)
    blob = torch.load(a.ckpt, map_location=device)
    model.load_state_dict(blob["model"], strict=False)
    if a.pair_stack:
        npair = len([k for k in blob["model"] if k.startswith("pair.")])
        assert npair > 0, "--pair-stack given but checkpoint holds no pair.* tensors"
        print(f"[validate] pair.* tensors in checkpoint: {npair}", flush=True)
    model.eval()

    # flow=True so _assemble draws an interpolation time; is_training=False so eval_item's
    # explicit seed controls every draw. mask_pep_esm=False matches v4 training exactly.
    ds = PlaceDataset(data_conf=conf.data, diffuser=diffuser, is_training=False,
                      flow=True, mask_pep_esm=False, seed=0)
    n = len(ds.csv)
    tot, fapes, bbs, per_tau = [], [], [], {}
    with torch.no_grad():
        for idx in range(n):
            for r in range(a.reps):
                feats, name = ds.eval_item(idx, seed=100003 * idx + r)   # fixed -> comparable
                batch = tree.map_structure(
                    lambda x: (x if torch.is_tensor(x) else torch.tensor(x)).unsqueeze(0).to(device),
                    feats)
                batch["t"] = torch.as_tensor([float(feats["t"])], device=device)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    out = model(batch)
                    loss, parts = flow_loss(out, batch)
                if not torch.isfinite(loss):
                    continue
                tot.append(loss.item()); fapes.append(parts["fape"]); bbs.append(parts["bb"])
                tb = round(float(feats["t"]), 1)
                per_tau.setdefault(tb, []).append(parts["fape"])

    res = dict(ckpt=a.ckpt, step=int(blob.get("step", -1)), n_examples=n, reps=a.reps,
               n_evals=len(tot),
               val_loss=float(np.mean(tot)), val_loss_sd=float(np.std(tot)),
               val_fape=float(np.mean(fapes)), val_bb=float(np.mean(bbs)),
               fape_by_t={str(k): float(np.mean(v)) for k, v in sorted(per_tau.items())})
    print(f"step {res['step']:>6}  val_loss {res['val_loss']:9.3f}  "
          f"fape {res['val_fape']:.4f}  bb {res['val_bb']:9.3f}  (n={res['n_evals']})")
    print("  fape by t: " + "  ".join(f"t={k} {v:.3f}" for k, v in res["fape_by_t"].items()))
    if a.json:
        with open(a.json, "a") as fh:
            fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
