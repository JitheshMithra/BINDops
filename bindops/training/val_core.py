"""Shared held-out evaluation for the flow arms.

One implementation, used by BOTH validate_ckpt.py (out-of-band, scores a saved checkpoint) and
train_flow_v5.py (in-trainer, every CKPT_EVERY steps). They must not drift: an in-trainer curve
that disagreed with the out-of-band curve would be impossible to interpret.

DETERMINISM FIX (2026-10-15): validate_ckpt.py seeded eval_item, which fixes the interpolation
time and the blind-prior draw -- but NOT the complex re-centring jitter. _process_csv_row calls
du.parse_chain_feats(center_pos_noise=True, std=5.0) by default, and parse_chain_feats draws that
offset from the GLOBAL numpy RNG (data/utils.py:471), which eval_item's seed never touches. So
every checkpoint saw a different 5 A translation per example and part of the val curve was noise.
Two guards here: the jitter is switched OFF (it is training augmentation and has no place in an
eval), and the global RNG is seeded per item anyway, so anything else reaching for np.random is
pinned too. Numbers from this module are therefore NOT comparable to the pre-fix curves.
"""
import os
import copy
import numpy as np
import torch
import tree


def build_val_dataset(conf, diffuser, val_csv):
    """A second dataset over the val CSV. is_training=False so eval_item's seed drives every draw."""
    from speeddock.dataset import PlaceDataset
    # Augmentation off for evaluation. Set before the dataset is built: _process_csv_row reads
    # these at call time, and the loader may prefetch.
    os.environ["BINDOPS_CENTER_NOISE"] = "0"
    c = copy.deepcopy(conf)
    c.data.val_csv_path = val_csv
    c.data.train_csv_path = val_csv
    c.data.num_eval_samples = None
    c.data.center_pos_noise = False
    return PlaceDataset(data_conf=c.data, diffuser=diffuser, is_training=False,
                        flow=True, mask_pep_esm=False, seed=0)


def evaluate(model, ds, reps=4, device="cuda", recycle=0, max_examples=None):
    """Mean flow loss / FAPE over (example, repeat) pairs, each with a fixed seed.

    recycle: number of no-grad recycle passes before the scored pass, matching how the arm was
    trained. Fixed, not sampled -- an eval must not roll dice the checkpoints disagree about.
    """
    from speeddock.losses import flow_loss

    was_training = model.training
    model.eval()
    n = len(ds.csv) if max_examples is None else min(max_examples, len(ds.csv))
    tot, fapes, bbs, per_tau, nonfinite = [], [], [], {}, 0
    try:
        with torch.no_grad():
            for idx in range(n):
                for r in range(reps):
                    seed = 100003 * idx + r
                    np.random.seed(seed % (2 ** 31 - 1))   # pin the global RNG too, see module docstring
                    feats, _name = ds.eval_item(idx, seed=seed)
                    batch = tree.map_structure(
                        lambda x: (x if torch.is_tensor(x) else torch.tensor(x)).unsqueeze(0).to(device),
                        feats)
                    batch["t"] = torch.as_tensor([float(feats["t"])], device=device)
                    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                        for _ in range(recycle):
                            batch["sc_ca_t"] = model(batch)["rigids"][..., 4:].float()
                        out = model(batch)
                        loss, parts = flow_loss(out, batch)
                    if not torch.isfinite(loss):
                        nonfinite += 1
                        continue
                    tot.append(loss.item()); fapes.append(parts["fape"]); bbs.append(parts["bb"])
                    per_tau.setdefault(round(float(feats["t"]), 1), []).append(parts["fape"])
    finally:
        if was_training:
            model.train()
    if not tot:
        return dict(n_examples=n, reps=reps, n_evals=0, nonfinite=nonfinite,
                    val_loss=float("nan"), val_fape=float("nan"))
    return dict(n_examples=n, reps=reps, n_evals=len(tot), nonfinite=nonfinite, recycle=recycle,
                val_loss=float(np.mean(tot)), val_loss_sd=float(np.std(tot)),
                val_loss_se=float(np.std(tot) / np.sqrt(len(tot))),
                val_fape=float(np.mean(fapes)), val_bb=float(np.mean(bbs)),
                fape_by_t={str(k): float(np.mean(v)) for k, v in sorted(per_tau.items())})
