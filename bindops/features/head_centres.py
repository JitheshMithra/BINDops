"""Precompute start-site centres for the docking prior, for EVERY test target.

Three sets, so the DockQ arms differ in exactly one variable from the shipped
surface prior:
  head   -- BINDops's 1D interface head (BINDops == the unified 1D-interface +
            2D-contact model as of 2026-10-18; the old protein-protein patch stack is
            BINDops v4, superseded). The arm and the pkl keep the name 'head'.
  cninv  -- inverse coordination number, the free one-line scalar that already
            beats the whole docking model by +13.3pp on hit<10A
  surf   -- reference only: the centres the shipped surface prior draws from,
            recomputed here so the three can be compared off-GPU

Centres are greedy-NMS patch centres (same code the gate was judged with), then
pushed radially outward by the TRAIN-measured surface offset so they sit where a
peptide centroid actually sits rather than on the receptor shell. The offset is
estimated on train only -- no test structure touches it.

Keyed by pdb_name (== basename of processed_path), which is what the prior looks
up at generation time.
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os, sys, glob, pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bindops.features import cmap_gate as G
from bindops.model import bindops_unified as U

W = "/8TBDrive1/jithesh/docking_model"
V = f"{W}/verified_v2"
O = f"{W}/cmap_gate"
CB, CA = 3, 1
NC = int(os.environ.get("N_CENTRES", "8"))


def rec_from_pkl(d):
    """Receptor CB coords (CA for Gly) + receptor ESM, same convention as cmap_cache."""
    lm = d["ligand_mask"].astype(bool)
    ap, am = d["atom_positions"], d["atom_mask"]
    xyz = ap[:, CB, :].copy()
    miss = am[:, CB] < 0.5
    xyz[miss] = ap[miss, CA, :]
    ok = (am[:, CB] > 0.5) | (am[:, CA] > 0.5)
    rec = (~lm) & ok
    pep = lm & ok
    return xyz[rec], d["esm_embed"][rec], xyz[pep]


def struct_from_xyz(X):
    X = X.astype(np.float32)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    f = [((D < c).sum(1) - 1).astype(np.float32) for c in (8.0, 10.0, 12.0, 14.0)]
    cen = X.mean(0)
    rad = np.linalg.norm(X - cen, axis=1)
    f.append(rad / max(float(np.percentile(rad, 95)), 1e-6))
    F = np.stack(f, 1)
    mu, sd = F.mean(0, keepdims=True), F.std(0, keepdims=True) + 1e-6
    return ((F - mu) / sd).astype(np.float32), f[1]   # feats, coord_num@10A


def train_offset():
    off = []
    for r in G.load("train"):
        X, cen = r["rec_xyz"], r["rec_xyz"].mean(0)
        nat = r["pep_xyz"].mean(0)
        j = int(np.argmin(np.linalg.norm(X - nat, axis=1)))
        off.append(np.linalg.norm(nat - cen) - np.linalg.norm(X[j] - cen))
    return float(np.median(off))


def push_out(C, cen, delta):
    v = C - cen
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-6] = 1e-6
    return C + v / n * delta


def main():
    delta = train_offset()
    print(f"surface offset (TRAIN only): {delta:+.2f} A", flush=True)

    uni = U.Unified(drop=U.DROP)
    uni.load_state_dict(torch.load(f"{O}/unified.pth", map_location="cpu"))
    uni.eval()

    # HC_GLOB / HC_SUFFIX let this run on the blind set without changing the default
    # behaviour, which must stay byte-identical because centres_head.pkl underpins the
    # 2026-10-20 head-prior result.
    files = sorted(glob.glob(os.environ.get("HC_GLOB", f"{V}/proc_test/*.pkl")))
    sfx = os.environ.get("HC_SUFFIX", "")
    print("scoring %d pkls -> centres_*%s.pkl" % (len(files), sfx), flush=True)
    out = {k: {} for k in ("head", "cninv", "surf")}
    nat_d = {k: [] for k in ("head", "cninv", "surf")}
    bad = []
    for n, f in enumerate(files):
        name = os.path.basename(f)[:-4]
        try:
            d = pickle.load(open(f, "rb"))
            X, esm, P = rec_from_pkl(d)
        except Exception as e:
            bad.append((name, repr(e)[:60])); continue
        if len(X) < 10 or len(P) < 1:
            bad.append((name, f"R={len(X)} P={len(P)}")); continue
        SF, cn10 = struct_from_xyz(X)
        cen = X.mean(0)
        with torch.no_grad():
            rec = torch.from_numpy(np.concatenate([esm.astype(np.float32), SF], 1))
            pep = torch.from_numpy(d["esm_embed"][d["ligand_mask"].astype(bool)].astype(np.float32))
            iface, _ = uni(pep, rec)
        sc = {"head": iface.numpy(), "cninv": -cn10.astype(float)}
        # the shipped surface prior's own candidate set, scored so NMS can rank it
        nbr = cn10.astype(float)
        sc["surf"] = -nbr + 1e-9 * np.arange(len(nbr))   # == cninv up to tie-break; kept separate
        for k in ("head", "cninv"):
            C = G.patch_centres(sc[k], X, n=NC)
            C = push_out(C, cen, delta)
            out[k][name] = C.astype(np.float32)
            nat_d[k].append(float(np.linalg.norm(C - P.mean(0), axis=1).min()))
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(files)}", flush=True)

    for k in ("head", "cninv"):
        with open(f"{O}/centres_{k}{sfx}.pkl", "wb") as fh:
            pickle.dump(out[k], fh, protocol=4)
        d_ = np.asarray(nat_d[k])
        print(f"{k}: {len(out[k])} targets  best-of-{NC} median {np.median(d_):.2f} A  "
              f"hit<10A {(d_<10).mean()*100:.1f}%  hit<6A {(d_<6).mean()*100:.1f}%")
    if bad:
        print(f"skipped {len(bad)}: {bad[:5]}")


if __name__ == "__main__":
    main()
