"""Build a compact cache for the inter-chain contact-map gate.

Per target: peptide ESM, receptor ESM, CB coords, coord_num, native contact map.
ESM is SEQUENCE-ONLY so the primary head has no structural leak; coord_num (the
known-best 1D peptide-site predictor) is kept separate as the baseline to beat.
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os, sys, glob, pickle
import numpy as np

CB, CA = 3, 1

def cb_coords(atom_pos, atom_mask):
    """CB where present, CA for glycine."""
    xyz = atom_pos[:, CB, :].copy()
    miss = atom_mask[:, CB] < 0.5
    xyz[miss] = atom_pos[miss, CA, :]
    ok = (atom_mask[:, CB] > 0.5) | (atom_mask[:, CA] > 0.5)
    return xyz, ok

def build(split_dir, out_path, limit=None, seed=0, max_rec=400):
    files = sorted(glob.glob(os.path.join(split_dir, "*.pkl")))
    if limit and len(files) > limit:
        files = list(np.random.default_rng(seed).choice(files, limit, replace=False))
        files.sort()
    recs, skipped = [], {}
    for n, f in enumerate(files):
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            skipped["unreadable"] = skipped.get("unreadable", 0) + 1
            continue
        need = ("ligand_mask", "esm_embed", "atom_positions", "atom_mask")
        if not all(k in d for k in need):
            skipped["missing_keys"] = skipped.get("missing_keys", 0) + 1
            continue
        lm = d["ligand_mask"].astype(bool)
        xyz, ok = cb_coords(d["atom_positions"], d["atom_mask"])
        pep = lm & ok
        rec = (~lm) & ok
        npep, nrec = int(pep.sum()), int(rec.sum())
        if npep < 3 or nrec < 30 or nrec > max_rec:
            skipped["size"] = skipped.get("size", 0) + 1
            continue
        P, R = xyz[pep], xyz[rec]
        D = np.sqrt(((P[:, None, :] - R[None, :, :]) ** 2).sum(-1))
        C = (D < 8.0).astype(np.uint8)
        if C.sum() == 0:
            skipped["no_contact"] = skipped.get("no_contact", 0) + 1
            continue
        # coord_num of each receptor residue: receptor CBs within 10 A (self excluded)
        DR = np.sqrt(((R[:, None, :] - R[None, :, :]) ** 2).sum(-1))
        cn = ((DR < 10.0).sum(1) - 1).astype(np.float32)
        recs.append(dict(
            name=os.path.basename(f)[:-4],
            pep_esm=d["esm_embed"][pep].astype(np.float16),
            rec_esm=d["esm_embed"][rec].astype(np.float16),
            rec_xyz=R.astype(np.float32),
            pep_xyz=P.astype(np.float32),
            coord_num=cn,
            cmap=C,
        ))
        if (n + 1) % 500 == 0:
            print(f"  {n+1}/{len(files)}  kept={len(recs)}", flush=True)
    with open(out_path, "wb") as fh:
        pickle.dump(recs, fh, protocol=4)
    cr = np.mean([r["cmap"].mean() for r in recs]) * 100
    print(f"{split_dir} -> {out_path}: kept {len(recs)}/{len(files)}  "
          f"skipped={skipped}  mean contact density {cr:.2f}%")
    return recs

if __name__ == "__main__":
    V = "/8TBDrive1/jithesh/docking_model/verified_v2"
    O = "/8TBDrive1/jithesh/docking_model/cmap_gate"
    os.makedirs(O, exist_ok=True)
    build(f"{V}/proc_val_tune", f"{O}/val.pkl")
    build(f"{V}/proc_test", f"{O}/test.pkl")
