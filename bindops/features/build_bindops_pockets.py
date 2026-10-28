"""Step 1b: build BINDops docking inputs in DiffPepBuilder's processed-pkl format, with the pocket
chosen ONLY by leak-free BINDops v4 (pocket_tune.py rule, R chosen on train), never by the peptide.

Per complex:
  1. Re-parse the RCSB mmCIF (asym unit, or assembly1 where fullatom_v2 needed it) with Bio.PDB so
     atom NAMES are available for atom37; match residues to our receptor/peptide lists by CA.
  2. Write the native complex PDB (chain A = peptide, chain B = receptor) -> natives/, used as the
     DockQ reference at evaluation.
  3. Pocket = BINDops rule on the receptor; crop = full peptide + pocket residues.
  4. Features exactly as experiments/process_dataset.py:process_file (parsers.process_chain +
     du.parse_chain_feats), but centered on the BINDops pocket CENTER instead of the true peptide
     centroid. Extra keys: bindops_score (per crop residue; 0 for peptide positions),
     bindops_center.
  5. raw_seq_data (full chain seq + crop mask) for the ESM pass (experiments/embed_esm.py).
Must run with DiffPepBuilder's env python, via ~/bin/capped_run.sh.
Usage: build_bindops_pockets.py <R>"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import sys, os, gzip, glob, pickle, dataclasses, time, json
sys.path.insert(0, "/8TBDrive2/DiffPepBuilder")
import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from multiprocessing import get_context
from scipy.spatial import cKDTree
from data import utils as du
from data import parsers, residue_constants

BASE = "/8TBDrive1/jithesh/docking_model"
CIF = f"{BASE}/cif"
FREE = f"{BASE}/bindops_scores_free"
OUTD = f"{BASE}/bindops_dock_data"
NMAX = 64
STRICT, LOOSE, AMBIG = 0.05, 1.5, 3.0
R_POCKET = float(sys.argv[1]) if len(sys.argv) > 1 else 16.0


def pocket(ca, s, R):
    t = cKDTree(ca)
    nb = t.query_ball_point(ca, 10.0)
    sm = np.array([s[n].mean() for n in nb])
    seed = int(sm.argmax())
    near = nb[seed]; w = s[near]
    center = (ca[near] * w[:, None]).sum(0) / w.sum()
    d = np.linalg.norm(ca - center, axis=1)
    idx = np.where(d <= R)[0]
    return np.sort(idx[np.argsort(d[idx])][:NMAX]), center


def cif_residues(pdb, source):
    path = f"{CIF}/{pdb.lower()}{'-assembly1' if source == 'assembly1' else ''}.cif.gz"
    st = PDB.MMCIFParser(QUIET=True).get_structure(pdb, gzip.open(path, "rt"))
    models = list(st) if source == "assembly1" else [next(iter(st))]
    out = []
    for m in models:
        for ch in m:
            for r in ch:
                if "CA" in r and (r.id[0] == " " or r.resname in residue_constants.substitute_non_standard_restype):
                    out.append(r)
    return out


def match(ca_list, cres, tree, cca):
    d, j = tree.query(ca_list, k=2)
    got = []
    for (d1, d2), (j1, _) in zip(d, j):
        got.append(cres[j1] if (d1 <= STRICT or (d1 <= LOOSE and d2 > AMBIG)) else None)
    return got


def copy_res(r, new_id):
    """Fresh residue with plain (non-disordered) atoms: for alternate conformations take Bio.PDB's
    selected child (altloc A / highest occupancy). r.copy() keeps DisorderedAtom wrappers, which
    broke 29/93 complexes in the first chunk ('Atom' object has no attribute 'disordered_get_list')."""
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    nr = Residue((" ", new_id, " "), r.resname, r.segid)
    for a in r:
        if a.is_disordered():
            a = a.selected_child
        nr.add(Atom(a.get_name(), a.coord.copy(), a.bfactor, 1.0, " ", a.get_fullname(),
                    a.serial_number, a.element))
    return nr


def build_one(args):
    e, sc = args
    name = e["name"]
    cres = cif_residues(e["pdb"], e.get("cif_source", "asym"))
    cca = np.array([r["CA"].coord for r in cres]); tree = cKDTree(cca)
    rec_ca = np.stack([x["ca"] for x in e["receptor"]]); pep_ca = np.stack([x["ca"] for x in e["peptide"]])
    rec_m = match(rec_ca, cres, tree, cca); pep_m = match(pep_ca, cres, tree, cca)
    if any(m is None for m in pep_m):
        return None, "peptide_unmatched"
    keep = [i for i, m in enumerate(rec_m) if m is not None]  # receptor indices with atom names
    if len(keep) < 10:
        return None, "receptor_unmatched"
    # pocket over matched receptor residues only
    idx_local, center = pocket(rec_ca[keep].astype(float), sc["score"][keep].astype(float), R_POCKET)
    pocket_idx = [keep[i] for i in idx_local]

    s = Structure("x"); m = Model(0); s.add(m)
    ca_, cb_ = Chain("A"), Chain("B"); m.add(ca_); m.add(cb_)
    for k, r in enumerate(pep_m):
        ca_.add(copy_res(r, k + 1))
    rec_num = {}
    for k, i in enumerate(keep):
        cb_.add(copy_res(rec_m[i], k + 1)); rec_num[i] = k + 1
    nat = f"{OUTD}/natives/{name}.pdb"
    io = PDB.PDBIO(); io.set_structure(s); io.save(nat)

    # crop: peptide + pocket (same chain ids; residue numbering from the native so chain gaps persist)
    pocket_nums = set(rec_num[i] for i in pocket_idx)
    crop_chains = {}
    ca_crop = Chain("A"); [ca_crop.add(r.copy()) for r in ca_]
    cb_crop = Chain("B"); [cb_crop.add(r.copy()) for r in cb_ if r.id[1] in pocket_nums]
    feats, masks, res_count = [], {}, 0
    total = len(ca_crop) + len(cb_crop)
    for cid, ch in (("A", ca_crop), ("B", cb_crop)):
        prot = parsers.process_chain(ch, du.chain_str_to_int(cid))
        cd = du.parse_chain_feats(dataclasses.asdict(prot), center_pos=center.astype(np.float64))
        feats.append(cd)
        mk = np.zeros(total); mk[res_count:res_count + len(cd["aatype"])] = 1; masks[cid] = mk
        res_count += len(cd["aatype"])
    cf = du.concat_np_features(feats, False)
    cf["center_pos"] = center
    cf["modeled_idx"] = np.where(cf["aatype"] != 20)[0]
    cf["ligand_mask"] = masks["A"]
    bscore = np.zeros(total, dtype=np.float32)
    bscore[len(ca_crop):] = sc["score"][sorted(pocket_idx)]
    cf["bindops_score"] = bscore
    cf["bindops_center"] = center
    pkl = f"{OUTD}/processed/{name}.pkl"
    du.write_pkl(pkl, cf)

    seq_of = lambda ch: "".join(residue_constants.restype_3to1.get(
        residue_constants.substitute_non_standard_restype.get(r.resname, r.resname), "X") for r in ch)
    raw = {"A": {"seq": seq_of(ca_), "mask": [1] * len(ca_)},
           "B": {"seq": seq_of(cb_), "mask": [1 if r.id[1] in pocket_nums else 0 for r in cb_]}}
    true = np.where(sc["min_dist"] <= 8.0)[0]
    recall = float(np.isin(true, pocket_idx).mean()) if len(true) else float("nan")
    ctr_err = float(np.linalg.norm(center - pep_ca.mean(0)))
    meta = {"pdb_name": name, "processed_path": pkl, "native_pdb": nat, "num_chains": 2,
            "seq_len": total, "modeled_seq_len": int(len(cf["modeled_idx"])), "pep_len": len(ca_crop),
            "pocket_len": len(cb_crop), "pocket_recall": recall, "center_err": ctr_err,
            "gid": e.get("gid"), "heldout_idx": e.get("heldout_idx"), "pdb": e["pdb"]}
    return (meta, {name: raw}), None


def work(path):
    tag = os.path.basename(path).split("_")[0]
    out = f"{OUTD}/parts/{os.path.basename(path)}"
    if os.path.exists(out):
        return pickle.load(open(out, "rb"))["stats"]
    sc = {r["name"]: r for r in pickle.load(open(path, "rb"))}
    ents = [e for e in pickle.load(open(f"{BASE}/fullatom_chunks/{os.path.basename(path)}", "rb")) if e["name"] in sc]
    metas, raws, skips = [], {}, {}
    for e in ents:
        try:
            res, why = build_one((e, sc[e["name"]]))
        except Exception as ex:
            res, why = None, f"error:{type(ex).__name__}:{str(ex)[:60]}"
        if res is None:
            skips[why] = skips.get(why, 0) + 1
        else:
            metas.append({**res[0], "split": tag}); raws.update(res[1])
    stats = {"tag": tag, "n": len(ents), "ok": len(metas), "skips": skips}
    pickle.dump({"metas": metas, "raws": raws, "stats": stats}, open(out + ".tmp", "wb")); os.replace(out + ".tmp", out)
    return stats


if __name__ == "__main__":
    for d in ["natives", "processed", "parts"]:
        os.makedirs(f"{OUTD}/{d}", exist_ok=True)
    paths = sorted(glob.glob(f"{FREE}/*_[0-9][0-9][0-9][0-9].pkl"))
    print(f"R_POCKET={R_POCKET}, {len(paths)} chunks", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(40) as pool:
        stats = []
        for i, st in enumerate(pool.imap_unordered(work, paths)):
            stats.append(st)
            if (i + 1) % 10 == 0 or i + 1 == len(paths):
                print(f"  {i+1}/{len(paths)} chunks, {time.time()-t0:.0f}s", flush=True)
    metas, raws = [], {}
    for p in sorted(glob.glob(f"{OUTD}/parts/*.pkl")):
        d = pickle.load(open(p, "rb")); metas += d["metas"]; raws.update(d["raws"])
    df = pd.DataFrame(metas)
    skips = {}
    for st in stats:
        for k, v in st["skips"].items():
            skips[k] = skips.get(k, 0) + v
    print(f"built {len(df)} complexes; skips {skips}", flush=True)
    for tag in ["train", "heldout"]:
        d = df[df.split == tag]
        print(f"  {tag}: n={len(d)} pocket_len med {d.pocket_len.median():.0f} | recall med {d.pocket_recall.median():.3f} | "
              f"site-hit(recall>=0.5) {(d.pocket_recall >= 0.5).mean():.3f} | center_err med {d.center_err.median():.1f} A", flush=True)
    df.to_csv(f"{OUTD}/all_metadata.csv", index=False)
    du.write_pkl(f"{OUTD}/raw_seq_data.pkl", raws)
    print("ALL DONE")
