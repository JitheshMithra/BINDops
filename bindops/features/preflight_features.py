"""Refuse to start a feature-conditioned arm unless the features actually cover the data.

This exists because of a specific, repeated failure on this project: a feature file loads, logs a
healthy-looking count, and then covers almost none of the training set. bindops_score was literal
zeros for every case; pssm_profiles.pkl covers 0.3% of verified_v2 train by gid. Both ran for
hours and produced numbers nobody could interpret.

Checks, per split, for whichever of MSA_PROFILE_PKL / SITE_SCORE_PKL are set:
  - coverage >= --min-cov (default 0.99)
  - every covered array has the right length and is FINITE
  - the values are not all zero, and not all identical (a constant feature is a dead channel)
  - for MSA: peptide rows are zero and receptor rows are not, matching the documented convention
Exits non-zero with a summary if anything fails, so a runner script can gate on it.

usage: MSA_PROFILE_PKL=... SITE_SCORE_PKL=... preflight_features.py [--min-cov F] [--splits a,b]
"""
import os, sys, csv, pickle, argparse
import numpy as np

V = "/8TBDrive1/jithesh/docking_model/verified_v2"
B = "/8TBDrive1/jithesh/docking_model/blindcases_v1"
SPLITS = {"train": V + "/data/metadata_train.csv",
          "val_real_A": V + "/data/metadata_val_real_A.csv",
          "val_real_B": V + "/data/metadata_val_real_B.csv",
          "test": V + "/proc_test/metadata_test.csv",
          # the 317 post-2025 cases; same gate, because a gid missing from the pkl is a silent
          # all-zero feature row, not an error
          "blind": B + "/bindops_proc/metadata_test.csv",
          "blind_tierA": B + "/bindops_proc/metadata_tierA.csv",
          "blind_tierB": B + "/bindops_proc/metadata_tierB.csv",
          "blind_tierC": B + "/bindops_proc/metadata_tierC.csv"}

ap = argparse.ArgumentParser()
ap.add_argument("--min-cov", type=float, default=0.99)
ap.add_argument("--splits", default="train,val_real_A,test")
a = ap.parse_args()

msa_p = os.environ.get("MSA_PROFILE_PKL", "")
site_p = os.environ.get("SITE_SCORE_PKL", "")
if not msa_p and not site_p:
    print("neither MSA_PROFILE_PKL nor SITE_SCORE_PKL is set -- nothing to check")
    sys.exit(0)

fails = []
msa = pickle.load(open(msa_p, "rb")) if msa_p else None
site = pickle.load(open(site_p, "rb")) if site_p else None
if msa is not None:
    print("MSA profiles loaded:", len(msa), "from", msa_p)
if site is not None:
    print("site scores loaded:", len(site), "from", site_p)
print()

for sp in a.splits.split(","):
    rows = list(csv.DictReader(open(SPLITS[sp])))
    gids = [r["pdb_name"] for r in rows]
    for nm, tbl, is_msa in (("MSA", msa, True), ("site", site, False)):
        if tbl is None:
            continue
        cov = [g for g in gids if g in tbl]
        frac = len(cov) / max(len(gids), 1)
        ok = frac >= a.min_cov
        print("%-11s %-5s coverage %5d/%-5d = %.3f %s"
              % (sp, nm, len(cov), len(gids), frac, "OK" if ok else "<-- BELOW THRESHOLD"))
        if not ok:
            fails.append(f"{sp}/{nm} coverage {frac:.3f} < {a.min_cov}")
        if not cov:
            continue
        sample = cov[:300]
        nonfinite = zeroall = constant = pepnonzero = recallzero = 0
        for g in sample:
            v = np.asarray(tbl[g], dtype=np.float32)
            if not np.isfinite(v).all():
                nonfinite += 1
            if np.abs(v).max() == 0:
                zeroall += 1
            elif v.ndim == 1 and np.unique(v).size == 1:
                constant += 1
            if is_msa:
                # peptide rows must be zero, receptor rows must not be
                nz = np.abs(v).sum(1) > 0
                if nz.all():
                    pepnonzero += 1      # no zero rows at all -> peptide convention broken
                if not nz.any():
                    recallzero += 1
        print("       of %d sampled: non-finite %d, all-zero %d, constant %d%s"
              % (len(sample), nonfinite, zeroall, constant,
                 ", no-zero-rows %d, all-zero-rows %d" % (pepnonzero, recallzero) if is_msa else ""))
        if nonfinite:
            fails.append(f"{sp}/{nm}: {nonfinite} non-finite arrays")
        if zeroall:
            fails.append(f"{sp}/{nm}: {zeroall} all-zero arrays (dead channel)")
        if is_msa and recallzero:
            fails.append(f"{sp}/{nm}: {recallzero} arrays with no nonzero row")

print()
if fails:
    print("PREFLIGHT FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PREFLIGHT PASSED -- features cover the data and carry real values")
