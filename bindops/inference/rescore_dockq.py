"""Accurate DockQ for a make_poseset poseset.

* STANDARD DockQ v2 (no --capri_peptide: DockQ's own docs say the DockQ score "cannot be trusted"
  in that mode), mapping AB:AB (A = peptide, B = receptor in both files).
* Matched atoms: poses carry peptide N,CA,C,O,CB only, so the native peptide is reduced to the same
  set before scoring -- fnat/contacts are then computed on identical atom sets on both sides.
  Receptor: full atom in both (it is the fixed native receptor).
* No-contact / unscoreable poses = 0 (failed dock), never dropped.
Writes <poseset>/dockq_std.csv: name, pose, dockq, irmsd, lrmsd, fnat, conf_score (from labels.csv).
usage: rescore_dockq.py <poseset_dir> [nproc]
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os, re, sys, glob, subprocess
import numpy as np, pandas as pd
from multiprocessing import Pool

DQ = "/8TBDrive2/DiffPepBuilder/env/bin/DockQ"
PEP_ATOMS = {"N", "CA", "C", "O", "CB"}
D = sys.argv[1]
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 12
KEYS = ["DockQ", "iRMSD", "LRMSD", "fnat"]


def native_bb(tdir):
    out = f"{tdir}/native_bbcb.pdb"
    if not os.path.exists(out):
        keep = [l for l in open(f"{tdir}/native.pdb")
                if not (l.startswith(("ATOM", "HETATM")) and l[21] == "A" and l[12:16].strip() not in PEP_ATOMS)]
        open(out + ".tmp", "w").writelines(keep); os.replace(out + ".tmp", out)
    return out


def one(args):
    name, pose = args
    tdir = f"{D}/{name}"
    r = subprocess.run([DQ, f"{tdir}/pose_{pose}.pdb", native_bb(tdir), "--short", "--mapping", "AB:AB"],
                       capture_output=True, text=True)
    vals = {}
    for k in KEYS:
        m = re.search(rf"\b{k} ([0-9.]+)", r.stdout)
        vals[k] = float(m.group(1)) if m else np.nan
    return dict(name=name, pose=pose, dockq=vals["DockQ"], irmsd=vals["iRMSD"],
                lrmsd=vals["LRMSD"], fnat=vals["fnat"])


if __name__ == "__main__":
    lab = pd.read_csv(f"{D}/labels.csv")
    for n in lab.name.unique():
        native_bb(f"{D}/{n}")
    with Pool(NP) as P:
        rows = P.map(one, list(zip(lab.name, lab.pose)), chunksize=16)
    s = pd.DataFrame(rows)
    s["unscored"] = s.dockq.isna()
    s["dockq"] = s.dockq.fillna(0.0)
    s = s.merge(lab[["name", "pose", "conf_score"]], on=["name", "pose"], how="left")
    s.to_csv(f"{D}/dockq_std.csv", index=False)
    g = s.groupby("name")
    # conf_score is a PREDICTED ERROR, not a quality score: SpeedDocker ConfidenceHead regresses
    # the flip-aware peptide-CA LRMSD to native through a final Softplus, so LOWER IS BETTER and
    # argmin is the correct selection. Settled 2026-10-07 --
    # runs/blind/TIER1_SIGN_AUDIT_HEADLINE.md. Every number recorded before that date used argmax,
    # which was an ANTI-selection, so the superseded argmax line is printed too and every old
    # .score file can still be reconciled against this log. Do not remove it.
    conf = s.loc[g.conf_score.idxmin()].set_index("name").dockq
    conf_argmax = s.loc[g.conf_score.idxmax()].set_index("name").dockq
    print(f"{s.name.nunique()} targets, {len(s)} poses, unscored->0: {100*s.unscored.mean():.1f}%  ({D})")
    for lbl, v in [("oracle best-of-N", g.dockq.max()),
                   ("conf-selected top-1", conf),
                   ("mean pose", g.dockq.mean()),
                   ("[SUPERSEDED] argmax conf", conf_argmax)]:
        print(f"  {lbl:24s} median {v.median():.3f}  mean {v.mean():.3f}  acceptable {100*(v>=0.23).mean():.1f}%  "
              f"medium {100*(v>=0.49).mean():.1f}%  high {100*(v>=0.80).mean():.1f}%")
    print("  conf-selected top-1 = argmin(conf_score) (CORRECT: the head predicts LRMSD, lower is "
          "better). The [SUPERSEDED] line is argmax(conf_score), the inverted convention every "
          "number recorded before 2026-10-07 used -- kept only so old logs can be reconciled.")
