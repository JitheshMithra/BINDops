"""A real, non-zero site score for the adapter, replacing the literal zeros.

Background: every training pkl carries bindops_score = exactly 0.0, and the arms ran
BINDOPS_MODE=control with the adapter frozen at its zero init, so the "BINDops-conditioned"
model contained no BINDops at all.

Connecting the published v4 field would mean regenerating ScanNet + ISIS + TriGlue + PeSTo +
coord_num for 2,021 receptors -- days of external-tool infrastructure -- and the 2026-10-13
benchmark says it would not pay: on real peptide sites (n=51) every protein-protein interface
predictor collapses to 0.54-0.61 AUROC, v4 included (0.615), and INVERSE COORDINATION NUMBER
ALONE ties the whole 185-dim v4 stack (p=0.23) while beating it on top-10% enrichment
(2.06x vs 1.95x). So coord_num is the measured content of that channel, and it costs one pass
over the receptor CA coordinates.

Caveat recorded deliberately: the shipped surface prior already samples by the coord_num
percentile, so this channel is largely REDUNDANT with the prior, and on the 2026-10-13 evidence
it should be expected to do little on its own. It is wired because the adapter path itself needs
to carry something real, and because the same path then carries the MSA features
(build_profiles.py), which are NOT redundant with burial.

No leak: only receptor CA coordinates are read. The peptide's coordinates are never touched, and
peptide rows are left at zero -- the same convention bindops_score used.

usage: build_site_scores.py [--out O]
"""
import os, csv, json, pickle, argparse
import numpy as np

V = "/8TBDrive1/jithesh/docking_model/verified_v2"
RADIUS = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=V + "/msa/site_scores.pkl")
    ap.add_argument("--splits", action="append", default=None,
                    help='"name=path" (repeatable); default: the four verified_v2 splits. Used to emit features for a set the builder does not know about, e.g. blindcases_v1.')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    out, report = {}, {}
    CA = 1   # atom37 index of CA
    SPLITS = (("train", V + "/data/metadata_train.csv"),
              ("val_real_A", V + "/data/metadata_val_real_A.csv"),
              ("val_real_B", V + "/data/metadata_val_real_B.csv"),
              ("test", V + "/proc_test/metadata_test.csv"))
    if a.splits:
        SPLITS = tuple(tuple(x.split("=", 1)) for x in a.splits)
        for nm, pth in SPLITS:
            if not os.path.exists(pth):
                raise SystemExit("--splits %s: no such csv %s" % (nm, pth))
    for split, path in SPLITS:
        rows = list(csv.DictReader(open(path)))
        ok = bad = 0
        for r in rows:
            try:
                with open(r["processed_path"], "rb") as fh:
                    d = pickle.load(fh)
            except Exception:
                bad += 1; continue
            pos = np.asarray(d["atom_positions"])
            mask = np.asarray(d["atom_mask"])
            lm = np.asarray(d["ligand_mask"]).astype(bool)
            ca = pos[:, CA, :]
            has = mask[:, CA] > 0
            rec = (~lm) & has
            if rec.sum() < 5:
                bad += 1; continue
            # coordination number among RECEPTOR residues only
            rc = ca[rec]
            dist = np.linalg.norm(rc[:, None, :] - rc[None, :, :], axis=-1)
            cn = ((dist < RADIUS).sum(1) - 1).astype(np.float32)
            # percentile within this receptor, then inverted so EXPOSED = high, matching the
            # direction that measured best on peptide sites (inverse coord_num)
            order = cn.argsort().argsort().astype(np.float32)
            pct = order / max(len(cn) - 1, 1)
            score = np.zeros(len(lm), dtype=np.float32)
            idx = np.where(rec)[0]
            score[idx] = 1.0 - pct
            out[r["pdb_name"]] = score
            ok += 1
        report[split] = dict(rows=len(rows), built=ok, failed=bad)
        print("%-11s rows %5d  built %5d  failed %d" % (split, len(rows), ok, bad))

    if not out:
        # A relative processed_path plus a wrong CWD makes every row unreadable, and the
        # per-row handler counts that as `bad` rather than raising -- so this is the first
        # place the failure becomes visible. Say what it is instead of dying in numpy with
        # 'need at least one array to concatenate', which obscured exactly this for two
        # runs on 2026-10-20.
        raise SystemExit('build_site_scores: covered ZERO cases across every split. Usual cause: processed_path in the metadata CSV is RELATIVE and the working directory is wrong -- run from /8TBDrive1/jithesh/docking_model, and check the first processed_path resolves from the cwd before rerunning.')
    s = np.concatenate([v for v in list(out.values())[:200]])
    print("score stats over first 200 cases: min %.3f max %.3f mean %.3f  (zeros = peptide rows)"
          % (s.min(), s.max(), s.mean()))
    with open(a.out, "wb") as f:
        pickle.dump(out, f, protocol=4)
    with open(os.path.dirname(a.out) + "/site_scores_report.json", "w") as f:
        json.dump(dict(report=report, radius=RADIUS, feature="1 - coord_num percentile (receptor only)"), f, indent=1)
    print("WROTE", a.out, len(out), "cases")


main()
