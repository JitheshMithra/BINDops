"""Turn the UniRef90 MSAs into per-residue evolutionary features for the docking model.

This is the "no MSA" hole being filled. The model's only sequence signal was ESM, a single-sequence
language model: it approximates family statistics but never sees THIS receptor's actual alignment.
Per residue we compute, from the alignment columns:
    20  log-odds   log2(f_obs / f_background), the PSSM quantity
     1  entropy    -sum f log f  -- conservation, which is what marks functional surface patches
     1  gap frac   how much of the family is missing here (loops, termini)
     1  log1p(N)   alignment depth, so the model can discount a shallow profile
= 23 numbers per receptor residue. Peptide rows are ZERO, the same convention bindops_score uses,
so the trainer applies modeled_idx identically.

Two alignment safeguards, because a profile silently offset by one residue is worse than no profile:
  - the query is row 0 of its own MSA, and its ungapped sequence must equal the query FASTA
  - the per-case array is only written if the receptor sequence rebuilt from the pkl's OWN aatype
    equals the sequence that profile was computed for
Anything that fails is counted as uncovered and reported, never quietly zero-filled -- the
0.3%-coverage pssm_profiles.pkl is exactly what that failure mode looks like in production.

usage: build_profiles.py [--msa-dir D] [--out O]
"""
import os, sys, csv, json, pickle, argparse, hashlib, collections
import numpy as np

W = "/8TBDrive1/jithesh/docking_model"
V = W + "/verified_v2"
DB = "/8TBDrive2/msa_uniref90"
RESTYPES = "ARNDCQEGHILKMFPSTWYV"
IDX = {c: i for i, c in enumerate(RESTYPES)}
# Robinson & Robinson background amino-acid frequencies, in RESTYPES order
BG = np.array([0.0787, 0.0514, 0.0419, 0.0536, 0.0193, 0.0423, 0.0632, 0.0739, 0.0228, 0.0514,
               0.0906, 0.0586, 0.0221, 0.0393, 0.0520, 0.0721, 0.0584, 0.0129, 0.0322, 0.0637])
BG = BG / BG.sum()
NFEAT = 23


def read_fasta_stream(path):
    name, cur = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line[0] == ">":
                if name is not None:
                    yield name, "".join(cur)
                name, cur = line[1:], []
            else:
                cur.append(line)
    if name is not None:
        yield name, "".join(cur)


def profile_from_msa(path, query_seq, max_seqs=5000):
    """[L,23] float32 for the query's L residues, or None if the MSA does not match the query."""
    it = read_fasta_stream(path)
    try:
        _qn, qaln = next(it)
    except StopIteration:
        return None
    keep = np.array([i for i, c in enumerate(qaln) if c != "-" and c != "."], dtype=np.int64)
    if "".join(qaln[i] for i in keep).upper() != query_seq.upper():
        return None
    L = len(keep)
    counts = np.zeros((L, 20), dtype=np.float64)
    gaps = np.zeros(L, dtype=np.float64)
    n = 0
    for _nm, aln in it:
        if len(aln) != len(qaln):
            continue
        n += 1
        if n > max_seqs:
            break
        row = np.frombuffer(aln.upper().encode("ascii", "replace"), dtype=np.uint8)[keep]
        for a, i in IDX.items():
            counts[:, i] += (row == ord(a))
        gaps += (row == ord("-")) | (row == ord("."))
    obs = counts.sum(1)
    # one pseudo-observation spread over the background, so an all-gap column scores 0 not -inf
    f = (counts + BG[None, :]) / (obs[:, None] + 1.0)
    logodds = np.clip(np.log2(f / BG[None, :]), -8.0, 8.0)
    ent = -(f * np.log(np.maximum(f, 1e-12))).sum(1)
    gapfrac = gaps / max(n, 1)
    depth = np.full(L, np.log1p(n))
    out = np.concatenate([logodds, ent[:, None], gapfrac[:, None], depth[:, None]], axis=1)
    return out.astype(np.float32), n


def main():
    ap = argparse.ArgumentParser()
    # --db is repeatable: each mmseqs query DB dir carries its own queries.fasta,
    # queries.lookup and msa_flat. unpackdb names files by numeric DB key, so two builds
    # cannot share a directory -- they are unioned here by sequence md5 instead.
    ap.add_argument("--db", action="append", default=None,
                    help="mmseqs query DB dir (repeatable); default: the original build")
    ap.add_argument("--msa-dir", default=None,
                    help="override msa_flat for a SINGLE --db (back-compat)")
    ap.add_argument("--out", default=V + "/msa/msa_profiles.pkl")
    ap.add_argument("--splits", action="append", default=None,
                    help='"name=path" (repeatable); default: the four verified_v2 splits. Used to emit features for a set the builder does not know about, e.g. blindcases_v1.')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    dbs = a.db or [DB]

    byseq, depths, bad = {}, [], collections.Counter()
    for dbdir in dbs:
        qseq = {nm: sq for nm, sq in read_fasta_stream(dbdir + "/queries.fasta")}
        key2name = {}
        lk = dbdir + "/queries.lookup"
        if os.path.exists(lk):
            for line in open(lk):
                pp = line.split()
                if len(pp) >= 2:
                    key2name[pp[0]] = pp[1]
        mdir = a.msa_dir if (a.msa_dir and len(dbs) == 1) else dbdir + "/msa_flat"
        files = [f for f in os.listdir(mdir)] if os.path.isdir(mdir) else []
        print("db %s: queries %d  lookup %d  msa files %d"
              % (dbdir, len(qseq), len(key2name), len(files)), flush=True)
        if not files:
            raise RuntimeError("no MSA files under %s -- refusing to build a partial profile "
                               "set silently" % mdir)
        for i, fn in enumerate(files):
            key = fn.split(".")[0]
            nm = key2name.get(key, key)
            if nm not in qseq:
                bad["msa file with unknown query key"] += 1
                continue
            r = profile_from_msa(os.path.join(mdir, fn), qseq[nm])
            if r is None:
                bad["query row does not match query fasta"] += 1
                continue
            prof, nd = r
            byseq[nm] = prof.astype(np.float16)
            depths.append(nd)
            if (i + 1) % 250 == 0:
                print("  parsed %d/%d" % (i + 1, len(files)), flush=True)
    print("profiles built:", len(byseq), dict(bad))
    if depths:
        d = np.array(depths)
        print("alignment depth: median %d  mean %.0f  min %d  max %d  (<=1 seq: %d)"
              % (np.median(d), d.mean(), d.min(), d.max(), (d <= 1).sum()))

    # ---- map onto each case's residue rows, verified against the pkl's own aatype
    out, report = {}, {}
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
        hit = miss = mismatch = 0
        for r in rows:
            gid = r["pdb_name"]
            try:
                with open(r["processed_path"], "rb") as fh:
                    d = pickle.load(fh)
            except Exception:
                miss += 1; continue
            aa = np.asarray(d["aatype"])
            lm = np.asarray(d["ligand_mask"]).astype(bool)
            rec_seq = "".join(RESTYPES[i] if 0 <= i < 20 else "X" for i in aa[~lm])
            h = hashlib.md5(rec_seq.encode()).hexdigest()
            prof = byseq.get(h)
            if prof is None:
                miss += 1; continue
            if prof.shape[0] != (~lm).sum():
                mismatch += 1; continue
            full = np.zeros((len(aa), NFEAT), dtype=np.float16)
            full[~lm] = prof
            out[gid] = full
            hit += 1
        report[split] = dict(rows=len(rows), covered=hit, uncovered=miss, length_mismatch=mismatch,
                             coverage=round(100.0 * hit / max(len(rows), 1), 2))
        print("%-11s rows %5d  covered %5d (%.1f%%)  uncovered %5d  len-mismatch %d"
              % (split, len(rows), hit, 100.0 * hit / max(len(rows), 1), miss, mismatch))

    with open(a.out, "wb") as f:
        pickle.dump(out, f, protocol=4)
    with open(os.path.dirname(a.out) + "/coverage.json", "w") as f:
        json.dump(dict(report=report, n_profiles=len(byseq), nfeat=NFEAT,
                       features="20 log-odds, entropy, gap fraction, log1p(depth)"), f, indent=1)
    print("WROTE", a.out, len(out), "cases")


main()
