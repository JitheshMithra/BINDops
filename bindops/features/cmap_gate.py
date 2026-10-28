"""GATE: is the peptide-receptor contact map predictable from the features we
already have? If yes, the docking model never asked the question. If no, we are
at the same ceiling as every 1D site predictor and the framing needs rethinking.

Head: project ESM (sequence-only, no structural leak) to d, form pair features
[a_i, b_j, a_i*b_j], small MLP -> contact logit. This is exactly the information
the current one-shot edge MLP already has.

Decisive metric is NOT AUROC. It is site localisation: take the top-k receptor
residues by predicted contact marginal, and ask how far their centroid is from
the native peptide centroid -- the same question the docking model fails at.
Baselines computed identically: random-k, and inverse coord_num (the known-best
1D peptide-site predictor, which ties the whole 185-dim v4 stack).
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os, sys, pickle, time, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(8)
O = "/8TBDrive1/jithesh/docking_model/cmap_gate"
DEV = "cpu"
USE_CN = os.environ.get("USE_COORD_NUM", "0") == "1"
USE_STRUCT = os.environ.get("USE_STRUCT", "0") == "1"
NEXTRA = (1 if USE_CN else 0) + (5 if USE_STRUCT else 0)

def struct_feats(r):
    """Per-receptor-residue geometry: coordination at 4 radii + relative radius.
    This is exactly the receptor structure the docking model already sees."""
    X = r["rec_xyz"].astype(np.float32)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    f = [((D < c).sum(1) - 1).astype(np.float32) for c in (8.0, 10.0, 12.0, 14.0)]
    cen = X.mean(0)
    rad = np.linalg.norm(X - cen, axis=1)
    f.append(rad / max(float(np.percentile(rad, 95)), 1e-6))
    F = np.stack(f, 1)
    mu, sd = F.mean(0, keepdims=True), F.std(0, keepdims=True) + 1e-6
    return ((F - mu) / sd).astype(np.float32)

def auroc(score, label):
    score = np.asarray(score, float).ravel()
    label = np.asarray(label).ravel().astype(bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[label].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

class Head(nn.Module):
    def __init__(self, d=64, extra=0):
        super().__init__()
        self.pp = nn.Linear(1280, d)
        self.pr = nn.Linear(1280, d)
        self.mlp = nn.Sequential(
            nn.Linear(3 * d + extra, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))
    def forward(self, pe, re_, cn=None):
        a = self.pp(pe)                      # [P,d]
        b = self.pr(re_)                     # [R,d]
        P, R, d = a.shape[0], b.shape[0], a.shape[1]
        A = a[:, None, :].expand(P, R, d)
        B = b[None, :, :].expand(P, R, d)
        f = [A, B, A * B]
        if cn is not None:
            f.append(cn[None, :, :].expand(P, R, cn.shape[-1]))
        return self.mlp(torch.cat(f, -1)).squeeze(-1)   # [P,R]

def load(split):
    with open(f"{O}/{split}.pkl", "rb") as f:
        return pickle.load(f)

def prep(r):
    pe = torch.from_numpy(r["pep_esm"].astype(np.float32))
    re_ = torch.from_numpy(r["rec_esm"].astype(np.float32))
    y = torch.from_numpy(r["cmap"].astype(np.float32))
    extra = []
    if USE_CN:
        c = r["coord_num"]
        extra.append(((c - c.mean()) / (c.std() + 1e-6)).astype(np.float32)[:, None])
    if USE_STRUCT:
        extra.append(struct_feats(r))
    cn = torch.from_numpy(np.concatenate(extra, 1)) if extra else None
    return pe, re_, y, cn

def patch_centres(score_rec, X, n=8, rad=12.0):
    """Greedy non-max suppression -> up to n spatially distinct patch centres.
    Each centre is the score-weighted centroid of residues within `rad` of the
    local peak, so a diffuse prediction cannot collapse onto the protein core."""
    s = np.asarray(score_rec, float).copy()
    out, live = [], np.ones(len(s), bool)
    for _ in range(n):
        if not live.any():
            break
        i = int(np.argmax(np.where(live, s, -np.inf)))
        near = (np.linalg.norm(X - X[i], axis=1) < rad) & live
        w = s[near] - s[near].min() + 1e-6
        out.append((X[near] * w[:, None]).sum(0) / w.sum())
        live &= ~near
    return np.asarray(out)

def site_dists(score_rec, r, n=8):
    C = patch_centres(score_rec, r["rec_xyz"], n=n)
    nat = r["pep_xyz"].mean(0)
    return np.linalg.norm(C - nat, axis=1)

def evaluate(model, data, n=8, seed=0):
    rng = np.random.default_rng(seed)
    model.eval()
    keys = ("head", "rand", "cn", "cninv")
    out = {"auroc": [], "topL": [], "enrich": []}
    for k in keys:
        out["top1_" + k] = []
        out["best_" + k] = []
    with torch.no_grad():
        for r in data:
            pe, re_, y, cn = prep(r)
            logit = model(pe, re_, cn).numpy()
            yy = r["cmap"]
            out["auroc"].append(auroc(logit, yy))
            L = yy.shape[0]
            top = np.argsort(-logit.ravel())[:L]
            prec = float(yy.ravel()[top].mean())
            out["topL"].append(prec)
            out["enrich"].append(prec / max(float(yy.mean()), 1e-9))
            scores = dict(head=logit.max(0),
                          rand=rng.normal(size=r["rec_xyz"].shape[0]),
                          cn=r["coord_num"].astype(float),
                          cninv=-r["coord_num"].astype(float))
            for k, sc in scores.items():
                d = site_dists(sc, r, n=n)
                out["top1_" + k].append(float(d[0]))
                out["best_" + k].append(float(d.min()))
    return out

def boot_hit(d, thr, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    h = (np.asarray(d) < thr).astype(float)
    s = [h[rng.integers(0, len(h), len(h))].mean() for _ in range(B)]
    return h.mean() * 100, np.percentile(s, 2.5) * 100, np.percentile(s, 97.5) * 100

def report(out, tag, n):
    a = np.asarray(out["auroc"], float)
    a = a[~np.isnan(a)]
    print("\n--- %s ---" % tag)
    print("  contact-map AUROC   mean %.3f  median %.3f  frac>0.8 %.1f%%"
          % (a.mean(), np.median(a), np.mean(a > 0.8) * 100))
    print("  top-L precision     mean %.3f  enrichment over density %.2fx"
          % (np.mean(out["topL"]), np.mean(out["enrich"])))
    print("  site localisation, patch centres by greedy NMS (best-of-%d):" % n)
    print("  %-20s%10s%10s%20s%20s"
          % ("method", "top1 med", "best med", "best <10A", "best <6A"))
    for key, name in [("head", "CONTACT HEAD"), ("cninv", "inverse coord_num"),
                      ("cn", "coord_num"), ("rand", "random patches")]:
        t1 = np.asarray(out["top1_" + key])
        bb = np.asarray(out["best_" + key])
        h10, l10, u10 = boot_hit(bb, 10.0)
        h6, l6, u6 = boot_hit(bb, 6.0)
        print("  %-20s%9.1fA%9.1fA%9.1f%% [%4.1f,%4.1f]%9.1f%% [%4.1f,%4.1f]"
              % (name, np.median(t1), np.median(bb), h10, l10, u10, h6, l6, u6))
    hb = (np.asarray(out["best_head"]) < 10).astype(float)
    rb = (np.asarray(out["best_rand"]) < 10).astype(float)
    rng = np.random.default_rng(0)
    m = len(hb)
    d = [(hb[i] - rb[i]).mean() for i in (rng.integers(0, m, m) for _ in range(2000))]
    print("  PAIRED head-minus-random hit<10A: %+.1fpp CI[%+.1f,%+.1f]"
          % ((hb - rb).mean() * 100, np.percentile(d, 2.5) * 100,
             np.percentile(d, 97.5) * 100))

def main():
    tr, va, te = load("train"), load("val"), load("test")
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  "
          f"USE_COORD_NUM={USE_CN}")
    dens = np.mean([r["cmap"].mean() for r in tr])
    model = Head(extra=NEXTRA).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pw = torch.tensor(float((1 - dens) / max(dens, 1e-6)))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    print(f"contact density {dens*100:.2f}%  pos_weight {pw.item():.1f}")

    best, best_state, patience = -1, None, 0
    EPOCHS = int(os.environ.get("EPOCHS", "12"))
    rng = np.random.default_rng(0)
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(len(tr))
        t0, tot, nb = time.time(), 0.0, 0
        opt.zero_grad()
        for n, i in enumerate(order):
            pe, re_, y, cn = prep(tr[i])
            loss = lossf(model(pe, re_, cn), y)
            (loss / 8).backward()
            tot += loss.item(); nb += 1
            if (n + 1) % 8 == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        vo = evaluate(model, va)
        va_auc = float(np.nanmean(vo["auroc"]))
        print(f"  epoch {ep+1:2d}  train_loss {tot/max(nb,1):.4f}  "
              f"val_AUROC {va_auc:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        if va_auc > best:
            best, patience = va_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 3:
                print("  early stop"); break
    model.load_state_dict(best_state)
    print(f"\n=== BEST val AUROC {best:.3f}; evaluating on TEST (n={len(te)}) ===")
    tag = "ESM+STRUCT" if USE_STRUCT else ("ESM+coord_num" if USE_CN else "ESM only")
    report(evaluate(model, te, n=8), "TEST n=%d  [%s]" % (len(te), tag), 8)
    torch.save(best_state, "%s/head_cn%d_st%d.pth" % (O, int(USE_CN), int(USE_STRUCT)))

if __name__ == "__main__":
    main()
