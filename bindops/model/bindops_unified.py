"""ONE MODEL: BINDops' 1D interface task and the 2D contact map, sharing a trunk.

Why it is built this way instead of bolting onto v4 directly:
  * `bindops_score` in every verified_v2 pkl is ALL ZEROS (400/400, train and
    test), so there is no v4 output to consume.
  * v4's 185-dim feature stack is keyed to the OLD corpus and covers 0.3% of
    our train targets, so the real trunk is blocked behind a PSSM recompute.

So the merge is architectural. The receptor trunk mirrors the shape of the
distilled BINDops Student (185->256->128->1) with our feature set in place of
the 185-dim stack, and `head1d` IS that Student's final layer. A second head
reads the same trunk pairwise against the peptide to predict contacts.

The 1D interface label is free: receptor residue j is interface iff any peptide
residue contacts it, i.e. cmap[:, j].any(). It carries ~10x more positives per
target than the sparse 2D map, so it is the denser, better-conditioned signal
and is there to regularise the 2D task, which overfit by epoch 4 on its own.

Decisive test: the single-task contact head TIED inverse coord_num
(+3.8pp CI[-1.6,+9.4], n.s.). If the merged model does not clear that free
one-line baseline with a CI excluding zero, the extra machinery is not earning
its place and must be reported as such.
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os, sys, time, pickle, json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bindops.features import cmap_gate as G

torch.set_num_threads(8)
torch.manual_seed(0)
O = "/8TBDrive1/jithesh/docking_model/cmap_gate"
W1D = float(os.environ.get("W1D", "1.0"))
W2D = float(os.environ.get("W2D", "1.0"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
DROP = float(os.environ.get("DROP", "0.2"))
WD = float(os.environ.get("WD", "1e-4"))
TAG = os.environ.get("TAG", "unified")


class Unified(nn.Module):
    """Shared receptor trunk; BINDops-style 1D head + 2D contact head."""

    def __init__(self, d_in_rec=1285, d_pep=1280, h=256, d=128, drop=0.2):
        super().__init__()
        # trunk mirrors the distilled BINDops Student: in -> 256 -> 128
        self.trunk = nn.Sequential(
            nn.Linear(d_in_rec, h), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h, d), nn.GELU())
        # the Student's final layer: this IS the BINDops interface head
        self.head1d = nn.Linear(d, 1)
        self.pep = nn.Sequential(nn.Linear(d_pep, d), nn.GELU())
        self.head2d = nn.Sequential(
            nn.Linear(3 * d, 128), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, pep_esm, rec_feat):
        hr = self.trunk(rec_feat)                      # [R,d]
        hp = self.pep(pep_esm)                         # [P,d]
        iface = self.head1d(hr).squeeze(-1)            # [R]
        P, R, d = hp.shape[0], hr.shape[0], hr.shape[1]
        A = hp[:, None, :].expand(P, R, d)
        B = hr[None, :, :].expand(P, R, d)
        contact = self.head2d(torch.cat([A, B, A * B], -1)).squeeze(-1)   # [P,R]
        return iface, contact


def feats(r):
    pep = torch.from_numpy(r["pep_esm"].astype(np.float32))
    rec_esm = r["rec_esm"].astype(np.float32)
    rec = torch.from_numpy(np.concatenate([rec_esm, G.struct_feats(r)], 1))
    y2 = torch.from_numpy(r["cmap"].astype(np.float32))
    y1 = torch.from_numpy((r["cmap"].sum(0) > 0).astype(np.float32))
    return pep, rec, y1, y2


def evaluate(model, data, n=8, seed=0):
    rng = np.random.default_rng(seed)
    model.eval()
    out = {k: [] for k in ("auc2d", "auc1d", "topL", "enrich",
                           "d_head", "d_cninv", "d_rand", "auc_cninv")}
    with torch.no_grad():
        for r in data:
            pep, rec, y1, y2 = feats(r)
            iface, contact = model(pep, rec)
            iface = iface.numpy()
            contact = contact.numpy()
            yy = r["cmap"]
            lab1 = (yy.sum(0) > 0)
            out["auc2d"].append(G.auroc(contact, yy))
            out["auc1d"].append(G.auroc(iface, lab1))
            out["auc_cninv"].append(G.auroc(-r["coord_num"].astype(float), lab1))
            L = yy.shape[0]
            top = np.argsort(-contact.ravel())[:L]
            prec = float(yy.ravel()[top].mean())
            out["topL"].append(prec)
            out["enrich"].append(prec / max(float(yy.mean()), 1e-9))
            # localisation uses the 1D head -- that is the BINDops-shaped output
            for key, sc in (("d_head", iface),
                            ("d_cninv", -r["coord_num"].astype(float)),
                            ("d_rand", rng.normal(size=len(iface)))):
                out[key].append(float(G.site_dists(sc, r, n=n).min()))
    return out


def paired(a, b, thr=10.0, B=2000, seed=0):
    ha = (np.asarray(a) < thr).astype(float)
    hb = (np.asarray(b) < thr).astype(float)
    rng = np.random.default_rng(seed)
    n = len(ha)
    s = [(ha[i] - hb[i]).mean() for i in (rng.integers(0, n, n) for _ in range(B))]
    return (ha - hb).mean() * 100, np.percentile(s, 2.5) * 100, np.percentile(s, 97.5) * 100


def main():
    tr, va, te = G.load("train"), G.load("val"), G.load("test")
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}   "
          f"W1D={W1D} W2D={W2D} drop={DROP} wd={WD}")

    d2 = float(np.mean([r["cmap"].mean() for r in tr]))
    d1 = float(np.mean([(r["cmap"].sum(0) > 0).mean() for r in tr]))
    print(f"label density: 2D contacts {d2*100:.2f}%   1D interface {d1*100:.2f}%  "
          f"({d1/max(d2,1e-9):.1f}x denser)")

    model = Unified(drop=DROP)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WD)
    l1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - d1) / max(d1, 1e-9)))
    l2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - d2) / max(d2, 1e-9)))

    best, best_state, patience = -1.0, None, 0
    rng = np.random.default_rng(0)
    for ep in range(EPOCHS):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        opt.zero_grad()
        for n, i in enumerate(rng.permutation(len(tr))):
            pep, rec, y1, y2 = feats(tr[i])
            iface, contact = model(pep, rec)
            loss = W1D * l1(iface, y1) + W2D * l2(contact, y2)
            (loss / 8).backward()
            tot += float(loss); nb += 1
            if (n + 1) % 8 == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        vo = evaluate(model, va)
        sel = float(np.nanmean(vo["auc2d"]))
        print(f"  epoch {ep+1:2d}  loss {tot/max(nb,1):.4f}  "
              f"val auc2d {sel:.3f}  val auc1d {np.nanmean(vo['auc1d']):.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if sel > best:
            best, patience = sel, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 4:
                print("  early stop"); break

    model.load_state_dict(best_state)
    torch.save(best_state, f"{O}/{TAG}.pth")
    o = evaluate(model, te)
    a2 = np.asarray(o["auc2d"], float); a2 = a2[~np.isnan(a2)]
    a1 = np.asarray(o["auc1d"], float); a1 = a1[~np.isnan(a1)]
    ac = np.asarray(o["auc_cninv"], float); ac = ac[~np.isnan(ac)]
    print(f"\n=== UNIFIED MODEL on TEST (n={len(te)}) ===")
    print(f"  2D contact AUROC      mean {a2.mean():.3f}  median {np.median(a2):.3f}  "
          f"frac>0.8 {np.mean(a2>0.8)*100:.1f}%")
    print(f"  1D interface AUROC    mean {a1.mean():.3f}   "
          f"[inverse coord_num on the same label: {ac.mean():.3f}]")
    print(f"  top-L precision       {np.mean(o['topL']):.3f}  "
          f"enrichment {np.mean(o['enrich']):.2f}x")
    print(f"\n  {'method':<22}{'best-of-8':>11}{'hit<10A':>10}{'hit<6A':>10}")
    for k, nm in (("d_head", "UNIFIED (1D head)"), ("d_cninv", "inverse coord_num"),
                  ("d_rand", "random patches")):
        d = np.asarray(o[k])
        print(f"  {nm:<22}{np.median(d):>10.2f}A{(d<10).mean()*100:>9.1f}%"
              f"{(d<6).mean()*100:>9.1f}%")
    print("\n  THE TEST THAT MATTERS (single-task head was +3.8pp CI[-1.6,+9.4] n.s.):")
    for a, b, lab in ((o["d_head"], o["d_cninv"], "unified - inverse coord_num"),
                      (o["d_head"], o["d_rand"], "unified - random")):
        m, lo, hi = paired(a, b)
        print(f"    {lab:<30} {m:+6.1f}pp CI[{lo:+6.1f},{hi:+6.1f}]  "
              f"{'SIGNIFICANT' if (lo>0 or hi<0) else 'n.s.'}")
    json.dump({k: list(map(float, v)) for k, v in o.items()},
              open(f"{O}/{TAG}_test.json", "w"))


if __name__ == "__main__":
    main()
