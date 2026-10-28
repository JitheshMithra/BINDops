"""Make the pocket-finding model DIFFERENTIABLE so the docking loss can fine-tune it.

BINDops v4 is a HistGradientBoostingClassifier -- trees, no gradients, so it can never be trained
by a docking objective. This distills it into a small MLP over the identical 185 features, in the
identical order eval_heldout_bindops.py builds them:
    [ScanNet, ISIS, TriGlue_const, PSSM window 9x20 = 180, PeSTo, coord_num]
The student starts as a faithful copy of v4 (so the fine-tuned arm begins where the frozen arm is,
not from scratch), and is then free to move under the docking loss in the training arm.

Fidelity is reported, not assumed: Pearson r and MAE against v4's probabilities on held-out
complexes, plus both models' AUROC against TRUE peptide contacts. A student that does not match
v4 closely is a bad starting point and the run should stop.
usage: distill_bindops.py [n_train_complexes]
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os, sys, glob, pickle
import numpy as np, pandas as pd, joblib, torch
from torch import nn

W = "/8TBDrive1/jithesh/docking_model"
D = f"{W}/bindops_dock_data_full"
NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OUT = f"{W}/models/bindops_student.pt"
torch.manual_seed(0)


def window(p, w=9):
    """[n,20] -> [n,180] zero-padded centred window, same as pssm_common.window_features."""
    n, d = p.shape
    h = w // 2
    pad = np.zeros((n + 2 * h, d), np.float32)
    pad[h:h + n] = p
    return np.stack([pad[i:i + w].reshape(-1) for i in range(n)])


def feats(name, prof, comp, raw):
    """185-dim features for the RECEPTOR residues of one complex, in v4's column order."""
    npep = len(raw[name]["A"]["seq"])
    P, C = prof[name][npep:].astype(np.float32), comp[name][npep:].astype(np.float32)
    if len(P) == 0 or not C[:, 0].any():
        return None
    return np.concatenate([C[:, 0:1], C[:, 1:2], C[:, 2:3], window(P), C[:, 3:4], C[:, 4:5]], 1)


class Student(nn.Module):
    def __init__(self, d=185, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 128), nn.GELU(),
                                 nn.Linear(128, 1))

    def forward(self, x):            # logit; probability = sigmoid
        return self.net(x).squeeze(-1)


def main():
    v4 = joblib.load(f"{W}/models/bindops_v4.joblib")
    prof = pickle.load(open(f"{D}/pssm_profiles.pkl", "rb"))
    comp = pickle.load(open(f"{D}/bindops_components.pkl", "rb"))
    raw = pickle.load(open(f"{D}/raw_seq_data.pkl", "rb"))
    tr = pd.read_csv(f"{D}/metadata_train.csv").pdb_name.tolist()[:NTR]
    ho = pd.read_csv(f"{D}/metadata_heldout.csv").pdb_name.tolist()

    def build(names):
        X, Y = [], []
        for n in names:
            f = feats(n, prof, comp, raw)
            if f is None:
                continue
            X.append(f); Y.append(v4.predict_proba(f)[:, 1])
        return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)

    Xtr, Ytr = build(tr)
    Xho, Yho = build(ho)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    print(f"train residues {len(Xtr)}, held-out residues {len(Xho)}")

    dev = "cpu"
    m = Student().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    xb = torch.tensor((Xtr - mu) / sd); yb = torch.tensor(Ytr)
    xh = torch.tensor((Xho - mu) / sd); yh = torch.tensor(Yho)
    n, BS = len(xb), 4096
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    for ep in range(30):
        m.train(); perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            loss = nn.functional.binary_cross_entropy_with_logits(m(xb[idx]), yb[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
        if ep % 10 == 9 or ep == 0:
            m.eval()
            with torch.no_grad():
                ph = torch.sigmoid(m(xh)).numpy()
            r = np.corrcoef(ph, Yho)[0, 1]
            print(f"  epoch {ep+1:2d} train BCE {tot/n:.4f} | held-out vs v4: r={r:.4f} MAE={np.abs(ph-Yho).mean():.4f}")

    m.eval()
    with torch.no_grad():
        ph = torch.sigmoid(m(xh)).numpy()
    r = float(np.corrcoef(ph, Yho)[0, 1]); mae = float(np.abs(ph - Yho).mean())

    # agreement on the thing that is actually used: the per-complex argmax-weighted pocket centre
    from sklearn.metrics import roc_auc_score
    rows = {}
    for p in sorted(glob.glob(f"{W}/bindops_scores/*.pkl")):
        for rr in pickle.load(open(p, "rb")):
            rows[rr["name"]] = rr
    a_v4, a_st, k = [], [], 0
    for nm in ho:
        f = feats(nm, prof, comp, raw)
        rr = rows.get(nm)
        if f is None or rr is None or "min_dist_to_partner" not in rr:
            continue
        d = np.asarray(rr["min_dist_to_partner"], np.float32)
        if len(d) != len(f) or len(np.unique(d <= 8.0)) < 2:
            continue
        with torch.no_grad():
            s = torch.sigmoid(m(torch.tensor((f - mu) / sd, dtype=torch.float32))).numpy()
        a_v4.append(roc_auc_score(d <= 8.0, v4.predict_proba(f)[:, 1]))
        a_st.append(roc_auc_score(d <= 8.0, s)); k += 1
    print(f"\nFIDELITY (held-out): r={r:.4f}  MAE={mae:.4f}")
    print(f"AUROC vs TRUE contacts over {k} complexes: v4 {np.mean(a_v4):.4f}  student {np.mean(a_st):.4f} "
          f"(delta {np.mean(a_st)-np.mean(a_v4):+.4f})")
    ok = r >= 0.95 and mae <= 0.05
    print("VERDICT:", "GOOD starting point" if ok else "POOR -- do not fine-tune from this")
    os.makedirs(f"{W}/models", exist_ok=True)
    torch.save({"state": m.state_dict(), "mu": mu, "sd": sd, "r": r, "mae": mae,
                "auroc_v4": float(np.mean(a_v4)), "auroc_student": float(np.mean(a_st))}, OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
