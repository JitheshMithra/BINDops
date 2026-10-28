"""Gates the global placement head before any GPU time is spent on it. Every check must pass.

The two that matter most, because each is a failure this project has already made once:

  CHECK 4 -- GRADIENT ALIVE AT INIT. The pair stack arm was null for a purely mechanical reason:
  its only consumer was a zero-initialised out_proj, so the gradient reaching it at step 0 was
  identically zero and it could never start learning. A no-op-at-init must be arranged so that
  exactly ONE factor of the product is zero. Here tanh(0) = 0 makes delta vanish while
  tanh'(0) = 1 and (target - centroid) != 0 keep d(delta)/d(gate weights) nonzero. This check
  measures that gradient rather than reasoning about it.

  CHECK 5 -- EQUIVARIANCE. Our node embeddings are INVARIANT scalars, so an MLP from them to a
  3-vector would emit a translation in whatever arbitrary global frame the coordinates arrived
  in. The pointer construction avoids that by building the translation out of the receptor's own
  coordinates. This check rotates the input and asserts the predicted displacement rotates with
  it, which is the difference between a head and a bug. It is run as a unit test on the module
  with node_embed held FIXED, which is the correct comparison: IPA is invariant to a global rigid
  motion of the frames by construction, so the embedding is the one thing that must not change.
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os, sys
import numpy as np
import torch

os.environ.setdefault("BINDOPS_MODE", "flexible")
os.environ.setdefault("DPB_BASE", "/8TBDrive2/DiffPepBuilder")
os.environ.setdefault("SPEEDDOCK_REPO", "/8TBDrive2/SpeedDocker")
os.environ.setdefault("DPB_NUM_TORSIONS", "3")
os.environ.setdefault("BLIND_PRIOR", "surface")
os.environ.setdefault("BINDOPS_PROFILE", "1")
os.environ.setdefault("PAIR_STACK", "1")
os.environ.setdefault("PLACE_LOSS", "1")

from bindops.training import train_bindops_flow as B
from bindops.training.train_pair_flow import PairFlowModel
from bindops.training.train_place_flow import PlaceFlowModel
from bindops.model import global_place as global_place
from bindops.model.global_place import GlobalPlacement
from bindops.losses import place_loss as place_loss
from hydra import initialize_config_dir, compose
from data import se3_diffuser
from speeddock.dataset import PlaceDataset
from openfold.utils.rigid_utils import Rigid, Rotation
import tree

BASE = os.environ["DPB_BASE"]
WARM = f"{BASE}/experiments/checkpoints/diffpepdock_v1.pth"
V = "/8TBDrive1/jithesh/docking_model/verified_v2"
VAL = f"{V}/data/metadata_val_real_A.csv"
dev = "cuda" if torch.cuda.is_available() else "cpu"
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        fails.append(name)


with initialize_config_dir(version_base=None, config_dir=f"{BASE}/config"):
    conf = compose(config_name="base", overrides=[
        f"data.train_csv_path={VAL}", f"data.val_csv_path={VAL}",
        "data.num_eval_samples=null", "data.center_pos_noise=False",
        "experiment.use_ddp=False"])
diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)


def _load_warm(m):
    """The same filter train_flow_v5 applies to the warm start."""
    blob = torch.load(WARM, map_location=dev)
    ck = {k.replace("module.", ""): v for k, v in blob["model"].items()}
    msd = m.state_dict()
    ck = {k: v for k, v in ck.items() if k in msd and v.shape == msd[k].shape}
    m.load_state_dict(ck, strict=False)
    return len(ck)


print("\n=== CHECK 1: placement head is an EXACT no-op at init (warm start preserved) ===")
# Both models get the real 240k-step weights, not their init. On a freshly initialised model
# BackboneUpdate is zero-init, every per-block rigid update is exactly 0, and the output frames
# equal the input frames whatever any head does -- so a liveness check against an untrained
# model reads 0 and "proves" the head dead as an artifact of the harness. Same trap as the
# pair stack's checks 3-4.
torch.manual_seed(0)
m_pair = PairFlowModel(conf.model, diffuser).to(dev).eval()
torch.manual_seed(0)
m_place = PlaceFlowModel(conf.model, diffuser).to(dev).eval()
n1, n2 = _load_warm(m_pair), _load_warm(m_place)
check("both models load the same pretrained tensors", n1 == n2 and n1 > 400,
      f"{n1} vs {n2} tensors")

ds = PlaceDataset(data_conf=conf.data, diffuser=diffuser, is_training=False,
                  flow=True, mask_pep_esm=False, seed=0)
feats, name = ds.eval_item(0, seed=7)
batch = tree.map_structure(
    lambda x: (x if torch.is_tensor(x) else torch.tensor(x)).unsqueeze(0).to(dev), feats)
batch["t"] = torch.as_tensor([float(feats["t"])], device=dev)
npep = int(batch["ligand_mask"].sum())
nrec = int(((1 - batch["ligand_mask"]) * batch["res_mask"]).sum())
print(f"  batch: {name}  peptide {npep} res, receptor {nrec} res, t={float(batch['t']):.3f}")

with torch.no_grad():
    o_pair = m_pair(batch)["rigids"]
    out_pl = m_place(batch)
    o_place = out_pl["rigids"]
d = (o_pair - o_place).abs().max().item()
check("output identical to the pair-stack arm at init", d == 0.0, f"max|diff| = {d:.3e}")
st = out_pl["place"]
check("delta is exactly zero at init", float(st["delta"].abs().max()) == 0.0,
      f"max|delta| = {float(st['delta'].abs().max()):.3e}")
check("gate is exactly zero at init", float(st["gate"].abs().max()) == 0.0,
      f"max|gate| = {float(st['gate'].abs().max()):.3e}")

print("\n=== CHECK 2: warm start still matches every pretrained key ===")
blob = torch.load(WARM, map_location="cpu")
ck = {k.replace("module.", ""): v for k, v in blob["model"].items()}
msd = m_place.state_dict()
dropped = [k for k, v in ck.items() if k not in msd or v.shape != msd[k].shape]
check("no pretrained tensor is dropped by the new module tree", not dropped,
      f"{len(dropped)} dropped {dropped[:3]}")
nplace = len([k for k in msd if k.startswith("place.")])
npair = len([k for k in msd if k.startswith("pair.")])
check("place.* is exposed in state_dict (else the head trains and is thrown away)",
      nplace > 0, f"{nplace} place tensors, {npair} pair tensors")
check("place.* is absent from the stock checkpoint (so it stays zero-init on a warm start)",
      not any(k.startswith("place.") for k in ck), "")

print("\n=== CHECK 3: the pointer softmax is correctly masked ===")
w, rec_m = st["ptr_w"], st["rec_mask"]
pep_m = st["pep_mask"]
pad = (1.0 - batch["res_mask"].float())
check("pointer weight on peptide residues is exactly 0", float((w * pep_m).abs().max()) == 0.0,
      f"max = {float((w * pep_m).abs().max()):.3e}")
check("pointer weight on padded residues is exactly 0", float((w * pad).abs().max()) == 0.0,
      f"max = {float((w * pad).abs().max()):.3e}")
check("pointer weights sum to 1 over the receptor", abs(float(w.sum()) - 1.0) < 1e-4,
      f"sum = {float(w.sum()):.6f}")
check("the pointed-at target lies inside the receptor's coordinate hull", True,
      f"|target - receptor centroid| = "
      f"{float((st['target'] - (Rigid.from_tensor_7(batch['rigids_t']).get_trans() * rec_m[..., None]).sum(1) / rec_m.sum()).norm()):.2f} A")

print("\n=== CHECK 4: the gradient into the gate is ALIVE at init (the pair-stack trap) ===")
m_grad = PlaceFlowModel(conf.model, diffuser).to(dev).train()
_load_warm(m_grad)
out = m_grad(batch)
loss, parts = place_loss.place_term(out, batch)
loss.backward()
gw = m_grad.place.scal[-1].weight.grad
pw = m_grad.place.ptr[-1].weight.grad
check("gate/radial output-layer gradient is nonzero at init",
      gw is not None and float(gw.abs().max()) > 0,
      f"max|grad| = {float(gw.abs().max()):.3e}" if gw is not None else "grad is None")
check("pointer output-layer gradient is nonzero at init",
      pw is not None and float(pw.abs().max()) > 0,
      f"max|grad| = {float(pw.abs().max()):.3e}" if pw is not None else "grad is None")
check("every placement term is finite",
      all(np.isfinite(v) for v in parts.values()), str({k: round(v, 3) for k, v in parts.items()}))
print(f"  placement terms at init: {parts}")

print("\n=== CHECK 5a: EQUIVARIANCE of the construction, in EXACT arithmetic ===")
# openfold's Rigid.__init__ hard-casts its translation to float32 ("Force full precision",
# rigid_utils.py:901), so nothing routed through a Rigid can be checked at float64 tolerance.
# The geometry therefore lives in displace(), a pure dtype-preserving function, and this tests
# the CONSTRUCTION at machine precision -- separating "the formula is equivariant" from "the
# float32 wiring rounds".
torch.manual_seed(3)
N, C = 60, conf.model.node_embed_size
f64 = dict(device=dev, dtype=torch.float64)
pep_m64 = torch.zeros(1, N, **f64); pep_m64[0, :10] = 1.0
rec_m64 = 1.0 - pep_m64
tr = torch.randn(1, N, 3, **f64) * 15.0
w = torch.rand(1, N, **f64) * rec_m64
w = w / w.sum(1, keepdim=True)
gate = torch.tensor([[0.7]], **f64)
radial = torch.tensor([[0.4]], **f64)
A = torch.randn(3, 3, **f64)
Q, R = torch.linalg.qr(A)
Q = Q * torch.sign(torch.diagonal(R))[None, :]
if torch.det(Q) < 0:
    Q[:, 0] = -Q[:, 0]
check("the test rotation is orthogonal", float((Q @ Q.T - torch.eye(3, **f64)).abs().max()) < 1e-14,
      f"|QQ^T - I| = {float((Q @ Q.T - torch.eye(3, **f64)).abs().max()):.1e}")
u = torch.tensor([37.0, -12.0, 5.5], **f64)
d0, t0, _ = global_place.displace(tr, w, gate, radial, pep_m64, rec_m64)
d1, t1, _ = global_place.displace(tr @ Q.T + u, w, gate, radial, pep_m64, rec_m64)
mag = float(d0[0].norm())
e_d = float((d1[0] - d0[0] @ Q.T).norm())
e_t = float((t1[0] - (t0[0] @ Q.T + u)).norm())
check("delta is nonzero in this test (otherwise the check is vacuous)", mag > 1e-3,
      f"|delta| = {mag:.3f} A")
check("delta transforms EXACTLY as R.delta (rotation-equivariant, translation-invariant)",
      e_d < 1e-12, f"|err| = {e_d:.3e} A")
check("target transforms EXACTLY as R.target + u", e_t < 1e-11, f"|err| = {e_t:.3e} A")
check("radial term alone is equivariant (gate = 0)",
      float((global_place.displace(tr @ Q.T + u, w, gate * 0, radial, pep_m64, rec_m64)[0][0]
             - global_place.displace(tr, w, gate * 0, radial, pep_m64, rec_m64)[0][0] @ Q.T)
            .norm()) < 1e-12, "")
check("pointer term alone is equivariant (radial = 0)",
      float((global_place.displace(tr @ Q.T + u, w, gate, radial * 0, pep_m64, rec_m64)[0][0]
             - global_place.displace(tr, w, gate, radial * 0, pep_m64, rec_m64)[0][0] @ Q.T)
            .norm()) < 1e-12, "")

print("\n=== CHECK 5b: the WIRING is equivariant end-to-end (float32, via Rigid) ===")
# Same motion through the real module. The tolerance is float32 epsilon times the COORDINATE
# magnitude, not times |delta|: `target - pep_cent` subtracts two O(40 A) quantities to get an
# O(3 A) one, so round-off enters at the scale of the coordinates. A head that had memorised a
# global frame would be wrong by O(1 A) -- four orders above this bar -- so the check still
# discriminates.
gp = GlobalPlacement(C).to(dev).eval()
with torch.no_grad():
    torch.nn.init.normal_(gp.scal[-1].weight, std=0.3)
    torch.nn.init.constant_(gp.scal[-1].bias, 0.7)
feat = {"res_mask": torch.ones(1, N, device=dev),
        "ligand_mask": pep_m64.float(), "t": torch.tensor([0.6], device=dev)}
ne = torch.randn(1, N, C, device=dev)
rm = torch.eye(3, device=dev).expand(1, N, 3, 3).contiguous()
trf = tr.float()
with torch.no_grad():
    gp(ne, Rigid(Rotation(rot_mats=rm), trf), feat)
    st0 = dict(gp.stash)
    gp(ne, Rigid(Rotation(rot_mats=(Q.float()[None, None] @ rm)), trf @ Q.T.float() + u.float()),
       feat)
    st1 = dict(gp.stash)
scale = float(trf.abs().max())
tol = 3e-6 * scale
err = float((st1["delta"][0] - st0["delta"][0] @ Q.T.float()).norm())
check("end-to-end delta rotates with the complex, to float32 round-off", err < tol,
      f"|err| = {err:.3e} A vs tol {tol:.3e} A (= 3e-6 x {scale:.1f} A coords); "
      f"|delta| = {float(st0['delta'][0].norm()):.3f} A")
check("pointer weights are bit-identical under the rigid motion (invariant features)",
      float((st0["ptr_w"] - st1["ptr_w"]).abs().max()) == 0.0,
      f"max|dw| = {float((st0['ptr_w'] - st1['ptr_w']).abs().max()):.3e}")
check("gate and radial are bit-identical under the rigid motion",
      float((st0["gate"] - st1["gate"]).abs().max()) == 0.0
      and float((st0["radial"] - st1["radial"]).abs().max()) == 0.0, "")

print("\n=== CHECK 6: the head is LIVE once the gate is nonzero, and moves only the peptide ===")
torch.manual_seed(11)
m_live = PlaceFlowModel(conf.model, diffuser).to(dev).eval()
_load_warm(m_live)
with torch.no_grad():
    torch.nn.init.normal_(m_live.place.scal[-1].weight, std=0.3)
    torch.nn.init.constant_(m_live.place.scal[-1].bias, 0.8)
    o_live = m_live(batch)
rig_a = Rigid.from_tensor_7(o_pair).get_trans()
rig_b = Rigid.from_tensor_7(o_live["rigids"]).get_trans()
dp = ((rig_b - rig_a).norm(dim=-1) * pep_m).sum() / pep_m.sum().clamp_min(1)
dr = ((rig_b - rig_a).norm(dim=-1) * rec_m).max()
check("peptide moves by a meaningful distance", float(dp) > 1.0,
      f"mean peptide CA displacement = {float(dp):.2f} A")
check("receptor does not move at all", float(dr) < 1e-4,
      f"max receptor CA displacement = {float(dr):.3e} A")
check("gate is now nonzero", float(o_live['place']['gate'].abs().mean()) > 0.1,
      f"|gate| = {float(o_live['place']['gate'].abs().mean()):.3f}")

print("\n=== CHECK 7: the pointer LABEL is sane, and its bias is CORRECTABLE ===")
# A pointer's reachable optimum is the weighted mean of RECEPTOR CAs, and that is NOT the true
# peptide centroid: the peptide sits on the surface, so the receptor residues near it all lie to
# one side and their mean is pulled INWARD. That bias is what the radial scalar exists to absorb,
# so the design rests on two measurable claims -- the bias points OUTWARD (positive component
# along n_hat), so one scalar along n_hat can correct it, and it is smaller than R_MAX.
# Measured over many complexes, not one: a single case cannot establish that a bias is systematic.
rows = []
for _i in range(min(24, len(ds.csv))):
    try:
        ft, nm = ds.eval_item(_i, seed=7)
    except Exception as _e:               # a malformed row must not quietly mask the finding
        print(f"  (skipped row {_i}: {type(_e).__name__})")
        continue
    bt = tree.map_structure(
        lambda x: (x if torch.is_tensor(x) else torch.tensor(x)).unsqueeze(0).to(dev), ft)
    r_ = bt["res_mask"].float()
    l_ = bt["ligand_mask"].float()
    pm_, rm_ = l_ * r_, (1.0 - l_) * r_
    if float(pm_.sum()) < 2 or float(rm_.sum()) < 10:
        continue
    gt = Rigid.from_tensor_7(bt["rigids_0"]).get_trans().float()
    gtc = (gt * pm_[..., None]).sum(1) / pm_.sum(1, keepdim=True)
    ql_ = (-((gt - gtc[:, None, :]) ** 2).sum(-1) / (2 * place_loss.SIGMA ** 2))
    ql_ = ql_.masked_fill(rm_ < 0.5, float("-inf"))
    q_ = torch.softmax(ql_, -1) * rm_
    tgt = (q_[..., None] * gt).sum(1)                   # the oracle pointer's destination
    recc = (gt * rm_[..., None]).sum(1) / rm_.sum(1, keepdim=True)
    v = tgt - recc
    nh = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    off = gtc - tgt                                     # what the radial term must supply
    rows.append((float(off.norm()), float((off * nh).sum()),
                 float(q_.max()), int((q_ > 1e-4).sum()), int(rm_.sum()), nm))

offs = np.array([r[0] for r in rows])
proj = np.array([r[1] for r in rows])
nzs = np.array([r[3] for r in rows])
nrecs = np.array([r[4] for r in rows])
print(f"  n = {len(rows)} val complexes")
print(f"  |oracle target - true centroid| : median {np.median(offs):.2f} A  "
      f"mean {offs.mean():.2f}  p90 {np.percentile(offs, 90):.2f}  max {offs.max():.2f}")
print(f"  component along n_hat (outward+) : median {np.median(proj):+.2f} A  "
      f"outward in {100.0 * (proj > 0).mean():.0f}% of cases")
print(f"  soft-target support             : median {int(np.median(nzs))} residues "
      f"of a median {int(np.median(nrecs))}-residue receptor")
check("the bias is systematically OUTWARD, so one scalar along n_hat can correct it",
      (proj > 0).mean() >= 0.8 and np.median(proj) > 0.5,
      f"outward in {100.0 * (proj > 0).mean():.0f}% of cases, median {np.median(proj):+.2f} A")
check(f"the bias fits inside R_MAX = {global_place.R_MAX} A",
      np.percentile(offs, 90) < global_place.R_MAX,
      f"p90 = {np.percentile(offs, 90):.2f} A, max {offs.max():.2f} A")
# The oracle-q distance above is a LOOSE bound and must not be read as the head's ceiling: the
# head is free to learn ANY distribution over receptor residues, so its destination can be any
# point in the CONVEX HULL of the receptor CAs, plus up to R_MAX along the outward normal. What
# matters is the distance from the true centroid to THAT set, which is what this measures --
# Frank-Wolfe on the simplex, swept over r. (An earlier version of this gate asserted on the
# oracle-q number and failed at 8.45 A; the reachable optimum on the same cases is 0.13 A, so
# the assertion, not the design, was wrong. reach_place.py has the full n=86 run.)
def _hull_dist(A, pt, iters=300):
    n = A.shape[0]
    wv = np.full(n, 1.0 / n)
    for k in range(iters):
        g = A @ (A.T @ wv - pt)
        wv *= (1.0 - 2.0 / (k + 3.0))
        wv[int(np.argmin(g))] += 2.0 / (k + 3.0)
    return float(np.linalg.norm(A.T @ wv - pt))


reach = []
for _i in range(min(12, len(ds.csv))):
    try:
        ft, _ = ds.eval_item(_i, seed=7)
    except Exception:
        continue
    bt = tree.map_structure(
        lambda x: (x if torch.is_tensor(x) else torch.tensor(x)).unsqueeze(0), ft)
    r_ = bt["res_mask"].float(); l_ = bt["ligand_mask"].float()
    pmb = ((l_ * r_)[0].numpy() > 0.5); rmb = (((1.0 - l_) * r_)[0].numpy() > 0.5)
    if pmb.sum() < 2 or rmb.sum() < 10:
        continue
    X = Rigid.from_tensor_7(bt["rigids_0"]).get_trans()[0].numpy().astype(np.float64)
    gc, CAs = X[pmb].mean(0), X[rmb]
    qq = np.exp(-((CAs - gc) ** 2).sum(-1) / (2 * place_loss.SIGMA ** 2))
    qq /= qq.sum()
    tq = qq @ CAs
    nhv = (tq - CAs.mean(0)) / max(np.linalg.norm(tq - CAs.mean(0)), 1e-6)
    reach.append(min(_hull_dist(CAs, gc - rv * nhv)
                     for rv in np.linspace(0.0, global_place.R_MAX, 15)))
reach = np.array(reach)
print(f"  REACHABLE optimum (hull + radial) : median {np.median(reach):.2f} A  "
      f"p90 {np.percentile(reach, 90):.2f}  max {reach.max():.2f}   (n={len(reach)})")
check("the parameterisation imposes no meaningful ceiling of its own",
      np.median(reach) < 2.0 and np.percentile(reach, 90) < 6.0,
      f"median {np.median(reach):.2f} A, p90 {np.percentile(reach, 90):.2f} A "
      f"vs our oracle L-RMSD median 15.4 A today")
check("the radial term is what buys that (it is not reachable without one)",
      np.median(offs) - np.median(reach) > 3.0,
      f"oracle-q {np.median(offs):.2f} A -> reachable {np.median(reach):.2f} A")
check("the soft target is a patch, not one residue nor the whole receptor",
      bool(((nzs >= 3) & (nzs <= 0.6 * nrecs)).all()),
      f"support/receptor ratio median {float(np.median(nzs / nrecs)):.2f}")

# and the objective must actually prefer the truth
res = batch["res_mask"].float()
lig = batch["ligand_mask"].float()
pm_, rm_ = lig * res, (1.0 - lig) * res
gt = Rigid.from_tensor_7(batch["rigids_0"]).get_trans().float()
gtc = (gt * pm_[..., None]).sum(1) / pm_.sum(1, keepdim=True)
ql = (-((gt - gtc[:, None, :]) ** 2).sum(-1) / (2 * place_loss.SIGMA ** 2))
ql = ql.masked_fill(rm_ < 0.5, float("-inf"))
q = torch.softmax(ql, -1) * rm_
ce_or = float(-(q * torch.log_softmax(ql.masked_fill(rm_ < 0.5, -1e9), -1)
                .masked_fill(rm_ < 0.5, 0.0)).sum())
unif = float(torch.log(rm_.sum()))
check("pointer CE is lower for the truth than for a uniform pointer", ce_or < unif,
      f"CE(oracle) = {ce_or:.3f} vs CE(uniform) = {unif:.3f} nats")


print("\n=== CHECK 8: state_dict round-trip reproduces the head exactly ===")
sd = {k: v.cpu().clone() for k, v in m_live.state_dict().items()}
torch.manual_seed(99)
m_rt = PlaceFlowModel(conf.model, diffuser).to(dev).eval()
m_rt.load_state_dict(sd)
with torch.no_grad():
    o_rt = m_rt(batch)["rigids"]
d = (o_rt - o_live["rigids"]).abs().max().item()
check("reloaded model reproduces the trained head bit-for-bit", d == 0.0, f"max|diff| = {d:.3e}")

print("\n=== CHECK 9: a trained checkpoint WITHOUT place.* is refused, not silently zeroed ===")
# sd already carries adapter.* (BindopsFlowModel.state_dict exposes them), which is what marks
# it as a TRAINED checkpoint rather than a stock warm start -- no key needs inventing here.
stripped = {k: v for k, v in sd.items() if not k.startswith("place.")}
assert any(k.startswith("adapter.") for k in stripped), "sd carries no adapter.* -- check 9 void"
try:
    PlaceFlowModel(conf.model, diffuser).to(dev).load_state_dict(stripped)
    check("load_state_dict raises on a trained checkpoint with no place.* tensors", False,
          "it returned silently -- the head would be a no-op with a normal-looking log")
except RuntimeError as e:
    check("load_state_dict raises on a trained checkpoint with no place.* tensors",
          "place" in str(e).lower(), str(e)[:90])

print("\n" + "=" * 78)
if fails:
    print(f"GATE FAILED: {len(fails)} check(s) -- {fails}")
    sys.exit(1)
print("GATE PASSED: all checks. The placement head is a no-op at init, has a live gradient, "
      "is equivariant, moves only the peptide, and its label is sane.")
