"""Gates the pair stack before any GPU time is spent on it. Every check must pass.

The important one is CHECK 5: a direct demonstration that the new mechanism supplies the thing
the stock edge update cannot. Perturb edge (i,k) ONLY, and ask whether edge (i,j) moves.
Triangle multiplication must propagate it (out_ij = sum_k a_ik b_jk); the stock EdgeTransition
provably cannot, since it reads only the endpoint nodes i, j and edge (i,j) itself. Running both
in the same script turns the audit's claim into a measurement.
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os, sys
import numpy as np
import torch

os.environ.setdefault("BINDOPS_MODE", "control")
os.environ.setdefault("DPB_BASE", "/8TBDrive2/DiffPepBuilder")
os.environ.setdefault("SPEEDDOCK_REPO", "/8TBDrive2/SpeedDocker")
os.environ.setdefault("DPB_NUM_TORSIONS", "3")
os.environ.setdefault("BLIND_PRIOR", "surface")

from bindops.training import train_bindops_flow as B
from bindops.training.train_pair_flow import PairFlowModel
from bindops.model.pair_stack import PairStack
from hydra import initialize_config_dir, compose
from data import se3_diffuser
from speeddock.dataset import PlaceDataset
import tree

BASE = os.environ["DPB_BASE"]
WARM = "/8TBDrive2/DiffPepBuilder/experiments/checkpoints/diffpepdock_v1.pth"
VAL = "/8TBDrive1/jithesh/docking_model/verified_v2/data/metadata_val.csv"
dev = "cuda" if torch.cuda.is_available() else "cpu"
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        fails.append(name)


with initialize_config_dir(version_base=None, config_dir=f"{BASE}/config"):
    conf = compose(config_name="base", overrides=[
        f"data.train_csv_path={VAL}", f"data.val_csv_path={VAL}",
        "data.num_eval_samples=null", "data.center_pos_noise=False",
        "experiment.use_ddp=False"])
diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)

# BOTH models are loaded with the real 240k-step DiffPepDock weights, not left at init.
# This is load-bearing, not tidiness: BackboneUpdate and IPA's linear_out are built with
# init="final" (zero-initialised), so on a FRESHLY initialised model every per-block rigid
# update is exactly 0, the output frames equal the input frames whatever the edges say, and
# no gradient can reach the edge path at all. Checks 3 and 4 against an untrained model would
# therefore both read 0 and appear to "prove" the edge channel dead -- an artifact of the test
# harness, not a property of the model.
def _load_warm(m):
    blob = torch.load(WARM, map_location=dev)
    ck = {k.replace("module.", ""): v for k, v in blob["model"].items()}
    msd = m.state_dict()                   # the same filter train_flow_v4.py applies
    ck = {k: v for k, v in ck.items() if k in msd and v.shape == msd[k].shape}
    m.load_state_dict(ck, strict=False)
    return len(ck)

print()
print("=== CHECK 1: pair stack is an EXACT no-op at init (warm start preserved) ===")
torch.manual_seed(0)
m_plain = B.BindopsFlowModel(conf.model, diffuser).to(dev).eval()
torch.manual_seed(0)
m_pair = PairFlowModel(conf.model, diffuser).to(dev).eval()
n1, n2 = _load_warm(m_plain), _load_warm(m_pair)
check("both models load the same pretrained tensors", n1 == n2 and n1 > 400,
      f"{n1} vs {n2} tensors")

ds = PlaceDataset(data_conf=conf.data, diffuser=diffuser, is_training=False,
                  flow=True, mask_pep_esm=False, seed=0)
feats, name = ds.eval_item(0, seed=7)
batch = tree.map_structure(
    lambda x: (x if torch.is_tensor(x) else torch.tensor(x)).unsqueeze(0).to(dev), feats)
batch["t"] = torch.as_tensor([float(feats["t"])], device=dev)

with torch.no_grad():
    o1 = m_plain(batch)["rigids"]
    o2 = m_pair(batch)["rigids"]
d = (o1 - o2).abs().max().item()
check("output identical with pair stack at init", d == 0.0, f"max|diff| = {d:.3e}")

print("\n=== CHECK 2: warm start still matches every pretrained key ===")
blob = torch.load(WARM, map_location="cpu")
ck = {k.replace("module.", ""): v for k, v in blob["model"].items()}
msd = m_pair.state_dict()
dropped = [k for k, v in ck.items() if k not in msd or v.shape != msd[k].shape]
check("no pretrained tensor dropped by the state_dict filter", len(dropped) == 0,
      f"{len(ck)} pretrained tensors, {len(dropped)} dropped")
npair = len([k for k in msd if k.startswith("pair.")])
check("pair.* keys exist and are absent from the stock checkpoint",
      npair > 0 and not any(k.startswith("pair.") for k in ck), f"{npair} pair tensors")

print()
print("=== CHECK 3: pair stack is WIRED (not silently bypassed) ===")
# Two separate assertions, because one combined check cannot tell "the pair stack was never
# called" apart from "the edge tensor has no influence on the output".
with torch.no_grad():
    _probe = {}
    _po = m_pair.pair.forward
    def _spy(z, pm):
        out = _po(z, pm)
        _probe["delta"] = (out - z).abs().max().item()
        return out
    m_pair.pair.forward = _spy
    for p in m_pair.pair.out_proj.parameters():
        p.add_(torch.randn_like(p) * 0.05)
    o3 = m_pair(batch)["rigids"]
    m_pair.pair.forward = _po
d3 = (o1 - o3).abs().max().item()
check("pair stack is called and perturbs the edge tensor", _probe.get("delta", 0) > 1e-6,
      f"max|edge diff| = {_probe.get('delta', 0):.3e}")
check("that edge change propagates to the model output", d3 > 1e-6, f"max|diff| = {d3:.3e}")

print("\n=== CHECK 4: gradients reach the pair stack ===")
m_pair.train()
out = m_pair(batch)
from speeddock.losses import flow_loss
loss, _ = flow_loss(out, batch)
loss.backward()
gn = [(n, p.grad.abs().sum().item()) for n, p in m_pair.pair.named_parameters()
      if p.grad is not None]
nz = [n for n, g in gn if g > 0]
check("pair-stack parameters receive nonzero gradient", len(nz) > 0,
      f"{len(nz)}/{len(list(m_pair.pair.parameters()))} tensors with grad > 0")

print("\n=== CHECK 5: THE MECHANISM -- does edge (i,j) see edge (i,k)? ===")
torch.manual_seed(1)
N, C = 24, conf.model.edge_embed_size
ps = PairStack(C, c_pair=32, c_hidden=16, n_layers=2).to(dev).eval()
with torch.no_grad():                       # un-zero the output so the stack is live
    ps.out_proj.weight.normal_(0, 0.05); ps.out_proj.bias.normal_(0, 0.05)
z = torch.randn(1, N, N, C, device=dev)
pm = torch.ones(1, N, N, device=dev)
i, j, k = 3, 11, 19
with torch.no_grad():
    base = ps(z, pm)[0, i, j].clone()
    z2 = z.clone(); z2[0, i, k] += 5.0      # perturb a THIRD-PARTY edge only
    pert = ps(z2, pm)[0, i, j].clone()
dtri = (base - pert).abs().max().item()
check("triangle stack propagates edge (i,k) -> edge (i,j)", dtri > 1e-5,
      f"max|diff| = {dtri:.3e}")

# the control: the stock edge update, same perturbation
from model.ipa_module import EdgeTransition
et = EdgeTransition(node_embed_size=conf.model.node_embed_size,
                    edge_embed_in=C, edge_embed_out=C).to(dev).eval()
ne = torch.randn(1, N, conf.model.node_embed_size, device=dev)
with torch.no_grad():
    b0 = et(ne, z)[0, i, j].clone()
    b1 = et(ne, z2)[0, i, j].clone()
dstock = (b0 - b1).abs().max().item()
check("stock EdgeTransition does NOT (confirms the representational gap)", dstock < 1e-6,
      f"max|diff| = {dstock:.3e}")

print("\n=== CHECK 6: padding invariance ===")
with torch.no_grad():
    zp = torch.cat([z, torch.randn(1, N, 8, C, device=dev)], dim=2)
    zp = torch.cat([zp, torch.randn(1, 8, N + 8, C, device=dev)], dim=1)
    pmp = torch.zeros(1, N + 8, N + 8, device=dev); pmp[:, :N, :N] = 1.0
    a = ps(z, pm)[0, :N, :N]
    b = ps(zp, pmp)[0, :N, :N]
dp = (a - b).abs().max().item()
check("padded residues do not change real entries", dp < 1e-3, f"max|diff| = {dp:.3e}")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILED: {fails}"))
sys.exit(1 if fails else 0)
