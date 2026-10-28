"""Verify a trained checkpoint SURVIVES RELOAD into the exact class generation will use.

Why this gate exists. `gen_poses_flow.py` loads STRICT, so a shape or name drift there fails
loudly. `gen_poses_pair.py` does not: `PairFlowModel.load_state_dict` sends everything that is
not `pair.*` through `super().load_state_dict(rest, strict=False)` and **throws the returned
missing-key list away**, printing only the pair tensor count. Missing keys are therefore silent.
That is precisely how 425 pretrained IPA tensors were once voided without a word, and how
"BINDops-conditioned" arms came to contain no BINDops. A silent reload failure here would show
up only as bad DockQ three hours later and would be blamed on the pair stack.

Checks (any failure exits 1, before generation spends a card):
  1. adapter is built 24-wide, i.e. the MSA env reached the constructor at INFERENCE time too
  2. reload reports ZERO missing and ZERO unexpected keys
  3. every adapter.* and score-model tensor in the checkpoint actually landed (value-compared
     against the blob, not just key-counted -- a key can be present and still not be loaded)
  4. the adapter's FINAL layer is no longer exactly zero. It is zero-init by construction, so if
     it is still zero after training then the MSA/site features never influenced the model and
     the arm is a control run wearing the wrong label. This is the one check that distinguishes
     "features connected" from "features present but ignored".
  5. NEGATIVE CONTROL: build the same model with the MSA env unset and confirm the 24-dim
     checkpoint does NOT load cleanly. If it does, the checks above prove nothing.

usage: gate_reload.py <ckpt.pth> <msa|all|place>
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os
import sys

import numpy as np
import torch

CK = sys.argv[1]
ARM = sys.argv[2] if len(sys.argv) > 2 else "msa"

os.environ.setdefault("DPB_BASE", "/8TBDrive2/DiffPepBuilder")
os.environ.setdefault("SPEEDDOCK_REPO", "/8TBDrive2/SpeedDocker")
os.environ.setdefault("DPB_NUM_TORSIONS", "3")
os.environ.setdefault("BLIND_PRIOR", "surface")
BASE = os.environ["DPB_BASE"]
V = "/8TBDrive1/jithesh/docking_model/verified_v2"
fails = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("   " + detail if detail else ""), flush=True)
    if not cond:
        fails.append(name)


def build(with_msa):
    """Construct the model exactly as the generation script will, in a fresh import state."""
    for m in [k for k in list(sys.modules) if k.startswith(("train_bindops_flow", "train_pair_flow",
                                                            "train_place_flow", "global_place",
                                                            "model", "data", "speeddock"))]:
        del sys.modules[m]
    if with_msa:
        os.environ["BINDOPS_PROFILE"] = "1"
        os.environ["BINDOPS_MODE"] = "flexible"
        os.environ["MSA_PROFILE_PKL"] = V + "/msa/msa_profiles.pkl"
        os.environ["SITE_SCORE_PKL"] = V + "/msa/site_scores.pkl"
    else:
        os.environ["BINDOPS_PROFILE"] = "1"
        os.environ["BINDOPS_MODE"] = "flexible"
        os.environ.pop("MSA_PROFILE_PKL", None)
        os.environ.pop("SITE_SCORE_PKL", None)
    if ARM in ("all", "place"):
        os.environ.setdefault("PAIR_STACK", "1")
        os.environ.setdefault("PAIR_C", "64")
        os.environ.setdefault("PAIR_C_HIDDEN", "32")
        os.environ.setdefault("PAIR_LAYERS", "4")
    from bindops.training import train_bindops_flow as B
    from hydra import compose, initialize_config_dir
    from data import se3_diffuser
    csv = V + "/data/metadata_val_real_A.csv"
    with initialize_config_dir(version_base=None, config_dir=f"{BASE}/config"):
        conf = compose(config_name="base", overrides=[
            f"data.train_csv_path={csv}", f"data.val_csv_path={csv}",
            "data.num_eval_samples=null", "data.center_pos_noise=False",
            "experiment.use_ddp=False"])
    diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)
    if ARM == "place":
        from bindops.training import train_place_flow as PL
        return PL.PlaceFlowModel(conf.model, diffuser)
    if ARM == "all":
        from bindops.training import train_pair_flow as P
        return P.PairFlowModel(conf.model, diffuser)
    return B.BindopsFlowModel(conf.model, diffuser)


blob = torch.load(CK, map_location="cpu")
sd = blob["model"] if "model" in blob else blob
print(f"checkpoint {CK}: {len(sd)} tensors, arm={ARM}")
print()

model = build(True)
first = model.adapter.net[0]
check("1. adapter built 24-wide at INFERENCE construction", first.in_features == 24,
      "in_features = %d" % first.in_features)

before = {k: v.detach().clone() for k, v in model.state_dict().items()}
res = model.load_state_dict(sd)

# DO NOT trust the returned _IncompatibleKeys. Self-testing this gate showed that feeding
# PairFlowModel a checkpoint with zero pair.* tensors returns missing_keys == [] while printing
# "pair stack tensors loaded: 0" -- the override chain does not report them. Compare key sets
# directly, both directions.
model_keys = set(model.state_dict())
ckpt_keys = set(sd)
missing = sorted(model_keys - ckpt_keys)
unexpected = sorted(ckpt_keys - model_keys)
reported_missing = list(getattr(res, "missing_keys", []) or [])
if len(reported_missing) != len(missing):
    print("    NOTE: load_state_dict reported %d missing, direct key comparison finds %d -- "
          "the returned object is unreliable here, which is why this gate does not use it."
          % (len(reported_missing), len(missing)))
check("2a. no model tensor is left unloaded (direct key comparison)", not missing,
      "model tensors absent from checkpoint: %d%s"
      % (len(missing), (": " + ", ".join(missing[:6])) if missing else ""))
check("2b. no checkpoint tensor is unused (direct key comparison)", not unexpected,
      "checkpoint tensors absent from model: %d%s"
      % (len(unexpected), (": " + ", ".join(unexpected[:6])) if unexpected else ""))

# a loaded tensor must actually have CHANGED from its init, unless it happened to equal it
after = model.state_dict()
unchanged = [k for k in ckpt_keys & model_keys
             if before[k].shape == after[k].shape
             and torch.equal(before[k].float(), after[k].float())
             and not torch.equal(after[k].float(), sd[k].float().reshape(after[k].shape))]
check("2c. no checkpoint tensor silently failed to overwrite its init", not unchanged,
      "silently-not-loaded: %d%s" % (len(unchanged), (": " + ", ".join(unchanged[:6])) if unchanged else ""))

if ARM in ("all", "place"):
    pk = [k for k in ckpt_keys if k.startswith("pair.")]
    mpk = [k for k in model_keys if k.startswith("pair.")]
    check("2d. pair stack tensors present in the checkpoint AND all loaded",
          len(pk) > 0 and len(pk) == len(mpk),
          "checkpoint pair.* = %d, model pair.* = %d" % (len(pk), len(mpk)))

if ARM == "place":
    lk = [k for k in ckpt_keys if k.startswith("place.")]
    mlk = [k for k in model_keys if k.startswith("place.")]
    check("2e. placement head tensors present in the checkpoint AND all loaded",
          len(lk) > 0 and len(lk) == len(mlk),
          "checkpoint place.* = %d, model place.* = %d" % (len(lk), len(mlk)))

# 3. value-compare, because a key can be reported fine and still not be the loaded value
own = dict(model.state_dict())
nmatch = ndiff = nabsent = 0
diffs = []
for k, v in sd.items():
    if k not in own:
        nabsent += 1
        continue
    if own[k].shape != v.shape:
        ndiff += 1
        diffs.append(k + " shape")
        continue
    if torch.allclose(own[k].float(), v.float(), atol=0, rtol=0):
        nmatch += 1
    else:
        ndiff += 1
        diffs.append(k)
check("3. every checkpoint tensor landed bit-exactly", ndiff == 0 and nabsent == 0,
      "matched %d, differing %d, absent from model %d%s"
      % (nmatch, ndiff, nabsent, ("  e.g. " + ", ".join(diffs[:5])) if diffs else ""))

# 4. did the features actually train the adapter off its zero init?
last = [m for m in model.adapter.net if hasattr(m, "weight")][-1]
w = last.weight.detach().abs()
check("4. adapter FINAL layer is no longer zero (features were actually learned from)",
      float(w.max()) > 0,
      "max|W_final| = %.3e  (zero-init by construction, so zero here = features ignored)"
      % float(w.max()))
if ARM == "place":
    # THE check that decides this arm. scal[-1] is zero-initialised by construction (the exact
    # no-op that preserves the warm start), so if it is still zero then tanh(0) = 0 held for the
    # whole run, delta was identically zero, and these poses ARE the pair-stack arm's. That is a
    # mechanical null, not a scientific one, and the two must never be confused.
    sl = model.place.scal[-1].weight.detach().abs()
    sb = model.place.scal[-1].bias.detach().abs()
    check("4d. placement head is OFF its zero init (the head actually engaged)",
          float(sl.max()) > 0 or float(sb.max()) > 0,
          "max|W_scal| = %.3e  max|b_scal| = %.3e  (both zero = delta was 0 for the whole run)"
          % (float(sl.max()), float(sb.max())))
    pl = [m for m in model.place.ptr if hasattr(m, "weight")][-1].weight.detach().abs()
    check("4e. pointer head has trained", float(pl.max()) > 0,
          "max|W_ptr| = %.3e" % float(pl.max()))

adapter_keys = [k for k in sd if k.startswith("adapter.")]
check("4b. checkpoint actually carries adapter tensors", len(adapter_keys) > 0,
      "%d adapter.* tensors" % len(adapter_keys))

# 5. negative control
print()
print("--- negative control: same reload with the MSA env UNSET (must NOT load cleanly) ---")
try:
    bad = build(False)
    bw = bad.adapter.net[0].in_features
    print("    adapter in_features without MSA env: %d" % bw)
    try:
        r2 = bad.load_state_dict(sd)
        m2 = list(getattr(r2, "missing_keys", []) or [])
        u2 = list(getattr(r2, "unexpected_keys", []) or [])
        clean = (bw == 24) and not m2 and not u2
        check("5. negative control FAILS to load cleanly (so checks 1-3 are meaningful)",
              not clean, "in_features=%d missing=%d unexpected=%d" % (bw, len(m2), len(u2)))
    except Exception as e:
        check("5. negative control FAILS to load cleanly (so checks 1-3 are meaningful)", True,
              "raised %s: %s" % (type(e).__name__, str(e)[:90]))
except Exception as e:
    check("5. negative control FAILS to load cleanly (so checks 1-3 are meaningful)", True,
          "construction raised %s" % type(e).__name__)

print()
if fails:
    print("RELOAD GATE FAILED:", fails)
    print("NOT generating: a silent reload failure would look like a bad architecture result.")
    sys.exit(1)
print("RELOAD GATE PASSED: the trained checkpoint reloads intact, with features connected.")
