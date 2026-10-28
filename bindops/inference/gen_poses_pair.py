"""Pose generation from a PAIR-STACK checkpoint. Mirrors gen_poses_flow.py exactly, except the
class handed to make_poseset is PairFlowModel rather than BindopsFlowModel.

make_poseset calls `model.load_state_dict(ck['model'])` positionally, so PairFlowModel's own
override (strict=False, routing `pair.*` and `adapter.*`) is what runs -- the `pair.*` tensors
are restored rather than rejected. Verified by verify_pair_stack.py CHECK 2.

usage: PAIR_STACK=1 gen_poses_pair.py <ckpt> <csv> <outdir> [N] [steps] [--gpu G]
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os, sys, runpy
os.environ.setdefault("BINDOPS_MODE", "control")
os.environ.setdefault("DPB_BASE", "/8TBDrive2/DiffPepBuilder")
os.environ.setdefault("SPEEDDOCK_REPO", "/8TBDrive2/SpeedDocker")
os.environ.setdefault("DPB_NUM_TORSIONS", "3")

from bindops.training import train_pair_flow as P            # brings in train_bindops_flow's patches + PairFlowModel
import model as _pkg
import model.score_network as _real_mod


class _ScoreNetworkProxy:
    ScoreNetwork = P.PairFlowModel

    def __getattr__(self, k):
        return getattr(_real_mod, k)


# Same ConfidenceHead shims as gen_poses_flow.py: make_poseset was written against a newer head
# exposing from_state_dict/.gvp, while the installed speeddock head is the plain version our
# checkpoints match. The head only produces a ranking readout and never touches a coordinate.
from speeddock.confidence import ConfidenceHead as _CH
if not hasattr(_CH, "from_state_dict"):
    def _from_sd(cls, sd, c_s=512, **kw):
        h = cls(c_s=c_s)
        h.load_state_dict(sd, strict=True)
        return h
    _CH.from_state_dict = classmethod(_from_sd)
if not hasattr(_CH, "gvp"):
    _CH.gvp = None
_orig_fwd = _CH.forward
if _orig_fwd.__code__.co_argcount < 4:
    def _fwd(self, s, mask, *ignored, **kw):
        return _orig_fwd(self, s, mask)
    _CH.forward = _fwd

_pkg.score_network = _ScoreNetworkProxy()
print("[pair-flow-eval] model.score_network.ScoreNetwork -> PairFlowModel (package attr only)",
      flush=True)
runpy.run_path(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "make_poseset.py"), run_name="__main__")
