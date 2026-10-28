"""Pose generation from a GLOBAL-PLACEMENT checkpoint. Mirrors gen_poses_pair.py exactly, except
the class handed to make_poseset is PlaceFlowModel rather than PairFlowModel.

make_poseset calls `model.load_state_dict(ck['model'])` positionally, so PlaceFlowModel's own
override is what runs -- `place.*`, `pair.*` and `adapter.*` are all routed and restored rather
than dropped. And its audit raises if a trained checkpoint carries no `place.*`, which is the one
failure that would otherwise look exactly like a normal run: the head would silently revert to its
zero init, generate the pair-stack arm's poses, and be reported as a null placement result.
Verified by verify_place.py CHECKS 8 and 9.

PLACE_LOSS is NOT needed here, unlike CONTACT_LOSS on the contact arm: the placement head is built
unconditionally by PlaceFlowModel, so its tensors always have a home. Only the loss wrapper is env-
gated, and no loss runs at inference.

usage: PAIR_STACK=1 gen_poses_place.py <ckpt> <csv> <outdir> [N] [steps] [--gpu G]
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

from bindops.training import train_place_flow as PL         # train_pair_flow + train_bindops_flow patches + the head
import model as _pkg
import model.score_network as _real_mod


class _ScoreNetworkProxy:
    ScoreNetwork = PL.PlaceFlowModel

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
print("[place-flow-eval] model.score_network.ScoreNetwork -> PlaceFlowModel (package attr only)",
      flush=True)
runpy.run_path(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "make_poseset.py"), run_name="__main__")
