"""Pose generation from a BINDops-conditioned FLOW checkpoint.

Runs scoring_model/make_poseset.py (Stage 1's own flow inference path) with the patches from
train_bindops_flow.py applied on import:
  * ScoreNetwork -> BindopsFlowModel, because make_poseset does `model.load_state_dict(ck['model'])`
    STRICT into a plain ScoreNetwork and our checkpoints carry `adapter.*` keys.
  * PdbDataset._process_csv_row -> injects bindops_score, keeps the BINDops-centred frame
    (PlaceDataset inherits it).

The swap goes on the PACKAGE attribute (`model.score_network`), never on the module object itself:
make_poseset does `from model import score_network`, which resolves through the package, while
score_network.py's own globals stay intact. Rebinding the module's own attribute makes its internal
`super(ScoreNetwork, self).__init__()` resolve to the wrapper and raise
"TypeError: super(type, obj): obj must be an instance or subtype of type".

For the control arm the adapter is zero, so this is plain unconditioned flow inference -- verified
at exactly 0.000e+00 difference by verify_flow_wiring.py.
usage: BINDOPS_MODE=<mode> gen_poses_flow.py <ckpt> <csv> <outdir> [N] [steps] [--gpu G]
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

from bindops.training import train_bindops_flow as B          # model + loader patches, and the real module reference
import model as _pkg
import model.score_network as _real_mod


class _ScoreNetworkProxy:
    """What `from model import score_network` will return: our class, everything else delegated."""
    ScoreNetwork = B.BindopsFlowModel

    def __getattr__(self, k):
        return getattr(_real_mod, k)


# make_poseset.py was written against a newer ConfidenceHead that exposes a from_state_dict
# classmethod and a .gvp attribute (for --conf-vec-feats runs). The installed speeddock
# confidence.py is the plain version, and our checkpoints' head is plain `net.*` to match. Shim the
# two names rather than editing their script. The head only produces a confidence readout; it takes
# no part in generating the pose, so reconstructing it plainly changes no coordinate.
from speeddock.confidence import ConfidenceHead as _CH
if not hasattr(_CH, "from_state_dict"):
    def _from_sd(cls, sd, c_s=512, **kw):
        h = cls(c_s=c_s)
        h.load_state_dict(sd, strict=True)
        return h
    _CH.from_state_dict = classmethod(_from_sd)
if not hasattr(_CH, "gvp"):
    _CH.gvp = None
# The newer head's forward takes an extra coordinate argument for its GVP branch. With gvp=None
# there is no such branch, so accept and ignore any extras. The head only emits a confidence
# readout used for ranking in labels.csv; it never touches the pose coordinates.
_orig_fwd = _CH.forward
if _orig_fwd.__code__.co_argcount < 4:
    def _fwd(self, s, mask, *ignored, **kw):
        return _orig_fwd(self, s, mask)
    _CH.forward = _fwd

_pkg.score_network = _ScoreNetworkProxy()
print(f"[bindops-flow-eval] MODE={os.environ['BINDOPS_MODE']}; "
      f"model.score_network.ScoreNetwork -> BindopsFlowModel (package attr only)", flush=True)
runpy.run_path(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "make_poseset.py"), run_name="__main__")
