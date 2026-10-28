"""Flow generator + PAIR STACK + GLOBAL PLACEMENT HEAD (see global_place.py for the why).

A thin wrapper over train_pair_flow.py, which is itself a thin wrapper over train_bindops_flow.py,
so every verified patch in both is reused unchanged: the fixed per-pose surface prior, the
`_process_csv_row` loader, the backbone-transparent state_dict, the warm-start filtering, and the
pair stack's instance-level forward splice. Nothing is forked.

THE BASE IS `run_full.sh all`, deliberately -- same 6000 steps, batch, lr, seed, val CSV,
recycling, MSA + site features, surface prior, pair stack. That is the arm whose DockQ is on
record (oracle 0.128 / acceptable 26.2%, top-1 0.035 / 3.1%) and the same base the contact arm
used, so all three are directly comparable and this arm has exactly ONE new variable.

HOW THE HEAD IS INSERTED. Same discipline as the pair stack, for the same reason: wrapping
`score_network.score_model` in a module would move the real IPA trunk's parameters from
`score_model.trunk.*` to `score_model.real.trunk.*`, and the warm-start filter (`k in
model.state_dict()`) would then drop all 240k steps of pretrained DiffPepDock weights WITHOUT
erroring -- the model would train from scratch while the log still said it had warm-started. So
the head lives as a sibling submodule (`place.*`, a name absent from every stock checkpoint,
exactly like `adapter.*` and `pair.*`) and is spliced in by replacing the score_model INSTANCE's
bound `forward`. Module structure, and therefore every checkpoint key, is untouched.

ORDER OF THE TWO SPLICES MATTERS AND IS CHECKED. train_pair_flow has already replaced
`sm.forward` by the time this runs, so `_real` captured below is the PAIR-SPLICED function, and
the composition is pair-stack-then-trunk-then-placement. Capturing it in the other order would
silently drop the pair stack. assert_spliced() below verifies the composition is live rather than
trusting the import order.

TWO DEAD OUTPUTS, stated so nobody re-derives it later. `rot_score` and `trans_score` are computed
inside the trunk from the PRE-placement frames, so after this head runs they no longer describe
the returned rigids. They are safe to leave stale because nothing consumes them: the flow loss
reads only out['rigids'] and out['atom37'], and the sampler (make_poseset.infer_flow) reads only
out['rigids'] via interp_frames. Grepped both, 2026-10-19. If a future score-matching objective
is added, it must recompute them.

usage: PLACE_HEAD=1 PLACE_LOSS=1 BINDOPS_MODE=flexible BLIND_PRIOR=surface PAIR_STACK=1 \
         train_place_flow.py <out_dir> <data_dir> [flow args...]
knobs: PLACE_C_HIDDEN=128 PLACE_TEMP=1.0 W_POINTER=10.0 W_CENTROID=1.0 POINTER_SIGMA=6.0
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os, sys
import torch

os.environ.setdefault("BINDOPS_MODE", "control")

from bindops.training import train_pair_flow as P            # pair stack splice + every train_bindops_flow patch
from bindops.training import train_bindops_flow as B
from bindops.model.global_place import GlobalPlacement


class PlaceFlowModel(P.PairFlowModel):
    """PairFlowModel + one global pointer-based translation of the peptide per complex."""

    def __init__(self, model_conf, diffuser):
        super().__init__(model_conf, diffuser)
        self.place = GlobalPlacement(model_conf.node_embed_size)
        sm = self.score_network.score_model
        _real = sm.forward                 # ALREADY the pair-spliced bound function; see above
        _pl = self.place
        self._pair_spliced = "_forward_with_pair" in getattr(_real, "__name__", "")

        def _forward_with_place(init_node_embed, edge_embed, input_feats):
            out = dict(_real(init_node_embed, edge_embed, input_feats))
            out["final_rigids"] = _pl(out["node_embed"], out["final_rigids"], input_feats)
            return out

        sm.forward = _forward_with_place
        n = sum(p.numel() for p in self.place.parameters())
        print(f"[place-flow] GlobalPlacement: c_s={model_conf.node_embed_size}, "
              f"{n/1e6:.2f}M params, zero-init gate (exact no-op at step 0, live gradient), "
              f"pair_spliced={self._pair_spliced}", flush=True)
        if not self._pair_spliced:
            raise RuntimeError(
                "[place-flow] the pair stack splice is NOT underneath this head. train_pair_flow "
                "must have replaced score_model.forward before PlaceFlowModel.__init__ captured "
                "it; capturing in the other order drops the pair stack silently.")

    def forward(self, input_feats):
        # Surface the placement stash on the model output so place_loss can read the pointer
        # logits and the gate. Cleared immediately: holding these on the module would pin the
        # whole graph alive past the optimiser step.
        out = super().forward(input_feats)
        st = getattr(self.place, "stash", None)
        if st is not None:
            out["place"] = st
            self.place.stash = None
        return out

    # `place.*` rides alongside `adapter.*` and `pair.*`: absent from stock checkpoints, so it
    # stays zero-init on a warm start and reloads on --resume. Without this the head trains and
    # is then thrown away at save time, which is indistinguishable from a null arm.
    def state_dict(self, *a, **kw):
        sd = dict(super().state_dict(*a, **kw))
        for k, v in self.place.state_dict().items():
            sd[f"place.{k}"] = v
        return sd

    def load_state_dict(self, sd, strict=False):
        pl = {k[len("place."):]: v for k, v in sd.items() if k.startswith("place.")}
        rest = {k: v for k, v in sd.items() if not k.startswith("place.")}
        res = super().load_state_dict(rest, strict=False)
        print(f"[place-flow] placement head tensors loaded: {len(pl)}", flush=True)
        # Same audit as the pair stack's, for the same failure: a checkpoint that carries
        # adapter.* is a TRAINED checkpoint, so place.* being absent from it means the head would
        # silently revert to its zero init (an exact no-op) while the log looked normal.
        if not pl and any(k.startswith("adapter.") for k in sd) \
                and os.environ.get("PLACE_ALLOW_NO_HEAD") != "1":
            raise RuntimeError(
                "[place-flow] this checkpoint carries adapter.* but no place.* tensors, so the "
                "placement head would stay at its zero init (an exact no-op) while the log looked "
                "normal. If that is intended, set PLACE_ALLOW_NO_HEAD=1.")
        if pl:
            inc = self.place.load_state_dict(pl, strict=True)
            del inc
        return res


class _Proxy:
    ScoreNetwork = PlaceFlowModel

    def __getattr__(self, k):
        from model import score_network as _sn
        return getattr(_sn, k)


B.T.score_network = _Proxy()

if __name__ == "__main__":
    # read the weights from place_loss itself rather than re-reading the env with a second set
    # of defaults: the two drifted once already (this banner said 10.0 while the loss used 2.0),
    # and a banner that disagrees with the running loss is worse than no banner in a log that
    # gets read back weeks later.
    from bindops.losses import place_loss as _pl
    from bindops.model import global_place as _gp
    print(f"[place-flow] MODE={os.environ['BINDOPS_MODE']} "
          f"PAIR_LAYERS={os.environ.get('PAIR_LAYERS','4')} "
          f"W_POINTER={_pl.W_POINTER} W_CENTROID={_pl.W_CENTROID} "
          f"SIGMA={_pl.SIGMA} BETA={_pl.BETA} R_MAX={_gp.R_MAX}", flush=True)
    B.T.main()
