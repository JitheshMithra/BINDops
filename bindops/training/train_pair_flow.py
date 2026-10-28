"""Flow generator + triangle-multiplicative PAIR STACK (see pair_stack.py for the why).

This is a thin wrapper over train_bindops_flow.py so that EVERY verified patch in that file is
reused unchanged -- the fixed per-pose surface prior, the `_process_csv_row` loader, the
backbone-transparent state_dict, the warm-start filtering. Nothing is forked.

HOW THE PAIR STACK IS INSERTED, and why it looks indirect. The obvious move -- wrapping
`score_network.score_model` in a module that runs the pair stack and then delegates -- is WRONG
here, and silently so: the real IPA trunk's parameters would move from `score_model.trunk.*` to
`score_model.real.trunk.*`, and train_flow_v4's warm-start filter (`k in model.state_dict()`)
would then drop all 240k steps of pretrained DiffPepDock weights without erroring. The model
would train from scratch while the log still said it had warm-started.

So instead the pair stack lives as a sibling submodule (`pair.*`, a name absent from every stock
checkpoint, exactly like `adapter.*`) and is spliced in by replacing the score_model INSTANCE's
bound `forward`. Module structure, and therefore every checkpoint key, is untouched.

usage: PAIR_STACK=1 BINDOPS_MODE=control BLIND_PRIOR=surface \
         train_pair_flow.py <out_dir> <data_dir> [flow args...]
knobs: PAIR_C=64  PAIR_C_HIDDEN=32  PAIR_LAYERS=4
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import os, sys
import torch

os.environ.setdefault("BINDOPS_MODE", "control")

# PAIR_STACK is the documented on/off knob for the stack and until 2026-10-21 nothing read it, so
# PAIR_STACK=0 ran the full stack in silence. Parsed strictly: an unrecognised value raises rather
# than falling back to "on", because a typo that silently enables the stack is the failure this is
# being added to prevent.
_ps = os.environ.get("PAIR_STACK", "1").strip()
if _ps not in ("0", "1"):
    raise SystemExit("PAIR_STACK must be '0' or '1', got %r. Refusing to guess -- an unrecognised "
                     "value used to mean 'on' by accident." % _ps)
PAIR_STACK_ON = _ps == "1"
if not PAIR_STACK_ON:
    print("[pair-flow] PAIR_STACK=0: pair stack DISABLED -- not constructed, not spliced, and "
          "absent from the state dict", flush=True)

from bindops.training import train_bindops_flow as B          # applies every patch, incl. the fixed surface prior
from bindops.model.pair_stack import PairStack


def audit_pair_load(ckpt_keys, n_pair_loaded, missing, unexpected):
    """Raise on the two ways a pair-stack load silently throws weights away.

    (1) The backbone did not load completely. A stock DiffPepDock warm start gives
        missing=0 unexpected=0 (measured: train_full_all.log, 2026-10-16), and so does a resume,
        because train_flow_v4 filters the checkpoint to `k in model.state_dict()` first. Anything
        else means pretrained tensors were dropped by strict=False without an error.
    (2) The checkpoint is a trained BINDops/pair checkpoint (it carries adapter.*) but has no
        pair.* tensors, so the stack is left at its zero init -- an exact no-op -- while
        "[pair-flow] pair stack tensors loaded: 0" looks identical to a legitimate warm start.
        That line is CORRECT from diffpepdock_v1.pth and WRONG on a reload; this tells them
        apart. Set PAIR_ALLOW_NO_STACK=1 to warm-start a pair model from a stackless BINDops
        checkpoint on purpose.
    """
    if not PAIR_STACK_ON and any(k.startswith("pair.") for k in ckpt_keys) \
            and os.environ.get("PAIR_ALLOW_NO_STACK") != "1":
        raise RuntimeError(
            "[pair-flow] PAIR_STACK=0 but this checkpoint CARRIES pair.* tensors, which would be "
            "silently discarded -- the mirror image of the case below. If dropping them is "
            "intended, set PAIR_ALLOW_NO_STACK=1.")
    if missing or unexpected:
        raise RuntimeError(
            "[pair-flow] strict=False would have silently discarded pretrained weights: "
            f"{len(missing)} model tensors absent from the checkpoint {list(missing)[:6]}, "
            f"{len(unexpected)} checkpoint tensors unused {list(unexpected)[:6]}")
    if n_pair_loaded == 0 and any(k.startswith("adapter.") for k in ckpt_keys)             and os.environ.get("PAIR_ALLOW_NO_STACK") != "1":
        raise RuntimeError(
            "[pair-flow] this checkpoint carries adapter.* but no pair.* tensors, so the pair "
            "stack would stay at its zero init (an exact no-op) while the log looked normal. "
            "If that is intended, set PAIR_ALLOW_NO_STACK=1.")


class PairFlowModel(B.BindopsFlowModel):
    """BindopsFlowModel + a pair stack on the edge representation before the IPA trunk."""

    def __init__(self, model_conf, diffuser):
        super().__init__(model_conf, diffuser)
        if not PAIR_STACK_ON:
            # Leave the module tree exactly as the stackless model: no `pair.*` keys, no splice.
            return
        self.pair = PairStack(model_conf.edge_embed_size)
        sm = self.score_network.score_model
        _real_forward = sm.forward        # the bound method of the untouched real module
        _pair = self.pair

        def _forward_with_pair(init_node_embed, edge_embed, input_feats):
            node_mask = input_feats["res_mask"].type(torch.float32)
            pair_mask = node_mask[..., None] * node_mask[..., None, :]
            edge_embed = _pair(edge_embed, pair_mask)
            return _real_forward(init_node_embed, edge_embed, input_feats)

        # Instance attribute, not a class override: nn.Module.__call__ resolves `self.forward`
        # through normal attribute lookup, so this takes effect while leaving the module tree --
        # and every state_dict key -- exactly as the pretrained checkpoint expects.
        sm.forward = _forward_with_pair
        n = sum(p.numel() for p in self.pair.parameters())
        print(f"[pair-flow] PairStack: {_pair.n_layers} layers, c_pair={_pair.c_pair}, "
              f"{n/1e6:.2f}M params, zero-init output (exact no-op at step 0)", flush=True)

    def forward(self, input_feats):
        # Surface the pair stack's contact logits on the model output so the loss can read
        # them. Cleared immediately after: holding the tensor on the module would pin the
        # whole graph alive past the optimiser step.
        out = super().forward(input_feats)
        _p = getattr(self, "pair", None)          # absent entirely when PAIR_STACK=0
        if _p is None:
            return out
        lg = getattr(_p, "contact_logits", None)
        if lg is not None:
            out["contact_logits"] = lg
            _p.contact_logits = None
        return out

    # `pair.*` rides alongside `adapter.*`: absent from stock checkpoints, so it stays zero-init
    # on a warm start and reloads on --resume. Without this the stack trains and is discarded.
    def state_dict(self, *a, **kw):
        sd = dict(super().state_dict(*a, **kw))
        for k, v in self.pair.state_dict().items():
            sd[f"pair.{k}"] = v
        return sd

    def load_state_dict(self, sd, strict=False):
        pr = {k[len("pair."):]: v for k, v in sd.items() if k.startswith("pair.")}
        rest = {k: v for k, v in sd.items() if not k.startswith("pair.")}
        res = super().load_state_dict(rest, strict=False)
        print(f"[pair-flow] pair stack tensors loaded: {len(pr)}", flush=True)
        audit_pair_load(list(sd), len(pr), res.missing_keys, res.unexpected_keys)
        if pr:
            # strict would reject a pre-contact pair checkpoint the moment CONTACT_LOSS is
            # turned on. Allow exactly the contact head to be fresh, nothing else.
            inc = self.pair.load_state_dict(pr, strict=False)
            bad = [k for k in inc.missing_keys if not k.startswith("contact_head.")]
            if bad or inc.unexpected_keys:
                raise RuntimeError(f"[pair-flow] pair stack load mismatch: missing {bad[:6]}, "
                                   f"unexpected {list(inc.unexpected_keys)[:6]}")
            if inc.missing_keys:
                print("[pair-flow] contact head is FRESH (absent from this checkpoint) -- "
                      "expected only when turning CONTACT_LOSS on over an existing pair run",
                      flush=True)
        return res


class _Proxy:
    ScoreNetwork = PairFlowModel

    def __getattr__(self, k):
        from model import score_network as _sn
        return getattr(_sn, k)


B.T.score_network = _Proxy()

if __name__ == "__main__":
    print(f"[pair-flow] MODE={os.environ['BINDOPS_MODE']} "
          f"PAIR_LAYERS={os.environ.get('PAIR_LAYERS','4')} "
          f"PAIR_C={os.environ.get('PAIR_C','64')}", flush=True)
    B.T.main()
