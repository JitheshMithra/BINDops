"""SpeedDock v5 training = v4 + a real validation loop + recycling.

v4 (unchanged, still used by the stage1 arms) had two holes this closes:

1. NO VALIDATION. v4 passes data.val_csv_path into the hydra config and never reads it --
   PdbDataset only opens that path when is_training=False and v4 builds the training set only.
   So there was no held-out signal, checkpoints were written on step count alone, and the
   pipeline shipped whichever was LAST. Measured consequence on the 2026-10-11 arm: step_10000
   was shipped and was the WORST of its five checkpoints (val 241.5 vs 208.7 at step 8000).
   v5 evaluates a held-out set every --val-every steps through val_core (the same code the
   out-of-band validate_ckpt.py uses, so the two cannot drift), writes the curve to
   <out_dir>/val_curve.jsonl, and keeps <out_dir>/best.pth at the lowest val loss.

2. NO RECYCLING. v4 has self-conditioning -- one no-grad forward pass whose predicted CA
   coordinates re-enter the next pass as a distogram -- but fires it on a 50/50 coin flip and
   never iterates. Recycling is that same mechanism run R times, which is how AF2 lets a model
   refine its own answer. --recycle R samples r ~ U{0..R} per step (AF2's schedule) and runs r
   no-grad passes before the scored one. It adds NO parameters, so the 240k-step DiffPepDock
   warm start stays exact, and --recycle 0 reproduces v4's coin flip byte for byte.

usage: train_flow_v5.py <out_dir> <data_dir> [--max-steps N] [--batch N] [--lr F] [--max-sq-res F]
                        [--resume CKPT] [--val-csv CSV] [--val-every N] [--val-reps R]
                        [--recycle R]
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import argparse, os, random, sys, time, logging
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch
import torch.nn.functional as F

# ponytail: env override so the same file runs on yash (/12TBDrive1) and a1 (/8TBDrive2)
BASE = os.environ.get('DPB_BASE', '/12TBDrive1/DiffPepBuilder')
REPO = os.environ.get('SPEEDDOCK_REPO', '/12TBDrive1/speeddock')
os.environ['BASE_PATH'] = BASE
sys.path.insert(0, BASE); sys.path.insert(0, REPO)
from bindops.gpu_budget import Throttle  # stage1 patch: keep sustained GPU util <=ceiling during training
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from data import se3_diffuser, utils as du, pdb_data_loader
from model import score_network
from openfold.utils.rigid_utils import Rigid
from speeddock.dataset import PlaceDataset
from speeddock.train import to_dev
from speeddock.losses import flow_loss
if os.environ.get("IFACE_FAPE") == "1":
    # speeddock.losses lives on a read-only mount, so the docking-aware loss is a local
    # module swapped in here. See iface_loss.py: the stock FAPE takes its frames AND its
    # points from the peptide alone, which makes it invariant to where the peptide is
    # docked (measured). Off by default -- every earlier arm reproduces exactly.
    from bindops.losses.iface_loss import flow_loss  # noqa: F811
    print("[loss] IFACE_FAPE=1: receptor-frame interface FAPE ON", flush=True)
if os.environ.get("CONTACT_LOSS") == "1":
    # The pair stack had no loss term reading it at all, which is why that arm was null.
    # Wraps whichever flow_loss was selected above, so it composes with IFACE_FAPE.
    from bindops.losses.contact_loss import wrap_flow_loss, W_CONTACT as _WC
    flow_loss = wrap_flow_loss(flow_loss)
    print(f"[loss] CONTACT_LOSS=1: inter-chain contact BCE on the pair stack, "
          f"W_CONTACT={_WC}", flush=True)
if os.environ.get("PLACE_LOSS") == "1":
    # The global placement head needs a placement-shaped objective. The pointer term is a
    # classification over receptor residues, which is well conditioned at ANY error magnitude --
    # unlike every previous placement fix, which was a coordinate regression and so inherited
    # either a quadratic that explodes at 40 A or a clamp with zero gradient past its cutoff.
    from bindops.losses.place_loss import wrap_flow_loss as _wrap_place, W_POINTER as _WP, W_CENTROID as _WCN
    flow_loss = _wrap_place(flow_loss)
    print(f"[loss] PLACE_LOSS=1: pointer CE + centroid smooth-L1 on the global placement head, "
          f"W_POINTER={_WP} W_CENTROID={_WCN}", flush=True)
if os.environ.get("BOND_LOSS") == "1":
    # MEASURED DEFECT (analysis_ablation/grow.py): fine-tuning COMPRESSES the peptide chain --
    # consecutive CA-CA 2.55 A against the 3.81 A peptide bond, from a DiffPepDock warm start that
    # had it right at 3.99 A. MEASURED CAUSE (runs/full_all/val_curve.jsonl): FAPE, the only shipped
    # term that constrains internal geometry, is 0.103% of the loss -- val_fape 0.397 against
    # 0.5*val_bb 386.2 -- because the backbone term is an unnormalised mean SQUARE. Under
    # directional uncertainty a squared point loss is minimised by contracting the chain.
    # bond_loss adds a rotation-invariant bond term (the only form that prefers the correct scale,
    # lossfix.py gate 5) and reshapes the backbone term to a Huber on d/10 A (gates 1-2: geometry's
    # share of the gradient 0.103% -> 25.8%, and a bounded but never-zero gradient at any error).
    # Wraps whichever flow_loss was selected above, so it composes with the other three.
    from bindops.losses.bond_loss import wrap_flow_loss as _wrap_bond, W_BOND as _WBD, RESHAPE_BB as _RS
    flow_loss = _wrap_bond(flow_loss)
    print(f"[loss] BOND_LOSS=1: rotation-invariant CA-CA bond term, W_BOND={_WBD}, "
          f"backbone reshaped to Huber={_RS}", flush=True)
from speeddock.confidence import ConfidenceHead, true_lrmsd
from bindops.training import val_core as val_core
import json

WARM = f'{BASE}/experiments/checkpoints/diffpepdock_v1.pth'
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('speeddock')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('out_dir')
    p.add_argument('data_dir')
    p.add_argument('--max-steps', type=int, default=12000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max-sq-res', type=float, default=150000)
    p.add_argument('--resume', default=None,
                   help='checkpoint to start from (model+head). Default: the diffpepdock warm start. '
                        'Optimizer state is NOT stored in our checkpoints, so Adam moments restart.')
    p.add_argument('--seed', type=int, default=0,
                   help='seeds the data order (TrainSampler time_batch shuffles with the global '
                        'random module), the per-worker placement/tau RNG, and the '
                        'self-conditioning coin flip, so two runs differ only by the flags under '
                        'test. GPU kernel nondeterminism is not suppressed.')
    p.add_argument('--val-csv', default=None,
                   help='held-out CSV to validate on every --val-every steps. The default val set '
                        'is verified_v2/data/metadata_val_real_A.csv (86 sequence-verified '
                        'protein-peptide complexes, one per receptor cluster, mmseqs-checked '
                        'cluster-disjoint from train AND test).')
    p.add_argument('--val-every', type=int, default=0,
                   help='0 = validate at every checkpoint (CKPT_EVERY). Otherwise every N steps.')
    p.add_argument('--val-reps', type=int, default=4,
                   help='fixed-seed repeats per val example; 4 x 86 = 344 evals per curve point')
    p.add_argument('--recycle', type=int, default=0,
                   help='max recycle passes; r ~ U{0..R} per step. 0 = v4 self-conditioning coin flip')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dry-run', type=int, default=0,
                   help='run N steps and exit without writing checkpoints (pipeline sanity check)')
    return p.parse_args(argv)


# Checkpoint interval. Default 2000 keeps every existing caller byte-identical in behaviour.
# Set BINDOPS_CKPT_EVERY to shorten it: a 2026-10-12 host reboot killed a resumed run at
# step 1800 and the 2000-step interval meant NOTHING had been written -- 3.4 GPU-hours lost.
CKPT_EVERY = int(os.environ.get('BINDOPS_CKPT_EVERY', '2000'))


def main(args=None):
    args = args or parse_args()
    out_dir, data_dir = args.out_dir, args.data_dir
    max_steps = args.dry_run or args.max_steps
    batch_size, lr, max_sq_res = args.batch, args.lr, args.max_sq_res
    os.makedirs(out_dir, exist_ok=True)
    random.seed(args.seed)               # TrainSampler(time_batch) calls random.shuffle
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)         # self-conditioning coin flip + any re-init'd head weights
    torch.cuda.manual_seed_all(args.seed)
    device = args.device
    # stage1 patch: physical GPU index for nvidia-smi querying is CUDA_VISIBLE_DEVICES, not the
    # torch-local device index (which is always 0 once CUDA_VISIBLE_DEVICES restricts visibility).
    # v4 parsed CUDA_VISIBLE_DEVICES unconditionally, which crashes when it is empty (a CPU run)
    # and, worse, would duty-cycle against whatever GPU 0 happens to be doing. No GPU, no throttle.
    if device == 'cuda':
        _phys_gpu = int((os.environ.get('CUDA_VISIBLE_DEVICES') or '0').split(',')[0])
        throttle = Throttle(gpu=_phys_gpu, duty=0.55)
    else:
        class _NoThrottle:
            def tick(self):
                pass
        throttle = _NoThrottle()
    with initialize_config_dir(version_base=None, config_dir=f'{BASE}/config'):
        conf = compose(config_name='base', overrides=[
            f'data.train_csv_path={data_dir}/metadata_train.csv',
            f'data.val_csv_path={data_dir}/metadata_val.csv',
            'experiment.use_ddp=False', f'experiment.batch_size={batch_size}',
            'experiment.num_loader_workers=3', f'experiment.max_squared_res={max_sq_res}'])

    diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)
    model = score_network.ScoreNetwork(conf.model, diffuser).to(device)
    src = args.resume or WARM
    blob = du.read_pkl(src, use_torch=True, map_location=device)
    ck = {k.replace('module.', ''): v for k, v in blob['model'].items()}
    msd = model.state_dict()   # drop shape-mismatched keys (v5 torsion head grew 3->5 torsions)
    dropped = [k for k, v in ck.items() if k not in msd or v.shape != msd[k].shape]
    if dropped and args.resume:
        # The warm start is allowed to mismatch; an explicit --resume is not. score_network.py
        # defaults NUM_TORSIONS to 5, but every runs/flow_v4 checkpoint has the pre-v5 3-torsion
        # head, so without DPB_NUM_TORSIONS=3 the torsion layer is silently reinitialised and the
        # fine-tune starts from a broken sidechain/atom37 path.
        raise SystemExit(
            f'--resume {src}: {len(dropped)} tensor(s) do not fit this model and would be '
            f'reinitialised: {dropped[:3]}. If these are torsion_pred.*, set DPB_NUM_TORSIONS=3.')
    ck = {k: v for k, v in ck.items() if k in msd and v.shape == msd[k].shape}
    model.load_state_dict(ck, strict=False)
    head = ConfidenceHead(c_s=conf.model.node_embed_size).to(device)
    if 'head' in blob:                       # a v4 checkpoint carries a trained head; keep it
        head.load_state_dict(blob['head'])
    log.info(f'init from {src} (step {blob.get("step", "-")}, head={"head" in blob})')
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    log.info(f'v4 | flow+self_cond+FAPE+conf-head+unmask-ESM | batch={batch_size} '
             f'max_sq_res={max_sq_res:.0e} data={os.path.basename(data_dir)} seed={args.seed}')

    ds = PlaceDataset(data_conf=conf.data, diffuser=diffuser, is_training=True,
                      flow=True, mask_pep_esm=False, seed=args.seed)
    sampler = pdb_data_loader.TrainSampler(data_conf=conf.data, dataset=ds, batch_size=batch_size,
                                           sample_mode=conf.experiment.sample_mode)
    loader = du.create_data_loader(ds, sampler=sampler, np_collate=False, length_batch=True,
                                   batch_size=batch_size, shuffle=False, num_workers=3,
                                   drop_last=False, max_squared_res=max_sq_res)

    # Re-seed after all module construction. A differently shaped head (--conf-vec-feats builds a
    # wider LayerNorm/Linear plus a GVP layer) consumes a different number of draws from the global
    # RNG, which would otherwise shift every self-conditioning coin flip and reintroduce exactly the
    # confound --seed exists to remove. The dataset stream is unaffected: it is seeded per worker
    # from args.seed, independently of torch.
    torch.manual_seed(args.seed)

    # Held-out set, built once. A second PlaceDataset over the val CSV with is_training=False,
    # so eval_item's per-example seed drives every draw and the curve is comparable across steps.
    val_ds = None
    if args.val_csv:
        val_ds = val_core.build_val_dataset(conf, diffuser, args.val_csv)
        log.info(f'validation: {len(val_ds.csv)} examples from {os.path.basename(args.val_csv)}, '
                 f'reps={args.val_reps}, recycle={args.recycle}')
    val_every = args.val_every or CKPT_EVERY
    val_path = os.path.join(out_dir, 'val_curve.jsonl')
    best = {'val_loss': float('inf'), 'step': -1}

    def run_val(step):
        """Validate, append to the curve, and keep best.pth at the lowest val loss."""
        res = val_core.evaluate(model, val_ds, reps=args.val_reps, device=device,
                                recycle=args.recycle)
        res['step'] = step
        log.info(f'VAL step {step} loss={res["val_loss"]:.3f} +-{res.get("val_loss_se", 0):.3f} '
                 f'fape={res["val_fape"]:.4f} n={res["n_evals"]} '
                 f'nonfinite={res.get("nonfinite", 0)}')
        with open(val_path, 'a') as fh:
            fh.write(json.dumps(res) + chr(10))
        if res['val_loss'] < best['val_loss']:
            best.update(val_loss=res['val_loss'], step=step)
            torch.save({'model': model.state_dict(), 'head': head.state_dict(), 'step': step,
                        'resumed_from': src, 'val_loss': res['val_loss'],
                        'conf': OmegaConf.to_container(conf, resolve=False)},
                       os.path.join(out_dir, 'best.pth'))
            log.info(f'  new best val loss -> best.pth (step {step})')
        else:
            log.info(f'  best remains step {best["step"]} (loss {best["val_loss"]:.3f})')

    model.train(); head.train()
    step, t0, skips, seen, ema = 0, time.time(), 0, 0, None
    while step < max_steps:
        for batch in loader:
            batch = to_dev(batch, device)
            bsz = batch['aatype'].shape[0]
            opt.zero_grad()
            # Recycling. r no-grad passes, each feeding its predicted CA back in as the
            # self-conditioning distogram, then the scored pass. r ~ U{0..R} per step is AF2's
            # training schedule: it teaches the model to be useful at every recycle depth, so
            # inference can spend more passes than training did. R=0 keeps v4 exactly.
            if args.recycle > 0:
                nrec = int(torch.randint(0, args.recycle + 1, (1,)).item())
                if nrec:
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                        for _ in range(nrec):
                            batch['sc_ca_t'] = model(batch)['rigids'][..., 4:].float()
            elif torch.rand(1).item() > 0.5:                        # v4 self-conditioning
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                    batch['sc_ca_t'] = model(batch)['rigids'][..., 4:].float()
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                out = model(batch)
                loss_g, parts = flow_loss(out, batch)
                pm = (batch['ligand_mask'] * batch['res_mask']).float()
                pred_ca = Rigid.from_tensor_7(out['rigids']).get_trans()
                gt_ca = Rigid.from_tensor_7(batch['rigids_0']).get_trans()
                tgt = true_lrmsd(pred_ca.float(), gt_ca.float(), pm)
                conf_pred = head(out['node_embed'].detach().float(), pm)  # readout: detach model
                l_conf = F.smooth_l1_loss(conf_pred, tgt)
                loss = loss_g + 0.1 * l_conf
            if not torch.isfinite(loss):
                opt.zero_grad(); skips += 1
                if skips <= 5:
                    log.info(f'[skip nonfinite step {step}]')
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            # Clip the generator and the head separately. A joint norm makes the clip coefficient
            # depend on the head's gradient magnitude and applies it to the generator too, so the
            # head's architecture leaks into the generator's update even though its input is
            # detached -- which would leave --conf-vec-feats runs incomparable to plain ones.
            # This does change v4's dynamics slightly: the generator's effective clip is no longer
            # diluted by the head's contribution to the joint norm.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            throttle.tick()  # stage1 patch: duty-cycle this step to keep sustained util <=ceiling
            step += 1; seen += bsz
            ema = loss_g.item() if ema is None else 0.98 * ema + 0.02 * loss_g.item()
            if step % 25 == 0:
                el = time.time() - t0
                gb = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0.0
                log.info(f'step {step} bsz={bsz} loss={loss_g.item():.2f} ema={ema:.2f} '
                         f'fape={parts["fape"]:.3f} '
                         # placement term; absent for the pre-2026-10-16 blind loss
                         + (f'iface={parts["iface"]:.3f} ' if 'iface' in parts else '')
                         # contact BCE on the pair stack, and the pos-minus-neg probability gap
                         # that says whether the head is learning rather than just shrinking
                         + (f'cmap={parts["contact"]:.3f} sep={parts["c_sep"]:+.3f} '
                            if 'contact' in parts else '')
                         # placement head: ptr_d/ptr_hit are the localisation metric itself (same
                         # 10 A convention as the frozen-feature gate run), and `gate` is the only
                         # thing that distinguishes "the head never engaged" from "the head
                         # engaged and did not help" -- the two outcomes the pair stack conflated.
                         + (f'ptrd={parts["ptr_d"]:.1f} hit={parts["ptr_hit"]:.2f} '
                            f'ce={parts["ptr_ce"]:.2f} cent={parts["cent_d"]:.1f} '
                            f'gate={parts["gate"]:.3f} rad={parts["radial"]:+.1f} '
                            if 'ptr_d' in parts else '')
                         + (f'bond={parts["bond"]:.4f} caca={parts["ca_ca"]:.2f} '
                            if 'bond' in parts else '')
                         + (f'bbsq={parts["bb_sq"]:.1f} ' if 'bb_sq' in parts else '') +
                         f'bb={parts["bb"]:.3f} conf={l_conf.item():.3f} '
                         f'{seen/el:.1f} cx/s gpu={gb:.1f}G skips={skips}')
            if args.dry_run:
                pass
            elif step % CKPT_EVERY == 0 or step >= max_steps:
                p = os.path.join(out_dir, f'step_{step}.pth')
                torch.save({'model': model.state_dict(), 'head': head.state_dict(), 'step': step,
                            'resumed_from': src,
                            'conf': OmegaConf.to_container(conf, resolve=False)}, p)
                log.info(f'saved {p}')
                if val_ds is not None and step % val_every == 0:
                    run_val(step)
            if step >= max_steps:
                break
    log.info(f'DONE {step} steps, {seen} complexes in {(time.time()-t0)/60:.1f} min, '
             f'{seen/(time.time()-t0):.1f} cx/s, {skips} skips')
    if val_ds is not None:
        log.info(f'SELECTED by held-out val loss: step {best["step"]} '
                 f'(val_loss {best["val_loss"]:.3f}) -> {out_dir}/best.pth . '
                 f'Use THAT, not the last checkpoint.')


if __name__ == '__main__':
    main()
