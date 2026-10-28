"""Build a labelled pose set for the SpeedDock scoring-model experiment.

Fork of symfix_eval/val_dockq.py -- SAME model/head load, SAME flow inference, SAME per-sample
seeds (1000*idx+s), SAME DockQ invocation (capri_peptide, mapping AB:AB) -- with three changes:
  1. the complex PDBs are KEPT (outdir/<name>/native.pdb, pose_0.pdb ..), for offline feature calc
  2. a GPU-utilisation governor gates every forward pass (user rule: our GPU never sustains >70%)
  3. per-target meta.json + a flat labels.csv, and it is resumable (skips targets already done)

Peptide atoms written = N,CA,C,CB,O (atom37 slots 0..4), exactly as val_dockq.py; pdbfixer rebuilds
side chains later in the physics-feature step.

usage (DiffPepBuilder env):
  make_poseset.py <ckpt> <csv> <outdir> [N=8] [steps=5] [--limit L] [--start S] [--gpu G]
"""

# repo root on sys.path so `python bindops/<pkg>/<file>.py` works directly
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
import gc
import os, sys, re, json, time, argparse, subprocess
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('DPB_NUM_TORSIONS', '3')          # v4 ckpt: pre-v5 3-torsion head
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bindops import gpu_budget as gpu_budget

BASE = os.environ.get('DPB_BASE', '/8TBDrive2/DiffPepBuilder')
REPO = os.environ.get('SPEEDDOCK_REPO', '/8TBDrive2/wt_fix3b')
os.environ['BASE_PATH'] = BASE
sys.path.insert(0, BASE); sys.path.insert(0, REPO)
from hydra import initialize_config_dir, compose
from data import se3_diffuser
from model import score_network
from openfold.np import protein
from speeddock.dataset import PlaceDataset
from speeddock.eval import collate, to_dev
from speeddock.eval_flow import interp_frames
from speeddock.confidence import ConfidenceHead

DOCKQ = f'{BASE}/env/bin/DockQ'
_CHUNK = None          # pose-batch size, shrunk on OOM and carried across targets


def infer_flow(model, batch, x0, steps, device):
    taus = np.linspace(0, 1, steps + 1); x = x0.clone()
    sc = torch.zeros_like(batch['sc_ca_t'])
    with torch.no_grad():
        for i in range(steps):
            batch['sc_ca_t'] = sc; batch['rigids_t'] = x
            batch['t'] = torch.full((x0.shape[0],), float(1 - taus[i]), device=device)
            out = model(batch); sc = out['rigids'][..., 4:]
            x = interp_frames(x0, out['rigids'], taus[i + 1])
    return out


def write_complex(atom37, mask, aatype, resid, chain_idx, path):
    order = np.argsort(chain_idx, kind='stable')
    open(path, 'w').write(protein.to_pdb(protein.Protein(
        atom_positions=atom37[order], atom_mask=mask[order], aatype=aatype[order],
        residue_index=resid[order], chain_index=chain_idx[order],
        b_factors=np.zeros_like(mask[order]))))


def dockq(model_pdb, native_pdb):
    r = subprocess.run([DOCKQ, model_pdb, native_pdb, '--capri_peptide', '--short',
                        '--mapping', 'AB:AB'], capture_output=True, text=True)
    m = re.search(r'DockQ[:\s]+([0-9.]+)', r.stdout)
    return float(m.group(1)) if m else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('csv'); ap.add_argument('outdir')
    ap.add_argument('N', nargs='?', type=int, default=8)
    ap.add_argument('steps', nargs='?', type=int, default=5)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()

    # Cap N without touching the caller. See the module docstring note on POSE_N_CAP: the two
    # live arms were launched with `32 20` hardcoded in run_full.sh and that script cannot be
    # safely edited underneath a running bash.
    _capf = os.path.join(os.path.dirname(os.path.abspath(a.outdir)), 'POSE_N_CAP')
    _cap = os.environ.get('POSE_N_CAP') or (
        open(_capf).read().strip() if os.path.exists(_capf) else '')
    if _cap and int(_cap) < a.N:
        print(f'POSE_N_CAP: generating {int(_cap)} poses per target, not {a.N} '
              f'(source: {"env" if os.environ.get("POSE_N_CAP") else _capf})', flush=True)
        a.N = int(_cap)

    device = 'cuda'
    os.makedirs(a.outdir, exist_ok=True)

    with initialize_config_dir(version_base=None, config_dir=f'{BASE}/config'):
        cfg = compose(config_name='base', overrides=[
            f'data.train_csv_path={a.csv}', f'data.val_csv_path={a.csv}',
            'data.num_eval_samples=null', 'data.center_pos_noise=False', 'experiment.use_ddp=False']
            # base.yaml caps data.filtering.max_len at 512 and the loader drops anything longer
            # SILENTLY -- no error, no log line, the target simply never appears. That quietly
            # biases any benchmark against large complexes (it removed the 13 targets over 512
            # residues from the 317-case blind set). Opt in to a higher cap per run; unset, the
            # behaviour is exactly as before.
            + ([f'data.filtering.max_len={os.environ["POSE_MAX_LEN"]}']
               if os.environ.get('POSE_MAX_LEN') else []))
    diffuser = se3_diffuser.SE3Diffuser(cfg.diffuser)
    ck = torch.load(a.ckpt, map_location=device)
    model = score_network.ScoreNetwork(cfg.model, diffuser).to(device)
    model.load_state_dict(ck['model']); model.eval()
    head = ConfidenceHead.from_state_dict(ck['head'], c_s=cfg.model.node_embed_size).to(device)
    head.eval()
    print(f'ckpt {a.ckpt} step {ck.get("step")}  N={a.N} steps={a.steps} gvp_head={head.gvp is not None}',
          flush=True)

    ds = PlaceDataset(data_conf=cfg.data, diffuser=diffuser, is_training=False, flow=False,
                      mask_pep_esm=False)
    throttle = gpu_budget.Throttle(a.gpu, duty=0.82)    # short inference bursts; avg util stays low
    idxs = list(range(len(ds.csv)))[a.start:]
    if a.limit:
        idxs = idxs[:a.limit]

    # POSE_GID_SUBSET: a .json list or newline-separated file of pdb_names to generate. The seed
    # of pose s is 1000*idx+s with idx the row index in THIS csv, so passing a subset CSV would
    # change every seed and quietly make a paired comparison against a full run a comparison of
    # different draws. Filtering idxs leaves the indices -- and therefore the seeds -- alone.
    _sub = os.environ.get('POSE_GID_SUBSET')
    if _sub:
        if _sub.endswith('.json'):
            want = set(json.load(open(_sub)))
        else:
            want = {ln.strip() for ln in open(_sub) if ln.strip()}
        have = {ds.csv.iloc[i]['pdb_name'] for i in idxs}
        keep = [i for i in idxs if ds.csv.iloc[i]['pdb_name'] in want]
        print(f'POSE_GID_SUBSET: keeping {len(keep)} of {len(idxs)} targets listed in {_sub}; '
              f'dataset indices preserved, so seeds match the full run', flush=True)
        missing = sorted(want - have)
        if missing:
            print(f'POSE_GID_SUBSET: {len(missing)} requested gids are NOT in the csv: '
                  f'{missing[:5]}', flush=True)
        print('POSE_GID_SUBSET: idx -> name  ' +
              ', '.join(f'{i}:{ds.csv.iloc[i]["pdb_name"]}' for i in keep[:5]) + ' ...', flush=True)
        idxs = keep

    manifest_path = os.path.join(a.outdir, 'manifest.json')
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else []
    labels_path = os.path.join(a.outdir, 'labels.csv')
    all_rows = []
    if os.path.exists(labels_path):
        all_rows = pd.read_csv(labels_path).to_dict('records')

    for idx in idxs:
        name = ds.csv.iloc[idx]['pdb_name']
        tdir = os.path.join(a.outdir, name)
        if os.path.exists(os.path.join(tdir, 'meta.json')):
            if name not in manifest:
                manifest.append(name)
            print(f'{name:18s} skip (done)', flush=True)
            continue
        os.makedirs(tdir, exist_ok=True)
        try:
            # POSE_SEED_OFFSET lets an oversized target be generated in seed-aligned
            # chunks: offsets 0/8/16/24 with N=8 use exactly the seeds a single N=32
            # run would have used, so merging the chunks reproduces the same pose set
            # on a GPU that cannot hold the full batch. Default 0 = unchanged.
            _off = int(os.environ.get('POSE_SEED_OFFSET', '0'))
            feats = [ds.eval_item(idx, seed=1000 * idx + _off + s)[0] for s in range(a.N)]
        except Exception as e:
            print(f'skip {idx} {name}: {e}', flush=True); continue

        try:
            _process(name, tdir, feats, model, head, a, device, throttle,
                     manifest, manifest_path, labels_path, all_rows)
        except Exception as e:
            import traceback
            print(f'FAIL {name}: {e}', flush=True)
            traceback.print_exc()
            continue
        finally:
            # Release this target's GPU memory before the next one. Without this the
            # caching allocator keeps every target's freed blocks and fragments across
            # differently-shaped targets, so a long split OOMs after ~5 targets
            # (observed: 14.81 GiB held, 115 OOMs in 120 targets) regardless of size --
            # a 369-residue target succeeded while a 185-residue one failed. `finally`
            # so it also runs on the `continue` path above.
            feats = None
            gc.collect()
            torch.cuda.empty_cache()

    print(f'\ndone: {len(manifest)} targets in {a.outdir}', flush=True)


def _process(name, tdir, feats, model, head, a, device, throttle,
             manifest, manifest_path, labels_path, all_rows):
        t0 = time.time()
        # OOM-SAFE CHUNKING. Collating all N poses into one batch OOMs on large targets, and it
        # does so SELECTIVELY: the anchorfix test run lost 113 of 359 targets, every single one a
        # CUDA OOM, leaving the small receptors -- a silent size bias in the wrong direction,
        # since small receptors are also the degenerate cases. The pair arm lost 119 of its first
        # 153 for the same reason (its pair tensor is O(L^2 C)). Poses are independent given their
        # seed and the seed of pose s is fixed at 1000*idx+s, so generating them in chunks gives
        # the SAME pose set as one batch; only the memory peak changes. Halve on OOM down to 1.
        gpu_budget.wait_for_headroom(a.gpu)
        pa_list, score_list, batch = [], [], None
        # sticky: the csv is sorted largest-first, so a chunk size that fitted the previous target
        # fits this one too. Without this every large target re-probes 32 -> 16 -> 8 and pays a
        # dead forward pass for each step down.
        global _CHUNK
        if _CHUNK is None:
            _CHUNK = max(1, int(os.environ.get('POSE_CHUNK', len(feats))))
        cs = min(_CHUNK, len(feats))
        i = 0
        while i < len(feats):
            sub_f = feats[i:i + cs]
            try:
                b_ = to_dev(collate(sub_f), device)
                x0 = b_['rigids_t'].clone()
                out = infer_flow(model, b_, x0, a.steps, device)
                pmf = (b_['ligand_mask'] * b_['res_mask']).float()
                with torch.no_grad():
                    sc_ = head(out['node_embed'].float(), pmf,
                               out['atom37'].float() if head.gvp is not None else None
                               ).cpu().numpy().ravel()
                torch.cuda.synchronize()
            except RuntimeError as e:
                if 'out of memory' not in str(e).lower():
                    raise
                out = None; b_ = None; x0 = None
                gc.collect(); torch.cuda.empty_cache()
                if cs == 1:
                    raise
                cs = max(1, cs // 2)
                _CHUNK = cs
                print(f'{name:18s} OOM -> retry this target at chunk {cs} '
                      f'(kept for the rest of the run)', flush=True)
                continue
            if batch is None:
                # target-level fields are identical across chunks (same receptor, same native),
                # and everything downstream reads them at index 0. Keep them on the CPU so the
                # copy does not sit on the card while the remaining chunks run.
                batch = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in b_.items()}
            for j in range(len(sub_f)):
                pa_list.append(out['atom37'][j].cpu().numpy())
            score_list.extend(np.atleast_1d(sc_).tolist())
            out = None; b_ = None; x0 = None
            gc.collect(); torch.cuda.empty_cache()
            i += cs
        score = np.asarray(score_list, float)
        assert len(pa_list) == len(feats), 'chunking lost poses: %d of %d' % (len(pa_list), len(feats))

        keep = batch['res_mask'][0].bool().cpu().numpy()
        lm = (batch['ligand_mask'][0] > 0.5).cpu().numpy()
        nat37 = batch['atom37_pos'][0].cpu().numpy()
        amask = batch['atom37_mask'][0].cpu().numpy()
        aatype = batch['aatype'][0].cpu().numpy()
        resid = batch['residue_index'][0].cpu().numpy()
        chain = np.where(lm, 0, 1)                     # 0=A=peptide, 1=B=receptor
        pkeep = lm & keep
        gtca = nat37[pkeep][:, 1, :]

        write_complex(nat37[keep], amask[keep], aatype[keep], resid[keep], chain[keep],
                      os.path.join(tdir, 'native.pdb'))
        poses = []
        for b in range(a.N):
            pa = pa_list[b]
            m37 = nat37.copy(); m37[lm] = pa[lm]
            pmask = amask.copy(); pmask[lm, 5:] = 0            # peptide = N,CA,C,CB,O
            mp = os.path.join(tdir, f'pose_{b}.pdb')
            write_complex(m37[keep], pmask[keep], aatype[keep], resid[keep], chain[keep], mp)
            pca = pa[pkeep][:, 1, :]
            fwd = float(np.sqrt(((pca - gtca) ** 2).sum(-1).mean()))
            rev = float(np.sqrt(((pca - gtca[::-1]) ** 2).sum(-1).mean()))
            dq = dockq(mp, os.path.join(tdir, 'native.pdb'))
            row = dict(name=name, pose=b, dockq=dq, conf_score=float(score[b]),
                       n_pep=int(pkeep.sum()), n_rec=int((keep & ~lm).sum()),
                       fwd_rmsd=fwd, rev_rmsd=rev, reversed=bool(fwd - rev > 0.01))
            poses.append(row); all_rows.append(row)

        dqs = np.array([p['dockq'] for p in poses], float)
        allnan = bool(np.all(np.isnan(dqs)))
        meta = dict(name=name, n=a.N,
                    oracle_idx=0 if allnan else int(np.nanargmax(dqs)),
                    head_sel_idx=int(np.argmin(score)),
                    all_dockq_nan=allnan,
                    poses=poses)
        json.dump(meta, open(os.path.join(tdir, 'meta.json'), 'w'), indent=1)
        if name not in manifest:
            manifest.append(name)
        json.dump(manifest, open(manifest_path, 'w'))
        pd.DataFrame(all_rows).to_csv(labels_path, index=False)
        tag = 'ALL-NaN DockQ' if allnan else (f'best={np.nanmax(dqs):.3f} '
              f'conf-sel={dqs[meta["head_sel_idx"]]:.3f} median={np.nanmedian(dqs):.3f}')
        print(f'{name:18s} {tag}  {time.time()-t0:.1f}s', flush=True)
        throttle.tick()


if __name__ == '__main__':
    main()
