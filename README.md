<p align="center">
  <img width="1280" height="389" alt="Thumbnail_Size_Guide_YouTube_Thumbnail_in_Blue_White_Neo_Digital_Style-removebg" src="https://github.com/user-attachments/assets/25732e73-b058-452f-b25c-5ad5cbb82051" />
</p>
<p align="center">
  The code depends on DiffPepBuilder / DiffPepDock and a `speeddock` package (SpeedDock is not public). Neither is vendored here. You also need PyTorch, OpenFold, DockQ and ESM.
  
>**This repo is for documentation and is not runnable as-is.** `run_full.sh` and several modules hardcode my machine paths (`/8TBDrive1/jithesh/docking_model`, `/8TBDrive2/DiffPepBuilder`, `~/bin/capped_run.sh`, `~/bin/gpu_lease.sh`). A runnable public release is   planned once the results justify it.


BINDops is a research codebase for peptide-protein docking. It adds receptor-side interface and contact information to a pretrained flow-matching peptide docking model, and it contains the gates, losses, and ablations used to test whether that information, and several architectural changes, actually improve where the peptide lands.

BINDops does not compute or search for MSAs. The msa and all arms in scripts/run_full.sh consume precomputed UniRef90 alignments, reduced to 23 per-residue profile features (PSSM log-odds, entropy, gap fraction, depth) that feed the adapter. There is no MSA module, and the base model needs no alignment. 

## What's in here

| Package | Purpose |
|---|---|
| `bindops/model/` | Conditioned score network, triangle pair stack, global placement head, unified 1D+2D interface model, distillation of the v4 pocket model |
| `bindops/losses/` | Interface FAPE, bond-length loss, inter-chain contact loss, placement loss |
| `bindops/features/` | MSA profiles (UniRef90), site scores, contact-map cache and gate, pocket building, start-site centres, feature coverage preflight |
| `bindops/training/` | Flow trainers (base, v5 with validation, pair stack, placement), checkpoint soup, held-out validation |
| `bindops/gates/` | Pre-GPU checks: reload, wiring, pair stack, placement |
| `bindops/inference/` | Pose generation, labelled pose sets, DockQ rescoring |
| `bindops/gpu_budget.py` | Keeps the GPU you touch under a utilization ceiling (default 68%) |
| `scripts/run_full.sh` | End-to-end pipeline for the two main arms |

## Method

**Adapter.** We feed a per-residue interface probability into the frozen DiffPepDock score network through a small MLP. The last layer starts at zero, so at step 0 the model gives exactly the same output as the one without conditioning. `verify_flow_wiring.py` checks that instead of trusting it.

**Pair stack.** DiffPepBuilder updates edge (i,j) from its two endpoint nodes only, so it never sees edges (i,k) and (k,j). We added a small triangle-multiplicative stack before the IPA trunk (no triangle attention, gradient checkpointed to fit on a P100). Its output projection also starts at zero so the warm start isn't lost. The first run with it did nothing, because nothing in the loss used pair information. `contact_loss.py` fixes that with a contact target between peptide and receptor residues (pseudo-CB under 8 Å).

**What went wrong with fine-tuning.** Our fine-tuned weights squeezed the peptide: consecutive Cα–Cα distance dropped to 2.52 Å when real peptides sit at 3.81 Å. FAPE was only 0.129% of the training loss, and backbone L2 made up almost all of the rest. We tried an interface FAPE (receptor frames, peptide points) because the stock FAPE can't tell where the peptide sits on the receptor. We also tried a bond-length term. It fixed the bond geometry and made docking worse, so a better-shaped peptide still ended up in the wrong place.

**Placement head (experimental).** The score network is good at refining a pose and bad at moving a whole peptide to another patch. The head points at a receptor Cα and adds a radial offset, so it stays equivariant. `place_loss.py` trains it with a pointer cross-entropy plus a Huber centroid term. It's built and gated but we haven't tested it end to end, and it isn't in the poster results.

**Features.** The site score is a real one now, where earlier arms had all zeros. `cmap_gate.py` checks whether ESM alone can predict the contact map before we try docking. Some arms also use UniRef90 profiles (log-odds, entropy, gap fraction, depth) that were computed outside this repo. BINDops doesn't build MSAs.

## Pipeline

`scripts/run_full.sh` runs two arms:
- `msa`: v5 trainer with a validation loop and best-checkpoint selection, real MSA features, real site score, adapter unfrozen (`BINDOPS_MODE=flexible`).
- `all`: the `msa` arm plus the triangle pair stack and recycling (`PAIR_C=64 PAIR_C_HIDDEN=32 PAIR_LAYERS=4 PAIR_STACK=1`).

Shared settings: 6000 steps, batch 64, lr 1e-4, `--max-sq-res 150000`, seed 0, validation every 1000 steps, `best.pth` chosen on held-out loss, `BLIND_PRIOR=surface`.

Guards:

1. Preflight feature coverage (`--min-cov 0.97` over train, val and test splits) before any training starts.
2. A check that the log contains "native peptide pose never read", so a native-derived start pose cannot leak.
3. A reload gate before generation, so a silent `strict=False` reload failure cannot void pretrained weights.
4. Standard DockQ v2 rescoring (`rescore_dockq.py`), with unscoreable poses counted as 0, never dropped.

Most variables are environment, for example:

```bash
W_CONTACT=1.0 CONTACT_CUTOFF=8.0     # contact loss
W_IFACE=1.0 IFACE_CLAMP=30.0         # interface FAPE
GPU_UTIL_CEIL=68                      # gpu_budget ceiling
BINDOPS_MODE=flexible                 # unfreeze the adapter
```

## Running the gates

Run these before spending GPU time:

```bash
python bindops/gates/verify_flow_wiring.py
python bindops/gates/verify_pair_stack.py
python bindops/gates/verify_place.py
python bindops/gates/gate_reload.py <checkpoint>
```

Generate and score poses:

```bash
PAIR_STACK=1 python bindops/inference/gen_poses_pair.py <ckpt> <csv> <outdir> [N] [steps] [--gpu G]
python bindops/inference/rescore_dockq.py <poseset_dir> [nproc]
```
## Results

Blind evaluation with DockQ on 304 unseen complexes released after 2025. The best model is a weight soup, : **38.8% acceptable (oracle), 9.2% top-1**. RAPiDock is still the top-performing model on this test set, and 13 large targets are missing from every BINDops arm.

- Starting the peptide at the true binding site only raised per-pose success within 10 Å from 0.8% to 2.9%. The sampler often moves the peptide away, so pose generation looks like the bottleneck, not placement.
- The `msa` and `all` arms in `run_full.sh` are experimental and are not the soup. The recorded `all` arm outcome in the code comments is oracle DockQ 0.128 (26.2% acceptable), top-1 DockQ 0.035 (3.1%).
- Interface predictors, v4 included, drop to about 0.54-0.61 AUROC on real peptide sites (n=51), and inverse coordination number alone ties the 185-feature v4 stack.
- Evaluation sets are small. Treat single-target numbers as anecdotal.

## Status

- The recorded outcome for the `all` arm in the code comments is oracle DockQ 0.128 (acceptable 26.2%) and top-1 DockQ 0.035 (3.1%). This is not a state-of-the-art result.
- Interface predictors, including v4, drop to roughly 0.54-0.61 AUROC on real peptide sites (n=51), and inverse coordination number alone ties the 185-feature v4 stack. 
- Evaluation sets are small. Treat any single-target or small-n number as anecdotal, and report confidence intervals.

## Credits

Built on DiffPepBuilder / DiffPepDock and RAPiDock ideas, and on the SpeedDock (not public) flow-matching training code.

Thanks to Yash Bhargava, Bhargava Systems Research inc., for constant support and mentorship with the project and its direction.

## License

MIT. Copyright (c) 2026 Jithesh Mithra.
