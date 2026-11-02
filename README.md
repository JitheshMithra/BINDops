<p align="center">
  <img width="1280" height="389" alt="Thumbnail_Size_Guide_YouTube_Thumbnail_in_Blue_White_Neo_Digital_Style-removebg" src="https://github.com/user-attachments/assets/25732e73-b058-452f-b25c-5ad5cbb82051" />
</p>

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

## Method

**Conditioned adapter.** A small MLP maps a per-residue receptor interface probability (from ESM2-650M) into the node-embedding space of the pretrained score network. The last layer is zero-initialised, so at step 0 the conditioned model is bit-for-bit the unconditioned one. `verify_flow_wiring.py` checks this.

**Pair stack.** DiffPepBuilder's edge representation cannot see edge (i,k) and edge (k,j) when updating edge (i,j). A low-rank triangle-multiplicative stack (AlphaFold2 style, no triangle attention, gradient checkpointed) is inserted before the IPA trunk with a zero-initialised output projection, so the warm start survives. `contact_loss.py` gives it a pair-shaped BCE target (peptide residue i vs receptor residue j, pseudo-CB < 8 Å), because the first pair-stack arm was null: nothing read it.

**Loss experiments.** Fine-tuning compressed consecutive Cα–Cα spacing to 2.52 Å against 3.81 Å in native peptides, and FAPE contributed only 0.129% of training loss, with backbone L2 dominating. Interface FAPE (receptor frames, peptide points) was added because the stock peptide-only FAPE cannot see where the peptide sits. A rotation- and translation-invariant bond term was added to restore chain geometry. It improved bond geometry but made docking worse, so better shape did not fix location.

**Placement head (experimental, not in reported results).** Per-residue updates refine a pose well but move a whole peptide poorly. The head predicts a pointer over receptor Cα atoms plus a radial offset, and is equivariant by construction. `place_loss.py` combines a pointer cross-entropy with a Huber centroid term. It is implemented and gated, but untested end to end.

**Features.** A real site score replaces the literal zeros that earlier arms carried, and a contact-map gate tests whether the contact map is predictable from ESM before any docking is attempted. Optional arms also use precomputed UniRef90 profiles (log-odds, entropy, gap fraction, depth). BINDops does not compute MSAs itself.

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

## Setup

The code depends on the DiffPepBuilder / DiffPepDock codebase and a `speeddock` package (speedock doesn't exist publically), neither of which is vendored here, plus PyTorch, OpenFold, DockQ, ESM, and Python stack.

**This repo is not portable/fully public yet. Simply for documentation.** `run_full.sh` and several modules hardcode my machine paths (`/8TBDrive1/jithesh/docking_model`, `/8TBDrive2/DiffPepBuilder`, `~/bin/capped_run.sh`, `~/bin/gpu_lease.sh`). It will be fully public with access and support for all parts of this repo so anyone can run it once we achieve a considerable "win".

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

## Status

- The recorded outcome for the `all` arm in the code comments is oracle DockQ 0.128 (acceptable 26.2%) and top-1 DockQ 0.035 (3.1%). This is not a state-of-the-art result.
- The first pair-stack arm came back null, and the earlier attempts to fix placement by adding coordinate terms are documented as null. The placement head is the newest attempt and should be treated as unproven until its results are in.
- Fine-tuning bought pose diversity at the cost of per-pose quality. The checkpoint soup (`make_soup.py`) interpolates between the pretrained and fine-tuned weights to study that trade.
- Interface predictors, including v4, drop to roughly 0.54-0.61 AUROC on real peptide sites (n=51), and inverse coordination number alone ties the 185-feature v4 stack. Conditioning on a better site score is not a given win.
- Evaluation sets are small. Treat any single-target or small-n number as anecdotal, and report confidence intervals.

## Credits

Built on DiffPepBuilder / DiffPepDock and RAPiDock ideas, and on the SpeedDock (not public) flow-matching training code.

Thanks to Yash Bhargava, Johns Hopkins University, for constant support and mentorship with the project and its direction.

## License

MIT. Copyright (c) 2026 Jithesh Mithra.
