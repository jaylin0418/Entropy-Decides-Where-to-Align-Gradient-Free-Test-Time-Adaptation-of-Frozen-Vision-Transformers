> Historical document, committed before the held-out images were used. Script and file names refer to the original research codebase; in this repository the corresponding entry points are `scripts/episodic/run_forward_only.py` and `scripts/streams/run_streams.py` (see README).

# Pre-registration — SPA held-out test on the second backbone, ViT-S/16 augreg (ImageNet-C)

Declared 2026-10-21 (time of the commit), before any image of the slice was decoded or passed through a model.
Governing log: `docs/spa_decision_log.md` (E3 on design data: PASS). Everything is frozen as in
`docs/spa_preregistration.md`; only the model and the slice differ. The held-out images are scored once.

## Claims under test

- **C1-S.** The gated forward-only stack (tokenizer CORAL + NEO + source-prior head covariance, drift gate) gains ≥ +2.0 pt
  over NEO on the 30 episodic cells and ≥ +2.0 over NEO-EMA on the continual set, with clean harm ≤ 0.3 on the clean
  stream and on the clean segments of the clean-return stream.
- **C2-S.** The tokenizer stage adds ≥ +1.0 pt to ETA (+ NEO) on the episodic cells and on the continual set, with clean
  harm ≤ 0.5.

## Frozen settings

timm `vit_small_patch16_224` with the official augreg in21k→in1k weights (state dict sha256 `f23048cacd0faf77`,
`manifests/models.json`), FP32, TF32 off, mean/std 0.5, clean resize 248 / crop 224, batch 64. Source statistics from
the same 3,000 clean design images (`data/derived/spa/source_stats_vits.pt`, embed dim 384). Tokenizer stage, head
stages, gate thresholds (0.4 / 0.3), EMA rates (0.2 / 0.5 / 0.1), shrinkage (0.1, prior 4 images / 256 samples), ETA
configuration: identical to the ViT-B pre-registration. Design-data values (E3): episodic +4.12, continual +3.25,
tokenizer on ETA +2.05 / +1.85.

## Data

Held-out images `perm[38000:40000]` (split seed 20261019), never decoded or passed through any model in this project.
Cells: clean (resize 248) and the 15 standard corruptions at severities 3 and 5; caches `spaheld2_*` built by
`scripts/spa/build_held_caches.py --lo 38000` only after this file is committed (sha256 in `manifests/spa_caches.json`).
Streams as in SPA4.

## Runs and scoring

`scripts/py.sh scripts/spa/probe_online.py --model vits --set held --held-lo 38000 --arms neo tok headcov tok+headcov`;
`… probe_eta.py --model vits --set held --held-lo 38000`; `… probe_streams.py --model vits --set held --held-lo 38000`
(≈ 1 GPU-h). Scoring once: `score_probe.py`, `score_eta.py`, `score_streams.py` with `--model vits --set held` →
`results/spa/spa{1,2,4}_held_vits.{json,md}`.

## Verdict rule

PASS if C1-S and C2-S both hold; PARTIAL if exactly one; FAIL otherwise. Thresholds fixed; no re-scoring; no further
arms on held-out data. Disclosure: no held-out image of this slice was used for any choice; the ViT-S pipeline was run
once on design data (E3) with the settings frozen from ViT-B.
