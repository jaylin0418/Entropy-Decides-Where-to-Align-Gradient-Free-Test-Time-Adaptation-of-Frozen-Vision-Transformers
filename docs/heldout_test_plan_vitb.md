> Historical document, committed before the held-out images were used. Script and file names refer to the original research codebase; in this repository the corresponding entry points are `scripts/episodic/run_forward_only.py` and `scripts/streams/run_streams.py` (see README).

# Pre-registration — SPA held-out test: closed-form tokenizer alignment and the forward-only stack (ViT-B/16 on ImageNet-C)

Declared 2026-10-20 17:50 PDT, before any image of the held-out slice was decoded or passed through a model. Governing
log: `docs/spa_decision_log.md` (SPA1 dropped; SPA1b, SPA2, SPA3, SPA4 on design data; the held-out outcome is
appended there). Everything below is frozen; the held-out images are scored once.

## Claims under test

- **C1 (forward-only stack).** The gated stack — tokenizer-level patch-space covariance alignment, NEO, source-prior
  head covariance, all closed-form and online, with the model-free drift gate — gains ≥ +2.0 pt over NEO on the 30
  episodic cells and ≥ +2.0 over NEO-EMA on the continual set, with clean harm ≤ 0.3 on the clean stream and on the
  clean segments of the clean-return stream, at ≤ 1.1 × the MACs of a forward.
- **C2 (universal stage).** The tokenizer stage adds ≥ +1.0 pt to ETA (+ NEO) on the episodic cells and on the
  continual set, with clean harm ≤ 0.5.
- **Secondary, reported either way:** the forward-only stack against ETA + NEO (design data: +0.6 episodic, −1.4 on
  continual streams), the per-family table, gate activity, warm-up, the head-covariance contribution.

## Frozen settings (identical to the design runs)

Model timm `vit_base_patch16_224` augreg in21k→in1k (local checkpoint sha `f6b8659a…`), FP32, TF32 off; mean/std 0.5,
clean resize 248 / crop 224; batch 64. Source statistics from 3,000 clean design images (`posdevx_clean_r248` rows
0–2999; `data/derived/spa/source_stats.pt`): patch mean/covariance, head mean/covariance. Tokenizer stage: online patch
mean and covariance (cumulative on episodic cells, EMA 0.2 on streams), prior n₀ = 4 images of patches plus constant
shrinkage 0.1 toward Σ_s, whitening–recolouring x′ = Σ_s^{1/2} Σ̂^{−1/2} (x − m_t) + μ_s. Head: NEO (cumulative /
NEO-EMA 0.5), covariance stage with the source prior (n₀ = 256 samples; cumulative / EMA 0.1), NEO's zero mean.
Gate thresholds 0.4 (spectral drift) and 0.3 (mean shift), off → NEO only. ETA: SAR-repo `eata.py`, fishers = None,
lr 0.001 at batch 64, one instance per episodic cell / per stream; NEO and head covariance post hoc on its recorded
head input. Arms: episodic `neo`, `tok`, `headcov`, `tok+headcov` (gated), `eta`, `tok+eta`; streams `neo`, `stack`,
`stack_gated`, `eta`, `tok+eta_gated`.

## Data

Held-out images `perm[36000:38000]` (split seed 20261019): never decoded or passed through any model in this project
(the slice `perm[34000:36000]` was used by an unscored FlexiViT dump and is not reused). Cells: clean (resize 248) and
the 15 standard corruptions at severities 3 and 5; caches `spaheld_*` built by `scripts/spa/build_held_caches.py` only
after this file is committed (sha256 in `manifests/spa_caches.json`). Streams built exactly as SPA4 (cont5, cont3,
alt, cleanret seed 0, clean) from these cells.

## Runs and scoring

`scripts/py.sh scripts/spa/probe_online.py --set held --arms neo tok headcov tok+headcov`;
`… probe_eta.py --set held`; `… probe_streams.py --set held` (≈ 2.5 GPU-h). Scoring once: `score_probe.py --set held`,
`score_eta.py --set held`, `score_streams.py --set held` → `results/spa/spa{1,2,4}_held.{json,md}`; run-registry
entries `SPA1-HELD`, `SPA2-HELD`, `SPA4-HELD`.

## Verdict rule

PASS if C1 and C2 both hold; PARTIAL if exactly one holds (reported as such); FAIL otherwise. Thresholds fixed; no
re-scoring; no additional arms on held-out data. The MAC claim is analytical (one 768 × 768 multiply per token and
per image plus statistics) with wall-clock per batch reported.

## Disclosure of the design-data path

SPA1 (multi-block mean alignment) was dropped after a CPU smoke showed it destructive. SPA1b failed its own +4.0
bar (+1.72; +2.65 after fixing an under-regularised online tokenizer estimate — the fix was WR1's declared constant
shrinkage, not a new tuning). SPA2/3/4 bars (+1.0 on ETA / FOA; +2.0 on streams; clean ≤ 0.3) passed as declared.
No held-out data was used for any choice. FOA was evaluated on 2,000-image cells where it does not converge (53.5 <
NEO); it is not a claim here. Expected: C1 ≈ +2.6 episodic / +2.1 continual, C2 ≈ +1.5 / +1.6, with ≈ 0.3 pt of
sampling noise on 2,000-image cells.
