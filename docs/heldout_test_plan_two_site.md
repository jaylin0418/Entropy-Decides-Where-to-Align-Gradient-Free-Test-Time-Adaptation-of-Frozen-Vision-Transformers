# Pre-registration — two-site alignment held-out test (ViT-B/16 and ViT-S/16 on ImageNet-C)

Declared 2026-11-08 14:40 MST, before any image of the slice `heldout3` (`perm[40000:42000]`, split seed 20261019) was
decoded or passed through a model in this project. The slice was reserved untouched when the full-scale study was set up
(2026-11-05) and is disjoint from `design`, `source`, `heldout`, `heldout2` and from the 10,000-image subsets of the
full-scale study, whose placement runs (block 6 chosen on `SUB10K`, severity 5, the augreg2 ViT-B/16 checkpoint)
overlap `heldout` and `heldout2` by about 23 % per cell. Everything below is frozen; the held-out images are scored once.

## Method under test

The two-site method: TokAlign at the pixel site (unchanged: Eq. 1–2 of the report, λ = 0.1, n₀ = 4 images of patches,
cumulative statistics within a cell, EMA 0.2 on streams) plus the same map on the 196 patch tokens at the output of
block 6 (CLS untouched), with the block's own source statistics (patch-token mean and covariance of the 3,000 source
images, `compute_source_stats.py --blocks`), the same λ, the same n₀ (4 images × 196 tokens), the same running
statistics and the same gate. Nothing is tuned on the course data: block 6 and the combination were fixed by the
full-scale study's amendment of 2026-11-08 (rules A–D of `docs/cvpr2027_prereg.md` in the research repository; SUB2K
sweep of all 12 blocks, SUB10K combination, four extra corruptions), on the augreg2 checkpoint. Arms (`run_forward_only.py`
`place_b6`, `tok+place_b6`, `tok+place_b6+headcov`; `run_gradient_host.py` `eta+place_b6`, `tok+eta+place_b6`;
`run_streams.py` `twosite_gated`, `twosite+eta_gated`; implementation `tokalign/aligner.py` key `place`, library
`tokalign/stages.py` `LayerAlign`; the CPU test `test_aligner_place_equals_library_layer_align` checks they agree).

## Design bar (course checkpoint, before any held-out run)

The held-out runs start only if, on the 30 design cells of ViT-B/16 (augreg checkpoint, timm 0.4.12, severities 3 and
5, 2,000 images per cell), `tok+place_b6` − `tok` ≥ +1.5 pt and its clean harm against NEO on the clean cell is
≥ −0.5 pt (`scripts/cvpr/course_two_site_bar.py` in the research repository, applied mechanically by the queue job).
Expected from the full-scale amendment (severity 5, augreg2): +4.96 pt on the frozen host, +2.27 pt on ETA; severity-3
cells should give less. If the bar fails, the held-out runs are not started and the report keeps the method as it is,
with the design result reported as a negative.

## Claims under test (heldout3, scored once)

- **T1 (forward-only).** `tok+place_b6+headcov` ≥ `tok+headcov` + 2.0 pt on the 30 ViT-B/16 cells, with clean harm
  ≥ −0.5 pt against NEO on the clean cell (ungated episodic, as Table 2's "clean" for the forward-only rows), and on
  the clean-return stream the gated two-site arm within 0.3 pt of NEO-EMA on the clean segments.
- **T2 (ETA host).** `tok+eta+place_b6` + NEO ≥ `tok+eta` + NEO + 1.0 pt on the 30 ViT-B/16 cells.
- **Reported either way:** `place_b6` alone; the `cont5` continual stream (gated two-site stack against the gated
  stack; two-site + ETA against TokAlign + ETA, both gated); ViT-S/16 with the same arms (T1 and T2 margins stated as
  expectations, not bars, because the block statistics of ViT-S/16 were never looked at); per-corruption gains; the
  comparison with the design values.

## Verdict rule

PASS if T1 and T2 hold on ViT-B/16; PARTIAL if exactly one holds; FAIL otherwise. Thresholds fixed; no re-scoring; no
additional arms on `heldout3`. The report states the outcome as it is.

## Settings

Model timm `vit_base_patch16_224` / `vit_small_patch16_224` (augreg in21k→in1k, timm 0.4.12 defaults), FP32, TF32 off;
mean/std 0.5, clean resize 248 / crop 224; batch 64; statistics files `stats/vitb_blocks.pt`, `stats/vits_blocks.pt`
(the source split, 3,000 images, with `--blocks`). ETA as the report (official `eata.py`, fishers = None, lr 0.001 at
batch 64), fresh weights per cell; NEO and the head covariance applied to the recorded head input when scoring.
Streams: EMA 0.2 / 0.5 / 0.1 and the gate (0.4 / 0.3) as the report. Run order and commands:
`scripts/cvpr/course_two_site.sh` (research repository) parts `stats`, `design`, `heldout_frozen`, `heldout_eta`,
`heldout_streams`, `heldout_vits`; outputs under `data/derived/course_two_site/`, scored files copied into
`results/scores/` as `forward_only_design_vitb_two_site`, `forward_only_heldout3_vitb`, `eta_heldout3_vitb`,
`streams_heldout3_vitb`, `forward_only_heldout3_vits`, `eta_heldout3_vits`.
