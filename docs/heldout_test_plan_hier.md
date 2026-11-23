# Pre-registration — hierarchical (within/between-image) alignment (ViT-B/16 and ViT-S/16 on ImageNet-C)

Declared 2026-11-08 21:30 MST, before the design run of the method and before any `heldout3` image was passed through
the method. Everything below is frozen; the held-out images are scored once.

## The method

At a site of the frozen ViT (the pixel patches, or the 196 patch tokens after block k), the token x_ip of image i at
position p is split into the image mean m_i = mean_p x_ip and the within-image deviation e_ip = x_ip − m_i. The pooled
covariance of the tokens is the sum of two parts, the between-image covariance Σ_B = Cov_i(m_i) and the within-image
covariance Σ_W = E_i Cov_p(e_ip). The pooled map (TokAlign, LayerAlign) whitens and recolours the sum with one matrix.
The hierarchical map aligns the two parts separately:

    y_ip = μ_s + (m_i − μ_t) M_B + e_ip M_W,    M_B = Σ̂_B^{-1/2} Σ_B^{s 1/2},    M_W = Σ̂_W^{-1/2} Σ_W^{s 1/2},

with running target estimates Σ̂_B (over images, prior n₀ = 64 images of the source between-image covariance) and Σ̂_W
(over tokens, prior n₀ = 4 images × T tokens of the source within-image covariance), both with the constant shrinkage
λ = 0.1 of TokAlign, cumulative within a cell (EMA 0.2 on streams), recomputed every batch, statistics detached. The
source parts come from the same 3,000 source images (`compute_source_stats.py --blocks`, keys `tok_cov_between` /
`tok_cov_within`, `patch_cov_between` / `patch_cov_within`). Implementation: `tokalign/aligner.py` keys `tok_mode` /
`place_mode` ∈ {hier, within, between}, `n0_between_images`; arms in `run_forward_only.py` (`hier_px`, `hier_b6`,
`hier_px+hier_b6`, `tok+hier_b6`, `within_b6`, `between_b6`, `hier_b6_n0x4`, `hier_b6_n0d4`), `run_gradient_host.py`
(`tok+eta+hier_b6`, `hier_px+eta+hier_b6`), `run_streams.py` (`hier_gated`, `hier+eta_gated`).

## Why it should work (the prediction)

On the design split the pooled block-6 map gains +3.2 pt over NEO on 30 cells but loses 1.5–1.9 pt on the six noise
cells, where the drift at block 6 is the largest of all corruptions. Our reading: pixel noise raises the within-image
variance of the tokens (independent per token) and leaves the image means, which carry the content, nearly where they
are; one pooled whitening shrinks the inflated directions for the means too and so compresses the between-image
separation that the head relies on. Blur and the geometric corruptions lower the within-image variance (tokens of an
image become alike), and fog, contrast and brightness move the means. Separate maps act on the part that moved.
Predicted: `hier_b6` is non-negative on the noise cells and keeps the pooled gains elsewhere; the hierarchical two-site
method gains over the pooled two-site method mainly on noise, by roughly +0.3 pt on the 30-cell mean, possibly more if
the separate estimates are also more accurate on the other families. The decomposition run (`decompose_by_depth.py`,
no labels) records the within and between parts at every site for every cell and will show whether the reading of the
shifts is right, independent of accuracy.

## Design bar (course checkpoint, before any held-out run)

The method arm is the better of `hier_px+hier_b6` and `tok+hier_b6` on the 30 design cells (`course_hier_bar.py`,
chosen once). The held-out runs start only if, on the 30 design cells of ViT-B/16:
(i) method ≥ `tok+place_b6` (the pooled two-site method, 67.47) + 0.5 pt; (ii) clean harm of the method against NEO on
the clean cell ≥ −0.5 pt; (iii) `hier_b6` − NEO averaged over the six noise cells ≥ 0. If the bar fails, the held-out
runs are not started and the report keeps the pooled two-site method, with the hierarchical result and the decomposition
reported as a negative or an analysis.

## Claims under test (heldout3, scored once)

- **T3 (forward-only).** method ≥ `tok+place_b6` + 0.5 pt on the 30 ViT-B/16 cells, clean harm ≥ −0.5 pt against NEO.
- **T4 (ETA host).** `tok+eta+hier_b6` + NEO ≥ `tok+eta+place_b6` + NEO + 0.3 pt on the 30 ViT-B/16 cells.
- **Reported either way:** `hier_b6` alone and its noise cells; the `cont5` and clean-return streams (`hier_gated` vs
  `twosite_gated`, `hier+eta_gated` vs `twosite+eta_gated`); ViT-S/16 with the same arms (margins as expectations);
  per-corruption gains; the comparison with the design values; `hier_px+eta+hier_b6`.

## Verdict rule

PASS if T3 and T4 hold on ViT-B/16; PARTIAL if exactly one holds; FAIL otherwise. Thresholds fixed; no re-scoring; no
additional arms on `heldout3`. The report states the outcome as it is. The pooled two-site results of
`docs/heldout_test_plan_two_site.md` stay as reported there whatever the outcome here.

## Settings

As `docs/heldout_test_plan_two_site.md` (models, FP32, batch 64, splits, ETA, streams, gate), with the statistics files
`stats/vitb_hier.pt`, `stats/vits_hier.pt` (the same source split and the same pooled statistics as `*_blocks.pt`, plus the
within/between parts). Run order: `scripts/cvpr/course_hier.sh` (research repository) parts `stats`, `design`,
`decompose`, `heldout_frozen`, `heldout_eta`, `heldout_streams`, `heldout_vits`; outputs under
`data/derived/course_two_site/` with the tag `_hier`; scored files copied into `results/scores/` as
`forward_only_design_vitb_hier`, `decompose_by_depth_design_vitb`, `forward_only_heldout3_vitb_hier`,
`eta_heldout3_vitb_hier`, `streams_heldout3_vitb_hier`, `forward_only_heldout3_vits_hier`, `eta_heldout3_vits_hier`.

## Outcome (2026-11-09, design bar applied mechanically by `scripts/cvpr/course_hier_bar.py` of the research repository)

Design run `results/scores/forward_only_design_vitb_hier.json` (ViT-B/16, 30 design cells, scored against the NEO arm of the
two-site design run): method arm = `hier_px+hier_b6` at 67.62 against `tok+place_b6` at 67.47, gain +0.15 (bar ≥ +0.5:
**fails**); clean harm against NEO −0.20 (bar ≥ −0.5: passes); `hier_b6` − NEO on the six noise cells −1.11 (bar ≥ 0:
**fails**; the pooled block-6 map is −0.91 there). **Verdict FAIL.** The held-out parts were not started (the job script
checked the bar file and exited), `heldout3` was not touched by any hierarchical arm, and the report keeps the pooled
two-site method. The hierarchical map and the decomposition are reported as an analysis with a negative result (report
Section 3.3, the paragraph "What moves at a site is not always what a map can restore", Table S9, Figure S5).

What the data say about the prediction above. At the pixel site the reading was right: noise inflates the within-image
part (+0.13 log trace ratio, mean of severities 3 and 5) and shrinks the between-image part (−0.42), blur shrinks the
within part (−0.33) and leaves the between part, contrast collapses the within part (−4.65), and snow, frost and brightness
move the means. At block 6 the reading was wrong: for every corruption the within-image part is within 9 % of the source
and the between-image part is 17–49 % below it, most under noise (−0.64). Accordingly `within_b6` (65.34) ≈ `hier_b6`
(65.39) ≈ `place_b6` (65.54), `between_b6` alone is 64.25, and the between map adds +0.05 on top of the within map: the
block-6 gain is a within-image correction, and rescaling the collapsed image means does not restore the separation that
noise removed. The two-level map at the pixels adds +0.33 over TokAlign (+3.6 on contrast, 9 of 15 corruptions better);
the two-level pair adds +0.15 over the pooled pair (6 of 15 better). The prior of the between-image estimate does not
matter (n₀ = 256: +0.10; n₀ = 16: +0.04 against n₀ = 64).
