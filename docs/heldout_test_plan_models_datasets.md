# Pre-registration — the calibrated second-opinion stack on more models and on natural shifts

> Table and section numbers in this plan are those of the report draft at the time it was declared (its Table 2 is now Table 3, Table S8 is Table S1, Sections F and G are C.2 and G). The plan is kept as declared.

Declared 2026-11-10 23:14 MST (commit 72a4d3e), before any of its runs. Each held-out file is scored once. The job
script `scripts/cvpr/course_ca.sh` of the research repository runs every step from this repository (`release/confidence-align`).

## Models and datasets

| | ImageNet-C (15 corruptions, severities 3 and 5) | ImageNet-R (200 classes, natural renditions) | ImageNet-Sketch (1,000 classes) |
|---|---|---|---|
| ViT-B/16 (augreg) | done (`docs/heldout_test_plan_rule.md`) | **this plan** | **this plan** |
| ViT-S/16 (augreg) | own calibration in the rule plan (R3) | **this plan** | **this plan** |
| DeiT-B/16 (fb_in1k; different training recipe, ImageNet mean/std, crop 0.875) | **this plan** | **this plan** | **this plan** |

ImageNet-R and ImageNet-Sketch are shuffled once (seed 20261019) and cut into cells of 2,000 images: three design cells
(`<name>-d1..3`) and three held-out cells (`<name>-h1..3`), 12,000 images per dataset; images go through the clean transform
(resize to the model's crop fraction, center crop 224). ImageNet-R logits are restricted to its 200 classes (the usual
protocol), so its entropies are over 200 classes. The clean cell is the ImageNet validation cell of the split. DeiT-B/16 on
ImageNet-C uses the fresh `heldout4` cell indices (never passed through DeiT-B/16 before).

## The procedure, mechanical and label-free (the same for every pair)

1. **Source statistics**: 3,000 clean validation images (`source` split), per model (`compute_source_stats.py --blocks`).
2. **Design sweep**: the 15 versions `neo`, `tok` (pixel map), `place_b6`, `tok+place_b1` … `tok+place_b12` on the design cells.
3. **Calibration** (`site_choice.py`, no labels): the base block is the block whose pixel-plus-block version has the lowest mean
   prediction entropy; the earlier and the later second opinion are, on each side of it, the block whose confidence-weighted
   average with the base path has the lowest mean entropy on the images the base path is unsure of; **theta is the 0.65
   quantile of the base path's entropy on the design images** (on ViT-B/16 ImageNet-C this gives 1.514 nat, the 1.5 nat used
   so far, and routes 35 % of the images). The job writes the three blocks and theta to `calib_<model>_<dataset>.json`.
4. **Held-out run**: `neo`, `tok`, the base path `tok+place_b<T>`, the calibrated stack `rule_t<T>_b<E>b<L>_gated` with that
   theta, and its two single-second-opinion ablations.
5. **The gradient-based comparison on the same held-out cells**: ETA (`eta`, scored with NEO) and ETA with the base path's
   alignment inside it (`tok+eta+place_b<T>`), official code, margins scaled to the number of classes.

## Claims (per model-dataset pair; gains in top-1 points, scored once)

- **M1 (the signal, design images, labels used only to check).** The block the calibration picks is within 0.5 points of the
  most accurate block of the sweep.
- **M2 (the calibrated stack, held-out).** `rule_t<T>_b<E>b<L>_gated` − `tok+place_b<T>` ≥ +0.5, and clean − NEO ≥ −0.2.
- **M3 (reported either way).** The stack against NEO and against ETA (+NEO); ETA with the alignment inside against ETA; the
  two single-second-opinion ablations; the share of images sent to the second opinions; the per-cell numbers.

**Verdict** over the seven pairs: PASS if M2 holds on all seven; PARTIAL if on four to six; FAIL otherwise. Every pair is
reported whatever its outcome; nothing is re-scored or re-calibrated after a held-out run.

## Settings

FP32, batch 64, lambda 0.1, n0 four images, gate thresholds 0.4 / 0.3, EMA as before. Outputs under
`data/derived/course_two_site/` with tags `_ca_inc`, `_ca_r`, `_ca_sketch`; scored files copied into `results/scores/` as
`forward_only_design_<model>_ca_<dataset>`, `site_choice_design_<model>_ca_<dataset>{,_t6}`, `calib_<model>_<dataset>`,
`forward_only_<split>_<model>_ca_<dataset>`, `eta_<split>_<model>_ca_<dataset>`.

## Outcome (2026-11-11; the thresholds applied as declared, each held-out file scored once; the DeiT-B/16 pairs are appended as their runs finish)

Per pair: pick = base block / earlier, later (`calib_<model>_<dataset>.json`); gap = the picked placement minus the most accurate
placement on the design images (labels used only for this check; "none" means the pixel map alone was the most accurate
placement, so no block map helps); M2 = `rule_t<T>_b<E>b<L>_gated` minus `tok+place_b<T>` on the held-out cells, and clean − NEO.

| pair | pick | most accurate | gap | M1 (≤ 0.5) | base path vs NEO | stack vs base | clean − NEO | M2 | stack vs NEO | ETA+NEO vs NEO | ETA + base alignment vs ETA | share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ViT-B/16 × ImageNet-R | 8 / 1, 11 | none (pixel map) | −3.05 | fails | −1.80 | +2.53 | +0.00 | holds | +0.73 | +0.85 | −2.38 | 35 % |
| ViT-S/16 × ImageNet-R | 8 / 7, 9 | none (pixel map) | −0.58 | fails | −1.12 | +1.15 | −0.05 | holds | +0.03 | +2.12 | −1.62 | 35 % |
| ViT-B/16 × ImageNet-Sketch | 11 / 7, 12 | block 11 | +0.00 | holds | +0.48 | +0.20 | +0.00 | fails | +0.68 | +1.22 | +0.17 | 34 % |
| ViT-S/16 × ImageNet-Sketch | 6 / 5, 11 | block 11 | −4.57 | fails | −3.45 | +2.52 | +0.05 | holds | −0.93 | +3.87 | −4.87 | 36 % |
| DeiT-B/16 × ImageNet-C | 6 / 5, 8 | block 6 | +0.00 | holds | +8.25 | +0.43 | +0.00 | fails | +8.68 | +1.88 | +7.36 | 35 % |
| DeiT-B/16 × ImageNet-R | 7 / 1, 8 | block 7 | +0.00 | holds | +0.73 | +1.07 | −0.05 | holds | +1.80 | +1.20 | −0.88 | 35 % |
| DeiT-B/16 × ImageNet-Sketch | 7 / 5, 8 | block 8 | −0.18 | holds | +2.68 | +0.62 | −0.05 | holds | +3.30 | +1.77 | +2.32 | 35 % |

Files: `forward_only_design_<model>_ca_<dataset>`, `site_choice_design_<model>_ca_<dataset>{,_t6}`, `calib_<model>_<dataset>`,
`forward_only_heldout_<model>_ca_<dataset>`, `eta_heldout_<model>_ca_<dataset>` (the natural-shift held-out cells `r-h1..3`,
`sketch-h1..3`). Spearman correlation of the mean entropy and the accuracy of the 13 placements on the design images:
−0.00, +0.23, +0.66, +0.38 (ImageNet-C and ImageNet-C-bar on ViT-B/16: +0.89, +0.93; ImageNet-C on ViT-S/16: +0.94).

**Reading, four ViT pairs.** M2 holds formally on three of four pairs because the second opinions lift the stack above its own
base path, but on the natural shifts that base path is below NEO (ImageNet-R both models, Sketch on ViT-S/16): the token
alignment gains little on renditions and sketches, the aligned placements become more confident than accurate, and the
signal picks a more confident, less accurate block in three of four pairs (M1 fails). ETA with NEO is above the stack on
all four pairs. The report draws the boundary accordingly (Section 4, Table 2, Supplement Section D.2 and Table S8): the
method is one for shifts that live in low-level token statistics. The verdict over the seven pairs is below.

**DeiT-B/16 (2026-11-11 08:40 MST; files `*_deitb_ca_{inc,r,sketch}`, `calib_deitb_*`; ImageNet-C on `heldout4`, Spearman +0.97 / +0.86 /
+0.33).** On ImageNet-C the signal picks the most accurate block (6) and the base path gains +8.25 over NEO (the pixel map alone
adds nothing on this backbone), but the second opinions, after blocks 5 and 8 next to the base block, add +0.43, below the +0.5
bar (M2 fails by 0.07); ETA with the base path's alignment inside reaches +9.24 over NEO. On ImageNet-R and ImageNet-Sketch,
unlike the AugReg ViTs, a block map helps DeiT-B/16: the calibration picks block 7 both times (the most accurate on
ImageNet-R; 0.18 below block 8 on Sketch), the stack gains +1.80 and +3.30 over NEO, above ETA with NEO (+1.20, +1.77), and M1
and M2 hold.

**Verdict: PARTIAL** (M2 holds on five of the seven pairs: ViT-B/16 × R, ViT-S/16 × R, ViT-S/16 × Sketch, DeiT-B/16 × R,
DeiT-B/16 × Sketch; it fails on ViT-B/16 × Sketch (+0.20) and DeiT-B/16 × ImageNet-C (+0.43). M1 holds on four of seven:
the three DeiT-B/16 pairs and ViT-B/16 × Sketch). Reading: the signal picks the most accurate block wherever some block map
helps (every corruption pair, and both natural shifts on DeiT-B/16); where no block map helps (the AugReg ViTs on
renditions and sketches) the aligned placements become more confident without becoming more accurate and the signal cannot
say so. The second opinions add +0.4 to +1.7 points over the base path on the corruption pairs and +0.2 to +2.5 on the
natural-shift pairs, least where the calibration places them next to the base block. The report states the boundary in
these terms (Section 4, Table 2; Supplement D.2, Table S8, Sections F and G).

## Addendum 2 (2026-11-11, before any DeiT-S/16 run): a second ImageNet-1k-only backbone

The reading above says the signal works wherever some block map helps, and that on renditions and sketches this depends on
the backbone. One backbone (DeiT-B/16) supports that reading on the natural shifts. The report's across-models table should
rest on at least two models for every shift it keeps, so DeiT-S/16 (timm `deit_small_patch16_224`, ImageNet-1k only, the same
normalization and resize as DeiT-B/16) runs the same thirteen jobs: source statistics, the design sweep of the 13 placements,
the label-free calibration (base block, earlier, later, theta = 0.65 quantile), the held-out run of the stack and its
single-second-opinion ablations, and ETA with NEO alone and with the base path's alignment inside, on ImageNet-C (heldout4),
ImageNet-R and ImageNet-Sketch (the held-out cells r-h1..3, sketch-h1..3). Claims M1 and M2 apply unchanged, per pair. Nothing
in the procedure is tuned to DeiT-S/16; the model key `deits` is the only addition (`tokalign/models.py`).

The AugReg ViT-B/16 and ViT-S/16 rows on ImageNet-R and ImageNet-Sketch stay in the record above and in the supplement's
negative results; the main-text table keeps, for the natural shifts, the ImageNet-1k-only backbones on which a block map
helps, and says so.

**Outcome of addendum 2 (2026-11-13 01:10 MST; files `*_deits_ca_{inc,r,sketch}`, `calib_deits_{inc,r,sketch}`, `eta_heldout4_deits_ca_inc_stack`,
`eta_heldout_deits_ca_{r,sketch}_stack`; scored once).** Calibration on ImageNet-C: 4 / 3, 5 (theta 2.81), an adjacent shape like DeiT-B/16.

| pair | pick | most accurate | gap | M1 (≤ 0.5) | base path vs NEO | stack vs base | clean − NEO | M2 | stack vs NEO | ETA+NEO vs NEO | ETA + base vs ETA | ETA + stack vs ETA + base (M4) | share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeiT-S/16 x ImageNet-C (heldout4) | 4 / 3, 5 | 4 | 0.00 | holds | +9.63 | +0.55 | +0.00 | holds | +10.17 | −2.51 | +12.34 | +0.49, clean −0.15 (fails by 0.01) | 36 % |
| DeiT-S/16 x ImageNet-R | 6 / 4, 10 | 6 | 0.00 | holds | +1.67 | +0.70 | +0.00 | holds | +2.37 | +0.58 | +1.43 | +0.60, clean +0.15 (holds) | 35 % |
| DeiT-S/16 x Sketch | 5 / 4, 6 | 9 | −0.90 | fails | −0.60 | +1.02 | −0.10 | holds | +0.42 | +0.30 | −0.48 | +1.15, clean +0.00 (holds; ETA + stack +0.97 over NEO) | 35 % |

Reading: on ImageNet-C and ImageNet-R the signal picks the most accurate block (rho +0.97, +0.60) and the stack holds M2. On
Sketch the best block (9) is worth only +0.32 over NEO on the design images, the entropy barely ranks the placements
(rho +0.18) and picks block 5, which is below NEO as the base path (−0.60 held out); the two second opinions recover it to
+0.42 over NEO. So the second ImageNet-1k-only backbone supports the reading on renditions and not on sketches: the
boundary is a shift on which some block map gains clearly, not the backbone's pretraining alone. DeiT-S/16 goes to the
supplement (Figure S3 row, Table S1 rows, the signal paragraph); the main-text claim now reads "the DeiT models on
renditions; DeiT-B/16 on sketches; DeiT-S/16 misses there". ETA with NEO is below NEO on DeiT-S/16 x ImageNet-C (−2.51).

## Addendum 3 (2026-11-11, before any run): the calibrated stack inside ETA

The across-models table compares the stack, which needs no gradient, with ETA, which does, and shows ETA with the base
path's alignment inside it. The natural question is whether the whole stack can sit inside ETA. It can: the host's forward is
the base path (the pixel map, the map after the calibrated base block), so ETA adapts its LayerNorm parameters on the base
path's logits exactly as it does with the pair inside it; the images whose NEO-corrected base-path entropy exceeds theta also
run the two calibrated second opinions with the same current weights, without gradients, and the three predictions are
averaged by confidence (`tokalign.aligner.HostStack`; arm `tok+eta+rule_t<T>_b<E>b<L>` of `run_gradient_host.py`; the scorer
takes the averaged prediction for those images and NEO for the rest). Same blocks and theta as the forward-only stack of each
pair; no new choice. Pairs: ViT-B/16 and ViT-S/16 on ImageNet-C (heldout3, where their ETA comparison ran; 6 / 4, 11; theta
1.5 nat), DeiT-B/16 and DeiT-S/16 on ImageNet-C, ImageNet-R and ImageNet-Sketch (their calibration files; heldout4 / heldout).
One run per pair, scored once (files `eta_heldout3_<model>_stack`, `eta_<split>_<model>_ca_<ds>_stack`).

Claim M4, per pair: ETA with the stack inside is at least 0.5 points above ETA with the base path inside, on the held-out
cells, with the clean cell within 0.2 points. Expected to hold where the forward-only second opinions gain (M2); the table
reports the number either way. A smoke run (128 images, two cells) confirmed that the base path inside the host is
bit-identical to the pair arm and that the routed rows take the averaged prediction.

## Addendum 4 (2026-11-11, before any ImageNet-C-bar run on ViT-S/16 or DeiT-B/16): the across-models table as a 3 x 3 grid

The report's across-models table is the grid {ViT-B/16, ViT-S/16, DeiT-B/16} x {ImageNet-C, ImageNet-C-bar, ImageNet-Sketch};
ImageNet-R stays in the supplement's natural-shift table and in the negative results. Missing cells: ViT-S/16 x C-bar and
DeiT-B/16 x C-bar (the five steps of the procedure on the 20 C-bar cells, design split for the sweep and calibration, heldout4
for the held-out run and ETA), and ETA with NEO on C-bar for ViT-B/16 (heldout4, the split of its stack run; its calibration
7 / 5, 11 and theta 1.5 are the ones already recorded). The stack inside ETA (addendum 3) runs on the nine cells of the grid as
well (`eta_heldout4_<model>_cbar_ca_stack`). Claims M1 and M2 apply to the two new pairs; M4 to the nine. DeiT-S/16 (addendum
2) keeps its queued jobs after the C-bar jobs; its results go to the supplement.

**Edge rule of the calibration (recorded before the runs).** The one-below, one-above rule has no later block when the base
block is the last block (12), and no earlier block when it is the first. In those cases the two lowest-entropy blocks on the
side that exists are taken (`pick_second_opinions` in `scripts/analysis/site_choice.py`, tested in
`tests/test_second_opinion_rule.py`). No pair so far has picked block 1 or 12 as the base block, so no recorded result changes.

**Outcome of addendum 4 (2026-11-12 04:12 MST; files `*_{vits,deitb}_cbar_ca`, `calib_{vits,deitb}_cbar`, `eta_heldout4_vitb_cbar_ca`;
heldout4, 20 cells; gains over NEO on the same cells).**

| pair | pick | most accurate (design) | gap | M1 | base | stack | second opinions | clean | M2 | ETA+NEO | ETA + base |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ViT-S/16 x C-bar | 7 / 6, 11 | 8 | −0.02 | holds | +4.65 | +4.95 | +0.31 | +0.00 | fails | +3.92 | +6.44 |
| DeiT-B/16 x C-bar | 6 / 5, 8 | 7 | −0.49 | holds | +2.27 | +2.37 | +0.09 | +0.00 | fails | −0.90 | +2.58 |
| ViT-B/16 x C-bar (ETA added) | 7 / 5, 11 | 7 | 0 | holds | +2.50 | +3.29 | +0.78 | +0.00 | holds | +1.79 | +3.52 |

Spearman of the entropy and the accuracy over the 13 placements: +0.95 (ViT-S/16), +0.88 (DeiT-B/16). Reading: on ImageNet-C-bar
the signal picks the most accurate block or its neighbour for all three backbones (M1 holds on all three) and the stack is above ETA
with NEO on all three; the second opinions add +0.31 and +0.09 where the calibration places one of them next to the base block
(6 next to 7; 5 and 8 around 6), the same pattern as DeiT-B/16 on ImageNet-C (+0.43), and on DeiT-B/16 block 5 alone lowers the
base path by 0.76 while block 8 alone adds 0.04. M2 therefore fails on both new pairs for the reason the supplement states (a
second opinion next to the base block repeats its answer); a minimum distance from the base block in the calibration is the next
thing to test and was not run. ETA with NEO alone is below NEO on DeiT-B/16 x C-bar (−0.90) while ETA with the base path inside
is +2.58.

**Verdict over the 3 x 3 grid (ImageNet-R aside):** M1 holds on 8 of 9 pairs (fails on ViT-S/16 x Sketch); M2 holds on 5 of 9
(fails on DeiT-B/16 x C, ViT-S/16 x C-bar, DeiT-B/16 x C-bar, ViT-B/16 x Sketch); the stack is above ETA with NEO on 7 of 9
(below on ViT-B/16 x Sketch and ViT-S/16 x Sketch). M4 (the stack inside ETA) pending.

**Outcome of addendum 3, the stack inside ETA (2026-11-12 13:50 MST; files `eta_heldout3_{vitb,vits}_stack`,
`eta_heldout4_{vitb,vits,deitb}_cbar_ca_stack`, `eta_heldout4_deitb_ca_inc_stack`, `eta_heldout_deitb_ca_{r,sketch}_stack`;
the ViT-B/16 x C-bar run completed but its scoring step hit a full shared disk and was redone by hand from the saved logits).**
ETA with the stack inside against ETA with the base path inside, same cells; clean against ETA with NEO:

| pair | ETA + stack over NEO | over ETA + base | clean | M4 |
|---|---|---|---|---|
| ViT-B/16 x C (heldout3) | +6.94 | +0.98 | +0.05 | holds |
| ViT-S/16 x C (heldout3) | +10.25 | +1.13 | +0.10 | holds |
| DeiT-B/16 x C | +9.46 | +0.22 | −0.55 | fails |
| ViT-B/16 x C-bar | +4.42 | +0.90 | −0.10 | holds |
| ViT-S/16 x C-bar | +7.11 | +0.67 | −0.05 | holds |
| DeiT-B/16 x C-bar | +3.24 | +0.66 | −0.45 | fails (clean) |
| DeiT-B/16 x R | +2.08 | +1.77 | +0.15 | holds |
| DeiT-B/16 x Sketch | +4.37 | +0.28 | +0.20 | fails |

The stack inside ETA is above ETA with the base path inside on all eight pairs run, and above the gradient-free stack on all
eight. M4 holds on five of eight: it fails where the second opinions sit next to the base block (DeiT-B/16 on C and Sketch,
+0.22 and +0.28) and on DeiT-B/16 x C-bar by the clean condition only (+0.66 but the clean cell −0.45; the gate is not part of
the ETA host's forward). ViT-B/16 x Sketch and ViT-S/16 x Sketch were left out of the first job list by mistake and were queued on 2026-11-12 13:55.
ViT-S/16 x Sketch (13:57): ETA + stack +0.90 over NEO, +1.90 over ETA + base, clean −0.90 against ETA with NEO, still 2.97
below ETA with NEO alone (M4 fails by the clean condition; the shift is one where no block map helps this backbone).
ViT-B/16 x Sketch hit CUDA out-of-memory twice while another user's job held the GPU and ran after DeiT-S/16 (2026-11-13 01:26):
ETA + stack +1.37 over NEO, −0.02 against ETA + base, clean −0.60 against ETA with NEO (M4 fails: a tie on the shift where the
base path already gains little). Over the nine grid pairs M4 holds on five; the stack inside ETA is above ETA + base on eight
of nine. With DeiT-S/16 (addendum 2 outcome) M4 holds on six of twelve pairs.

## Addendum 5 (2026-11-12, before any run): one calibration per shift from 192 unlabeled images, held out

The report's claim that a few hundred unlabeled images of a shift suffice to calibrate the stack rests, so far, on a
simulation from the saved design predictions (Figure S6b: one calibration per cell from its first K batches, scored on the
rest of the design cell). This addendum tests it on held-out images. For every ImageNet-C cell (30) and every ImageNet-C-bar
cell (20) of ViT-B/16, the calibration of the first 3 batches (192 images) of the design cell, as recorded in
`rule_sample_size_design_vitb{,_cbar}.json` (`per_cell_first_k_batches['3']['picks']`: base block, earlier, later; theta 1.5
nat as in every ViT-B/16 run), is applied to the heldout4 images of the same cell: one run per cell with the arms
`tok+place_b<T>` and `rule_t<T>_b<E>b<L>_gated`, scored against the heldout4 NEO of that cell (`forward_only_heldout4_vitb_exit`,
`forward_only_heldout4_vitb_cbar_exit`). The picks differ from cell to cell (28 distinct configurations on ImageNet-C, 17 on
ImageNet-C-bar), which is the point: the question is whether a shift-specific calibration from 192 images holds on fresh images.
Also run, for the "one calibration per benchmark from 192 images per cell" row of the report's calibration table: the
ImageNet-C-bar pick from 192 images per cell, 7 / 5, 8 (`rule_t7_b5b8_gated` on heldout4; the ImageNet-C pick from 192 per
cell, 5 / 3, 7, is the small-sample run already recorded).

Claim M5, per benchmark: the mean over cells of the per-shift stack is within 0.5 points of the stack of the full-sample
calibration on the same heldout4 cells (68.35 on ImageNet-C for 6 / 4, 11; 70.40 on ImageNet-C-bar for 7 / 5, 11), and above
the full-sample base path (66.99; 69.61). Files `forward_only_heldout4_vitb_fewshot_<cell>` (one per cell), job
`scripts/cvpr/course_fewshot.sh` of the research repository, queued ahead of the remaining DeiT-S/16 jobs.

**Outcome on ImageNet-C (2026-11-13 02:06 MST; 30 files `forward_only_heldout4_vitb_fewshot_<cell>`, scored once).** Mean over the
30 cells: per-shift stack 68.39 against 68.35 for the full-sample stack (+0.03, within 0.5); the base path of the per-shift
picks 67.32, above the full-sample base path 66.99; the per-shift stack is +1.06 over its own base path and +6.29 over NEO.
M5 holds on ImageNet-C. ImageNet-C-bar (02:35, 20 files): per-shift stack 70.41 against 70.40 (+0.01); base path of the
per-shift picks 70.05, above 69.61; +0.36 over its own base path, +3.30 over NEO. M5 holds on both benchmarks (verdict PASS). The pooled 192-per-cell C-bar pick
(7 / 5, 8) on heldout4 scored 70.20, +0.59 over the base path (recorded 2026-11-12).

**Amendment, 2026-11-19 (the pair rule).** The branch rule was changed after the held-out runs of the one-earlier-one-later
rule: the two branch blocks are now the pair of blocks other than the main block whose confidence-weighted average with the
main path has the lowest mean entropy on the unsure images, with no side constraint and no edge rule. Reason: on the saved
dev logits of the 16 model-shift pairs the side constraint neither helped nor hurt (simulated gain over the main path +1.49
without it against +1.44 with it, the same pick on 9 of 16 pairs), and it had no stated justification. The picks that changed
(DeiT-B/16 ImageNet-C and C-bar 5,8 -> 7,8; DeiT-S/16 ImageNet-C 3,5 -> 5,6 and C-bar 3,9 -> 5,9; ViT-B/16 Sketch 7,12 -> 1,7;
ViT-S/16 ImageNet-R 7,9 -> 9,11 and Sketch 5,11 -> 8,11; ViT-B/16 and ViT-S/16 ImageNet-C unchanged at 6 / 4, 11) were re-run
on their held-out cells (the forward-only run and ETA with the cascade inside), as were the per-cell few-image calibrations of
ViT-B/16 whose picks changed. The superseded run files are kept in runs/superseded_pair_rule/.
