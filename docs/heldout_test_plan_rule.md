# Pre-registration — the rule-chosen second-opinion stack (ViT-B/16 on ImageNet-C and ImageNet-C-bar, ViT-S/16 on ImageNet-C)

Declared 2026-11-10 21:21 MST (commit c2891f8), before any of the runs below. Everything is frozen; each
held-out file is scored once. The job script `scripts/cvpr/course_rule.sh` (research repository) reads the FROZEN lines below.

FROZEN_ARMS_INC: rule_t6_b4b11_gated
FROZEN_ARMS_CBAR: rule_t6_b4b11_gated rule_t7_b5b11_gated tok+place_b7

## The question

The second-opinion stack of the report uses blocks 5 and 9, chosen from labeled design images under a compute budget
(`docs/heldout_test_plan_exit.md`). This test asks whether the same stack still gains when every block is chosen by an
unlabeled signal, the prediction entropy, with no labels and no search over labeled gains:

1. **Trunk.** Over the arms "pixel map + the map after block k" (k = 1..12), the trunk is the block whose arm has the lowest
   mean prediction entropy on unlabeled images (`site_choice.py`, `label_free.trunk_pick_by_entropy`).
2. **Where to ask again.** An image whose trunk entropy exceeds theta = 1.5 nat asks two second opinions, one that aligns
   earlier (a block before the trunk) and one that aligns later (a block after it).
3. **Which earlier and which later block.** On each side, the block whose routed mixture with the trunk (the entropy-weighted
   average on the routed images) has the lowest mean entropy (`label_free.second_opinion_pick_below` / `_above`, the
   one-below-one-above rule added to `site_choice.py` for this test). The pixel site alone is not a block and is not a candidate.

Arms are named `rule_t<T>_b<A>b<B>_gated` (trunk block T, second opinions after blocks A and B, gate of the stack, theta 1.5;
`scripts/episodic/run_forward_only.py::rule_arm`); `rule_t6_b5b9_gated` is identical to the report's `exit_b5b9_gated`.

## What the rule picked on the design images (CPU, saved logits; no held-out image used)

| design images | trunk (rule 1) | earlier, later (rule 3) | routed-mixture gain over the pair of that trunk | best labeled pair |
|---|---|---|---|---|
| ImageNet-C, ViT-B/16 (`site_choice_design_vitb.json`) | block 6 (= the most accurate) | 4, 11 | +1.21 | 3+9: +1.30 |
| ImageNet-C-bar, ViT-B/16, own trunk (`site_choice_design_vitb_cbar_t7.json`) | block 7 (= the most accurate) | 5, 11 | +1.09 over the block-7 pair (69.48) | 4+10: +1.29 |
| ImageNet-C-bar, ViT-B/16, trunk 6 kept (`site_choice_design_vitb_cbar.json`) | 6 | 4, 11 | +1.63 | 2+9: +1.74 |

The ImageNet-C configuration (trunk 6, second opinions 4 and 11) is therefore the one the rule gives with no ImageNet-C-bar
data at all; on ImageNet-C-bar it is tested as a transfer to unseen corruptions. Cost in blocks per routed image: 8 + 6 = 14
against 7 + 6 = 13 for blocks 5 and 9 (the same order; the measured time is not part of the claims).

## Claims (each file scored once; gains against `tok+place_b6` of the same slice, clean against `neo`)

- **R1 (ImageNet-C, heldout4 = perm[42000:44000], 30 cells).** `rule_t6_b4b11_gated` − `tok+place_b6` ≥ +0.7 points, and
  clean − NEO ≥ −0.2 (baselines from `forward_only_heldout4_vitb_exit.json`, the same images).
- **R2 (ImageNet-C-bar, heldout4, 20 cells, transfer).** `rule_t6_b4b11_gated` − `tok+place_b6` ≥ +0.7, clean ≥ −0.2
  (baselines from `forward_only_heldout4_vitb_cbar_exit.json`).
- **R3 (ViT-S/16, heldout3, 30 cells).** The arm named by rules 1–3 applied to a new ViT-S/16 design sweep
  (`forward_only_design_vits_rule_sweep`, arms neo tok place_b6 tok+place_b1..b12, then `site_choice.py` twice: trunk 6 to
  read the trunk pick, then that trunk to read the earlier and later picks; the arm name is written by the job) −
  `tok+place_b6` ≥ +0.5, clean ≥ −0.2 (baselines from `forward_only_heldout3_vits.json`).

**Verdict:** PASS if R1, R2 and R3 all hold; PARTIAL if one or two hold; FAIL otherwise. Thresholds, theta, the rule and the
arm names are fixed; no re-scoring, no second arm on the same slice if a claim fails.

**Reported either way:** on ImageNet-C-bar `rule_t7_b5b11_gated` (the rule with its own trunk) and `tok+place_b7` (that trunk
alone); the comparison of every rule arm with `exit_b5b9_gated` of the same slice; routed fractions; the ViT-S/16 sweep, the
mean entropy per block and the picks; per-corruption gains.

## Settings

As `docs/heldout_test_plan_exit.md`: FP32, batch 64, statistics `vitb_blocks.pt` / `vits_blocks.pt`, gate thresholds 0.4 / 0.3,
NEO at the head; the branches keep their own statistics and NEO means. Outputs under `data/derived/course_two_site/` with the
tags `_rule`, `_cbar_rule`, `_rule_sweep`; scored files copied into `results/scores/` as `forward_only_heldout4_vitb_rule`,
`forward_only_heldout4_vitb_cbar_rule`, `forward_only_design_vits_rule_sweep`, `site_choice_design_vits_rule_t6`,
`site_choice_design_vits_rule`, `forward_only_heldout3_vits_rule`.

## Addendum 1 (declared 2026-11-10 21:26 MST, commit f5ce8ed, before the ablation runs; the inc job of the main plan was already running)

**Ablation of the two second opinions, reported either way (no claim).** On the same held-out slices, each scored once:
the trunk alone (`pair_gated`, already in the heldout4 files), the trunk with only the earlier second opinion, the trunk with
only the later one, and both (the arm of the main plan). ViT-B/16: `rule_t6_b4_gated` and `rule_t6_b11_gated` on heldout4
ImageNet-C (tag `_rule_abl`) and ImageNet-C-bar (tag `_cbar_rule_abl`). ViT-S/16: the two single-site arms named by the rule's
picks, run in the same heldout3 job as the main arm (tag `_rule`). Expectation from the design simulation (ImageNet-C, trunk 6):
earlier only +0.58, later only +0.80, both +1.21; the two should add, as blocks 5 and 9 did.

## Addendum 2 (declared 2026-11-10 21:32 MST, commit 8455489, before its runs)

**The small-sample calibration, tested on held-out images.** `scripts/analysis/rule_sample_size.py` (A9, A10) applies the
rule to the first n unlabeled images of every design cell, pooled: on ImageNet-C it picks trunk 5, earlier 3, later 7 for
every n from 64 to 640 per cell (simulated gain +1.09 against +1.21 for the full-sample pick 6 / 4 / 11), and on ImageNet-C-bar
trunk 7, earlier 5, later 9 for n = 64 and 128 (+1.91 against +1.84 for 7 / 5 / 11). Per shift, the first 3 batches (192 images)
give +1.24 and +1.75 in simulation. So that the paper can say "a few hundred unlabeled images per shift suffice" from a
held-out run and not from a simulation:

- **R4 (ImageNet-C, heldout4).** `rule_t5_b3b7_gated` − `tok+place_b6` ≥ +0.7, clean − NEO ≥ −0.2 (tag `_rule_small`).
- **R5 (ImageNet-C-bar, heldout4).** `rule_t7_b5b9_gated` − `tok+place_b6` ≥ +0.7, clean ≥ −0.2 (tag `_cbar_rule_small`).

R4 and R5 join the verdict: PASS if R1–R5 all hold; PARTIAL if one to four hold; FAIL otherwise. Also reported: the gains
against `tok+place_b5` / `tok+place_b7` (the pair of the chosen trunk; `tok+place_b5` is run in the same job, `tok+place_b7`
in the cbar job of the main plan) and against `exit_b5b9_gated`.

## Addendum 3 (declared 2026-11-10 21:56 MST, commit e5bb5fc, before its runs)

**The remaining measurements the calibrated stack needs, reported either way (no claim).** All with the ImageNet-C
calibration (trunk 6, earlier 4, later 11), arm `rule_t6_b4b11_gated`, each file scored once; the new script
`scripts/cvpr/course_rule2.sh` (research repository) runs them so that the script of the main plan is not edited while its jobs run.

1. **Timing** (design timing cells, as `docs/heldout_test_plan_exit.md`): `time_batches.py` arms neo, pair, blocks 5 and 9, and
   `rule_t6_b4b11_k16_fp32_stats32`; expectation: the calibrated stack within 0.1× of the blocks-5-and-9 stack (1.47× NEO).
2. **Streams** (heldout4, EMA statistics): `run_streams.py --streams cont5 cleanret --arms neo pair_gated exit_b5b9_gated
   rule_t6_b4b11_gated`; expectation: a gain over the gated pair on cont5 of the same order as blocks 5 and 9 (+1.59), and
   clean segments of the clean-return stream within 0.2 of NEO.
3. **The other held-out slices**: `rule_t6_b4b11_gated` on ImageNet-C heldout, heldout2, heldout3 and on ImageNet-C-bar heldout3
   (tags `_rule`, `_cbar_rule`; baselines of those slices from the earlier runs); reported as the range over slices.
4. **Random routing of the calibrated stack** (heldout4 ImageNet-C and ImageNet-C-bar): `rule_t6_b4b11_random_gated`, the same
   branches and weights with 35 % of the images routed at random (seed 0); expectation: about half the method's gain, as for
   blocks 5 and 9.

## Addendum 4 (declared 2026-11-10 22:55 MST, commit 3cb70db, before its runs)

**The calibrated stack on the design split, for the comparison with the other TTA methods.** FOA, PEA, ETA and SAR were
run on the design cells (official code), so the calibrated stack (`rule_t6_b4b11_gated`, the ImageNet-C calibration) is run
there too, ImageNet-C (31 cells) and ImageNet-C-bar (21 cells), tags `_rule`, `_cbar_rule`, by `scripts/cvpr/course_rule3.sh`
of the research repository. Reported either way: the comparison table of the report (gradient-free: NEO, PEA, FOA, ours;
gradient-based: ETA, SAR, with and without the alignment the signal picks) and the design-versus-held-out gap of the
calibrated stack (simulation +1.21 / implemented design / held-out +1.36).

## Outcome (2026-11-11 01:10 MST; the thresholds applied as declared, each file scored once)

| claim | held-out file | arm | gain over `tok+place_b6` | threshold | clean − NEO | result |
|---|---|---|---|---|---|---|
| R1 | `forward_only_heldout4_vitb_rule` | `rule_t6_b4b11_gated` | +1.36 (68.35 vs 66.99) | ≥ +0.7 | +0.00 | holds |
| R2 | `forward_only_heldout4_vitb_cbar_rule` | `rule_t6_b4b11_gated` | +1.24 (70.23 vs 68.99) | ≥ +0.7 | +0.00 | holds |
| R3 | `forward_only_heldout3_vits_rule` | `rule_t6_b4b11_gated` (ViT-S/16's own calibration: trunk 6, earlier 4, later 11) | +1.67 (62.22 vs 60.56) | ≥ +0.5 | −0.05 | holds |
| R4 | `forward_only_heldout4_vitb_rule_small` | `rule_t5_b3b7_gated` | +1.17 (68.16 vs 66.99) | ≥ +0.7 | +0.00 | holds |
| R5 | `forward_only_heldout4_vitb_cbar_rule_small` | `rule_t7_b5b9_gated` | +1.37 (70.36 vs 68.99) | ≥ +0.7 | +0.00 | holds |

**Verdict: PASS** (all five claims hold).

Reported either way, as declared: the ImageNet-C-bar own calibration `rule_t7_b5b11_gated` 70.40 (+1.41 over the block-6 pair,
+0.79 over `tok+place_b7` at 69.61); every rule arm beats the labeled-search stack `exit_b5b9_gated` of the same slice
(ImageNet-C 68.03, ImageNet-C-bar 70.13, ViT-S/16 61.85) by +0.32, +0.10 / +0.27 and +0.38. Addendum 1 (ablation,
heldout4): earlier only +0.71 (ImageNet-C) / +0.60 (ImageNet-C-bar), later only +0.80 / +1.02, both +1.36 / +1.24;
ViT-S/16 earlier only +1.30, later only +0.61, both +1.67. Addendum 2: R4 and R5 above; the small-sample stacks route 36 % and
26 % of the images. The ViT-S/16 sweep picked trunk 6 (the most accurate block is 5, 61.05 vs 60.86) with Spearman +0.94
between the mean entropy and the accuracy of the pixel-plus-block versions, and the second opinions 4 and 11 (simulated +1.67
against +1.79 for the best labeled pair, 4 and 9). Addenda 3 and 4 (timing, streams, the other slices, the random-routing
control, the design split) are appended below as their runs finish.

**Addendum 3, outcome (2026-11-11 02:53 MST).** Timing (design timing cells): NEO 273 ms per batch, the pair 285 (1.04×),
blocks 5 and 9 402 (1.47×), the calibrated stack 412 (1.51×), the one extra block per routed image. Streams (heldout4):
the calibrated stack 61.72 on the continual stream of the 15 severity-5 cells against 59.67 for the gated pair (+2.05),
61.27 for blocks 5 and 9 and 53.76 for NEO; the clean segments of the clean-return stream 84.14 against 84.12 for NEO. The
other held-out image sets, gain over the pair of the same set: heldout +1.28, heldout2 +1.43, heldout3 +1.29 (heldout4
+1.36), ImageNet-C-bar heldout3 +1.31 (blocks 5 and 9: +0.87, +1.09, +0.94, +1.04; +1.25). Random routing of the same
share with the calibrated blocks: +0.51 on ImageNet-C and +0.51 on ImageNet-C-bar against +1.36 and +1.24 for entropy
routing.

**Amendment, 2026-11-19 (the pair rule).** The branch rule was changed after the held-out runs of the one-earlier-one-later
rule: the two branch blocks are now the pair of blocks other than the main block whose confidence-weighted average with the
main path has the lowest mean entropy on the unsure images, with no side constraint and no edge rule. Reason: on the saved
dev logits of the 16 model-shift pairs the side constraint neither helped nor hurt (simulated gain over the main path +1.49
without it against +1.44 with it, the same pick on 9 of 16 pairs), and it had no stated justification. The picks that changed
(DeiT-B/16 ImageNet-C and C-bar 5,8 -> 7,8; DeiT-S/16 ImageNet-C 3,5 -> 5,6 and C-bar 3,9 -> 5,9; ViT-B/16 Sketch 7,12 -> 1,7;
ViT-S/16 ImageNet-R 7,9 -> 9,11 and Sketch 5,11 -> 8,11; ViT-B/16 and ViT-S/16 ImageNet-C unchanged at 6 / 4, 11) were re-run
on their held-out cells (the forward-only run and ETA with the cascade inside), as were the per-cell few-image calibrations of
ViT-B/16 whose picks changed. The superseded run files are kept in runs/superseded_pair_rule/.
