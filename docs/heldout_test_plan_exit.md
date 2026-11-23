# Pre-registration — adaptive site choice per image: the gated early-exit stack (ViT-B/16, heldout4)

Declared 2026-11-10 03:53 MST (commit c66d432; the draft said 04:00, corrected to the commit time), before any image of `heldout4 = perm[42000:44000]` was decoded or passed through a model by any
run of this project. Everything below is frozen; the slice is scored once, for ImageNet-C and ImageNet-C-bar.

FROZEN_ARMS: neo tok place_b6 tok+place_b6 tok+place_b6+headcov pair_gated exit_b9_gated exit_b5b9_gated

## The method (fixed on the design split; `tokalign/aligner.py::forward_exit`, arms of `run_forward_only.py`)

The trunk is the v1 pair: the pixel map (TokAlign), blocks 1 to 6, the block-6 map (LayerAlign), blocks 7 to 12, NEO. The
spectral gate of the stack (thresholds 0.4 / 0.3, read from the pixel patches) switches the whole stack off on a batch it
judges clean (then only NEO acts), so the clean cell is handled by the gate. For each image the trunk's prediction entropy is
compared with theta = 1.5 nat. Images below theta are output as they are. Images above theta also run the extra branches, one
per extra site, and the branch outputs (trunk included) are averaged per image with the weights w_a ∝ exp(−H_a):
- `exit_b5b9_gated` (**the method**): extra sites block 5 (the saved pre-map tokens of block 5 are mapped with the running
  block-5 statistics of the whole batch and run through blocks 6 to 12 without the block-6 map) and block 9 (the unmapped
  block-6 tokens of the routed images run through blocks 7 to 9, are mapped with running block-9 statistics estimated from the
  routed images, and continue). Each branch has its own NEO running mean, estimated from the images that reach it.
- `exit_b9_gated` (the cheaper variant): the block-9 branch only.
- `pair_gated`: the trunk alone under the same gate (the v1 pair, gated), the reference for the gate's own effect.
No label, no gradient, no new hyperparameter beyond theta = 1.5 and the two extra sites, both chosen on the design split
(`scripts/analysis` scratch grids over all extra-site sets; the rule was the best mean design gain over ImageNet-C and
ImageNet-C-bar at a measured cost of at most 1.5 forward passes relative to NEO).

Design values (`results/scores/forward_only_design_vitb_exit.json`, `..._cbar_exit.json`; corrupted-cell means, clean cell):
ImageNet-C pair 67.47 (clean 82.80, NEO 82.95); exit_b9 68.24 (+0.77), exit_b5b9 68.51 (+1.04), exit_b5b9_gated 68.51 (clean
83.00), pair_gated 67.47 (clean 82.95); 34 % of the images routed. ImageNet-C-bar pair 68.73; exit_b9 70.07 (+1.34), exit_b5b9
70.28 (+1.55), exit_b5b9_gated 69.97 (+1.24; the gate closes on some mild ImageNet-C-bar cells, 25 % routed), pair_gated 68.68.
The implementation matches the logit simulation within 0.04 (ImageNet-C) and 0.12 (ImageNet-C-bar) points. Wall-clock on the
six timing cells (`timing_design_vitb_exit.json`, k = 16, float32): NEO 270 ms per batch, pair 281 ms (1.04x), exit_b9 334 ms
(1.24x), exit_b5b9 396 ms (1.47x), branch mixture 416 ms (1.54x).

## Claims under test (heldout4, each scored once)

- **T11a (ImageNet-C).** `exit_b5b9_gated` − `tok+place_b6` ≥ +0.5 points on the 30 heldout4 cells (design +1.04).
- **T11b (ImageNet-C-bar).** `exit_b5b9_gated` − `tok+place_b6` ≥ +0.7 points on the 20 heldout4 cells (design +1.24).
- **T11c (clean does not drop).** `exit_b5b9_gated` − `neo` ≥ −0.2 points on the clean cell of heldout4 (design +0.05; the
  same clean images serve both benchmarks, so this is one test).
- **Reported either way, with design expectations in brackets:** `exit_b9_gated` (≥ +0.4 on ImageNet-C, ≥ +0.6 on
  ImageNet-C-bar, clean ≥ −0.2), `pair_gated` against `tok+place_b6` (the gate's cost on corrupted cells), the routed
  fraction per cell, per-corruption gains, the v1 stack `tok+place_b6+headcov`, and the cost from the design timing run.

## Verdict rule

PASS if T11a, T11b and T11c hold; PARTIAL if one or two hold; FAIL otherwise. Thresholds, theta, the extra sites and the arm
list are fixed; no re-scoring, no further arm on heldout4. The report states the outcome as it is. If PASS, the gated early-exit
stack becomes the method of the report's v2 (the v1 pair stays as the fixed trunk and as a reported row); otherwise v1 stands
and the early-exit result is reported as it is.

## Runs

`scripts/cvpr/course_exit_heldout4.sh` (research repository) parts `inc`, `cbar`, which read FROZEN_ARMS from this file;
outputs `data/derived/course_two_site/{runs,scores}/forward_only_heldout4_vitb_exit.json` and `..._cbar_exit.json`, copied to
`results/scores/`.

## Outcome (2026-11-10 05:45; `heldout4` scored once, `results/scores/forward_only_heldout4_vitb_exit.json`, `..._cbar_exit.json`)

Runs 03:53 to 05:40 (ImageNet-C 65 min, ImageNet-C-bar 42 min). Corrupted-cell means, clean cell (the same 2,000 clean images for both):

| arm | ImageNet-C (30) | vs pair | ImageNet-C-bar (20) | vs pair | clean | vs NEO | routed |
|---|---|---|---|---|---|---|---|
| NEO | 62.09 | −4.90 | 67.11 | −1.88 | 84.10 | 0.00 | |
| +px | 64.11 | −2.88 | 67.95 | −1.04 | 84.10 | 0.00 | |
| +b6 | 65.12 | −1.87 | 68.58 | −0.41 | 84.15 | +0.05 | |
| +px+b6 (pair) | 66.99 | | 68.99 | | 84.30 | +0.20 | |
| +px+b6+hc (v1 stack) | 66.80 | −0.20 | 69.14 | +0.15 | 83.50 | −0.60 | |
| pair, gated | 66.99 | 0.00 | 69.03 | +0.04 | 84.10 | 0.00 | |
| exit_b9, gated | 67.64 | +0.65 | 69.94 | +0.94 | 84.10 | 0.00 | 0.35 / 0.26 |
| **exit_b5b9, gated (the method)** | **68.03** | **+1.04** | **70.13** | **+1.14** | 84.10 | 0.00 | 0.35 / 0.26 |

**T11a +1.04 ≥ +0.5, T11b +1.14 ≥ +0.7, T11c 0.00 ≥ −0.2: Verdict PASS.** The design values were +1.04 / +1.24 / +0.05.
The cheaper variant meets its expectations too (+0.65 ≥ +0.4, +0.94 ≥ +0.6, clean 0.00). The gate closed every clean batch
(nothing routed, clean = NEO) and cost nothing on the corrupted cells (pair gated 66.99 / 69.03 against 66.99 / 68.99). The
v1 stack with the head covariance stage is below the pair on ImageNet-C and loses 0.60 on the clean cell, as on heldout3.
Per the rule, the gated early-exit stack becomes the method of the report's v2, with the pair as its trunk.

After the verdict, the gated early-exit arms were also run on the three earlier ImageNet-C held-out slices (`heldout`,
`heldout2`, `heldout3`) and on ImageNet-C-bar `heldout3`, to report the range over slices as Table S1 does for the pair; these
runs select nothing (`scripts/cvpr/course_exit_slices.sh`).

## Addendum (2026-11-10 12:58, commit ee859ec, declared before the runs; the draft said 08:20, corrected to the commit time): equal-compute controls and a second backbone

A review of the v2 report asked whether the gain comes from choosing which images get the extra branches or from the extra
compute alone, and for a second backbone. The following controls are run once on `heldout4` (both benchmarks) with the arms
of `run_forward_only.py` named here, and compared with `exit_b5b9_gated` of the same files; the ViT-S/16 runs use the sites
and threshold chosen on ViT-B/16 without any new choice.

- `exit_b5b9_random_gated`: the same branches and weights, but a fixed 35 % of the images chosen at random (seed 0) instead of
  by the trunk's entropy. Expectation (design, `site_choice_design_vitb*.json` controls): at least 0.3 points below the method
  on ImageNet-C (design +0.54 against +1.08) and on ImageNet-C-bar (+0.84 against +1.68).
- `exit_b5b9_equal_gated`: entropy routing, equal weights. Expectation: within 0.2 points of the method (design −0.04 / −0.09).
- `mix_b5b9_gated`: every image routed (the full three-branch mixture, 2.08 forward passes of the pair). Expectation: above the
  method by at most 0.4 points (design +0.14 / +0.32) at about 1.5 times its extra cost.
- ViT-S/16: `tok+place_b6 pair_gated exit_b9_gated exit_b5b9_gated` on the design split and `pair_gated exit_b9_gated
  exit_b5b9_gated` on `heldout3` (the ViT-S/16 slice of Table 2, whose baselines exist), stats `vits_blocks.pt`. Expectation:
  `exit_b5b9_gated` − `tok+place_b6` ≥ +0.5 on the 30 heldout3 cells, clean against NEO ≥ −0.2. Reported either way.

Design values of the controls on ViT-B/16 (`site_choice.py`): ImageNet-C pair 67.47; entropy routing 68.55 (+1.08), equal
weights 68.51, random routing 68.01 (+0.54, mean of five draws), the most confident images routed instead 67.57 (+0.10), full
mixture 68.69 (+1.22), block 9 only 68.25, block 5 only 67.82, oracle routing of the trunk's errors 70.39. ImageNet-C-bar pair
68.73; 70.40 / 70.32 / 69.56 / 68.86 / 70.73 / 70.19 / 69.18 / 71.99.

## Addendum 2 (2026-11-10 13:05, declared before the run): the continual and clean-return streams

`run_streams.py` now runs the second-opinion stack (arm `exit_b5b9_gated`: the same sites, threshold and gate, with the EMA
statistics of the stream arms: 0.2 for the site statistics, 0.5 for every NEO mean). One run on `heldout4` with the arms
`neo pair_gated exit_b5b9_gated` and the streams `cont5` (the 15 severity-5 cells in order, no resets) and `cleanret` (clean
and corrupted segments alternating), scored once. Expectation, reported either way: `exit_b5b9_gated` − `pair_gated` ≥ +0.5 on
`cont5`, and its clean segments of `cleanret` within 0.3 points of NEO.

## Outcome of addendum 1 (2026-11-10 14:45; each arm scored once)

Equal-compute controls on `heldout4` (`results/scores/forward_only_heldout4_vitb{,_cbar}_exitctl.json`), gains against the pair
of the same slice; the method is `exit_b5b9_gated` (+1.04 ImageNet-C, +1.14 ImageNet-C-bar):

| arm | ImageNet-C | vs method | ImageNet-C-bar | vs method | clean vs NEO | routed (C) |
|---|---|---|---|---|---|---|
| random routing, same share (`exit_b5b9_random_gated`) | +0.50 | −0.54 | +0.49 | −0.65 | 0.00 | 36 % |
| entropy routing, equal weights (`exit_b5b9_equal_gated`) | +1.04 | +0.01 | +1.10 | −0.04 | 0.00 | 35 % |
| every image routed (`mix_b5b9_gated`) | +1.36 | +0.32 | +1.40 | +0.26 | 0.00 | 100 % |

All three expectations hold: random routing is 0.54 / 0.65 below the method (≥ 0.3), equal weights within 0.2, the full
mixture at most 0.4 above. ViT-S/16 with the ViT-B/16 sites and threshold (`forward_only_{design,heldout3}_vits_exit.json`): design
+1.29 over the pair, heldout3 +1.29 (≥ +0.5), clean −0.05 against NEO (≥ −0.2), 42 % routed; block 9 only +0.61.

## Outcome of addendum 2 (2026-11-10 15:20; `results/scores/streams_heldout4_vitb_exit.json`, scored once)

Continual stream `cont5`: NEO 53.76, pair gated 59.67, second-opinion stack 61.27 (+1.59 over the pair, ≥ +0.5; 46 % of the
images routed). Clean-return stream: clean segments NEO 84.12, pair 84.14, stack 84.14 (+0.02 against NEO, within 0.3; 21 %
routed over the whole stream). Both expectations hold.
