# Pre-registration — dynamic use of the two sites: the block-6 branch mixture and the warm-up gate (ViT-B/16)

> Table, figure and section numbers in this plan are those of the report draft at the time it was declared. The plan is kept as declared, and outputs it names that the report does not use are not in this repository.

Declared 2026-11-09 21:50 MST (commit 01bdedd; the draft said 22:10, corrected to the commit time), before any held-out logit was read by the rules below. Everything is frozen; the held-out
slices are scored once, from the per-image logits that the earlier held-out runs saved (no new forward pass, no re-run).

## Motivation (design split only)

The fixed pair (pixel site + block 6) is within 0.27 points of the per-cell oracle over the four arms {NEO, +px, +b6, both}
on the 30 ImageNet-C design cells (67.47 against 67.73), so no per-cell choice of a site can gain much there. Per image the
arms are complementary: some arm is right on 75.1 % of the design images while the pair is right on 67.5 %. On ImageNet-C-bar
the per-cell oracle is 1.4 points above the pair (70.15 against 68.73) and the pair trails a single site on 11 of 20 cells.

## The methods (all label-free; fixed on the design split)

- **Branch mixture (the method).** One pass: the pixel map, blocks 1 to 6, then a branch after block 6: one copy of the
  tokens goes through the block-6 map (LayerAlign) and blocks 7 to 12, the other through blocks 7 to 12 unchanged; NEO acts on
  each head input with its own running mean. The two softmax outputs are averaged per image with the weights
  w_a ∝ exp(−H_a), H_a the prediction entropy of branch a on that image. Cost: 1.5 forward passes of the frozen network
  (blocks 1 to 6 once, blocks 7 to 12 twice). Arms of the saved runs: `tok` and `tok+place_b6`.
- **Three-arm mixture.** The same weights over `tok`, `place_b6`, `tok+place_b6` (2.5 passes). **Four-arm mixture** adds `neo`
  (3 passes). Both reported.
- **Warm-up gate (the selection alternative).** The four arms run on the first K = 3 batches (192 images) of a cell; the arm
  with the lowest mean prediction entropy over those batches is used for the remaining batches; the pair's predictions are
  output during the warm-up. Equivalent to two conditional gates (pixel site judged with block 6 on, block 6 judged with
  the pixel site on). Cost: 4 passes on 3 of 32 batches, 1 pass afterwards (1.28 passes).
- Plain averages (equal weights) are reported next to the entropy weights.

Implementation of the evaluation: `scripts/analysis/site_mixture.py` (reads the saved logits, batch size 64, split order).
Design values (`results/scores/site_mixture_design_vitb{,_cbar}.json`), gains over the pair on the corrupted cells, clean
delta against the pair in brackets: ImageNet-C branch +0.62 (+0.10), three arms +0.77 (+0.10), four arms +0.73 (+0.15),
warm-up gate −0.03 (+0.10); ImageNet-C-bar branch +1.18, three arms +1.65, four arms +1.73, warm-up gate +0.97.

## Claims under test (heldout3, scored once)

- **T10a (ImageNet-C).** Branch mixture − pair ≥ +0.3 points on the 30 heldout3 cells, and clean delta against the pair
  ≥ −0.2 points.
- **T10b (ImageNet-C-bar).** Branch mixture − pair ≥ +0.5 points on the 20 heldout3 cells.
- **Reported either way, with design expectations in brackets:** three-arm mixture (≥ +0.5 on ImageNet-C, ≥ +1.0 on
  ImageNet-C-bar), four-arm mixture, equal-weight averages, the warm-up gate (≥ −0.1 on ImageNet-C, ≥ +0.5 on
  ImageNet-C-bar), the per-cell oracle, per-corruption gains, and the branch mixture on the two other held-out slices of
  ImageNet-C (`heldout`, `heldout2`, from the slice-variance runs) as a range.

## Verdict rule

PASS if T10a and T10b hold; PARTIAL if exactly one holds; FAIL otherwise. Thresholds, K, tau and the arm sets are fixed;
no re-scoring, no further rule on these slices. The report states the outcome as it is. If the verdict is PASS, the branch
mixture is implemented as one arm of `run_forward_only.py` (shared blocks 1 to 6, two heads) and its wall-clock cost
measured on the design split; the held-out accuracy of that implementation must equal the value computed here on the same
images, otherwise the implementation is reported as a discrepancy.

## Files

Logits: `data/derived/course_two_site/runs/logits/forward_only_{heldout3,heldout,heldout2}_vitb*_{arm}_{cell}.npz` of the
runs already scored for the report. Outputs: `results/scores/site_mixture_heldout3_vitb.json`,
`site_mixture_heldout3_vitb_cbar.json`, `site_mixture_heldout_vitb.json`, `site_mixture_heldout2_vitb.json`.

## Outcome (2026-11-09 21:52, `scripts/analysis/site_mixture.py` on the saved held-out logits, run once)

Gains over the pair on the corrupted cells, clean delta against the pair in brackets (`results/scores/site_mixture_*.json`):

| slice | pair | branch mixture (1.5 passes) | three arms (2.5) | four arms (3) | warm-up gate K=3 | per-cell oracle |
|---|---|---|---|---|---|---|
| ImageNet-C heldout3 (30 cells) | 67.99 | +0.57 (−0.25) | +0.90 (−0.10) | +0.85 (−0.10) | −0.03 (0.00) | +0.26 |
| ImageNet-C heldout (30) | 67.72 | +0.61 (−0.25) | +0.83 (−0.10) | +0.81 | +0.03 | +0.36 |
| ImageNet-C heldout2 (30) | 69.32 | +0.79 (+0.10) | +1.06 (+0.20) | +1.05 | +0.01 | +0.20 |
| ImageNet-C-bar heldout3 (20) | 69.84 | +1.03 (−0.25) | +1.69 (−0.10) | +1.70 | +0.83 (0.00) | +1.33 |

**T10a: the gain holds (+0.57 ≥ +0.3) but the clean delta of the branch mixture is −0.25 against the bound −0.2 (five
images of 2,000; the clean cells of the three slices are disjoint: −0.25, −0.25 and +0.10), so T10a fails by the letter.** Against NEO,
the metric of the report, the branch mixture's clean cell is +0.15, −0.20 and −0.05 on the three slices (−0.05 on the design
split), while the pair's is +0.40, +0.05 and −0.15: the bound chosen here was tighter than the clean-cell variation of the pair itself. T10b holds (+1.03 ≥ +0.5).
**Verdict PARTIAL.** The branch mixture stays above NEO on the clean cell (85.85 against 85.70). The three-arm mixture
meets every expectation stated above (+0.90 / +1.69, clean −0.10) at 2.5 passes. The warm-up gate behaves as on the design
split: nothing on ImageNet-C, +0.83 on ImageNet-C-bar, where it chooses block 6 alone on caustic refraction 5,
checkerboard cutout, inverse sparkles and sparkles 5, the pixel site alone on plasma noise 5, and NEO on caustic
refraction 3. Equal weights are within 0.1 of the entropy weights everywhere. The branch mixture is below the pair on 12
of the 30 ImageNet-C cells (largest: elastic transform 5 −1.8, fog 3 −1.3, brightness 5 −1.3) and on 3 of 20
ImageNet-C-bar cells (at most −0.3). The implementation of the branch as one arm of `run_forward_only.py` and its timing
follow as planned; the report states the clean delta as measured.

## Implementation check and cost (2026-11-10 02:20; `branch` arm of `run_forward_only.py`, GPU runs 01:38 to 02:18)

The implemented branch (blocks 1 to 6 once, blocks 7 to 12 on the doubled batch, one NEO mean per branch, entropy mixture)
reproduces the simulation: on every one of the four runs (ImageNet-C design and heldout3, ImageNet-C-bar design and heldout3)
the unmapped branch's predictions equal the `tok` arm's and the mapped branch's equal the `tok+place_b6` arm's for 100 % of
the images, and the mixture's prediction equals the simulated mixture for 99.96 to 99.97 % of the images (float differences
of the batched blocks 7 to 12). Corrupted-cell means: 68.09 / 68.56 / 69.91 / 70.87 against 68.09 / 68.56 / 69.91 / 70.86
simulated (`results/scores/forward_only_{design,heldout3}_vitb_branch.json`, `..._cbar_branch.json`). Wall-clock on the six
timing cells at k = 16, float32 eigendecompositions and statistics (`timing_design_vitb_branch.json`): NEO 270.8 ms per batch,
pair 281.7 ms (1.04x), branch 416.8 ms (1.54x NEO, 1.48x the pair; the plan said 1.5 forward passes).
