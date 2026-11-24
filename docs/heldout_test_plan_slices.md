# Pre-registration — slice variance of the two-site stack (ViT-B/16, two further held-out slices)

> Table, figure and section numbers in this plan are those of the report draft at the time it was declared. The plan is kept as declared.

Declared 2026-11-09 18:58 MST, written before the queue was launched (18:58:48); no run had started. Everything below is frozen; each slice is scored once.

## Purpose

Every held-out number of the report's Table 2 is one run on one 2,000-image slice (`heldout3 = perm[40000:42000]`). This test
repeats the same arms on the two other held-out slices of the project, `heldout = perm[36000:38000]` and
`heldout2 = perm[38000:40000]`, so that the report can state each held-out gain as a mean and a range over three disjoint
slices. Neither slice was used for any choice of the two-site study: `heldout` carried the pixel-site study's ViT-B/16 test
and `heldout2` its ViT-S/16 test, both scored once after every setting was fixed on `design`, and no ViT-B/16 two-site arm has
touched either. Nothing is chosen on these images; block 6, λ, the priors and the gate thresholds are those of the report.

## Runs

`scripts/cvpr/course_slices.sh` (research repository) parts `frozen_h1`, `eta_h1`, `frozen_h2`, `eta_h2`: the heldout3 arms of
`docs/heldout_test_plan_two_site.md`, forward-only `neo tok headcov tok+headcov place_b6 tok+place_b6 tok+place_b6+headcov`
and ETA `eta tok+eta eta+place_b6 tok+eta+place_b6` (with NEO and head-covariance variants at scoring), ViT-B/16, FP32, batch
64, 30 ImageNet-C cells plus clean, stats `vitb_blocks.pt`, tag `_two_site`. Scored files copied into `results/scores/` as
`forward_only_{heldout,heldout2}_vitb_two_site`, `eta_{heldout,heldout2}_vitb_two_site`. The `neo`, `tok`, `headcov`,
`tok+headcov` and the ETA arms of `heldout` repeat runs of the pixel-site study; they are expected to reproduce the committed
files exactly (as the SAR arms did) and the comparison is recorded.

## Claims (each slice scored once)

- **T8a.** `tok+place_b6` − `neo` ≥ +3.0 pt on each of the two slices (heldout3: +4.88).
- **T8b.** `tok+place_b6+headcov` > `tok+headcov` on each slice (heldout3: +2.38).
- **T8c.** `tok+eta+place_b6+neo` − `tok+eta+neo` ≥ +1.0 pt on each slice (heldout3: +2.47).

Reported either way: every arm of Table 2 as mean and range (min to max) over the three slices, the per-slice values in the
supplement, clean harm per slice. Verdict PASS if T8a–T8c hold on both slices, PARTIAL if at least one claim holds on both,
FAIL otherwise. Thresholds fixed; no re-scoring; no further arms on these slices. The report states the outcome as it is and
the heldout3 numbers stay as reported.

## Outcome (2026-11-09 22:25, each slice scored once; `results/scores/{forward_only,eta}_{heldout,heldout2}_vitb_two_site.json`)

Runs: frozen_h1 46 min, eta_h1 53 min, frozen_h2 46 min, eta_h2 53 min (GPU, 2026-11-09 18:58 to 22:23). The four
forward-only arms shared with the pixel-site study (`neo`, `tok`, `headcov`, `tok+headcov` on `heldout`) reproduced
`forward_only_heldout_vitb.json` exactly, every cell. The six ETA arms shared with `eta_heldout_vitb.json` reproduced its
30-cell means to two decimals; 18 of 186 cells differ by 0.05 (one image) and one by 0.10 (the gradient host is not bitwise
deterministic), recorded by `tests/test_slices.py`.

| claim | heldout3 (reported) | heldout | heldout2 | bound | result |
|---|---|---|---|---|---|
| T8a `tok+place_b6` − `neo` | +4.88 | +4.90 | +4.70 | ≥ +3.0 | holds on both |
| T8b `tok+place_b6+headcov` − `tok+headcov` | +2.38 | +2.19 | +2.36 | > 0 | holds on both |
| T8c `tok+eta+place_b6+neo` − `tok+eta+neo` | +2.47 | +2.20 | +2.38 | ≥ +1.0 | holds on both |

**Verdict PASS.** Clean cell against NEO: the two sites +0.40, +0.05, −0.15; the two-site stack −1.15, −0.50, −0.65 (the head
covariance stage, as on heldout3); inside ETA (against ETA+NEO) +0.15, −0.15, −0.05. Report: Table 2 caption and
Supplement C (Table S1, every ViT-B/16 arm of Table 2 on the three slices with mean and range); the heldout3 numbers stay as
reported. Macros `ts.sl.*` (`paper/make_numbers.py`), loader key `two_site/slices_vitb`, steps E30–E33 and R35–R38 of
`reproduce_paper.sh`.
