# Pre-registration — three supplementary tests of the two-site stack (ViT-B/16)

Declared 2026-11-09 14:00 MST, before any of the runs below. Everything is frozen; held-out images are scored once. The
method is the pooled two-site stack of `docs/heldout_test_plan_two_site.md` (TokAlign at the pixels, LayerAlign after
block 6, NEO, head covariance, spectral gate), unchanged. These tests answer three questions a reviewer would ask: does the
pair transfer to corruptions it was never chosen on, does it add to a gradient host other than ETA, and what does it cost.

## Test 5: corruptions never seen (ImageNet-C-bar)

Data: the authors' released ImageNet-C-bar (Mintun et al., NeurIPS 2021), ten corruptions disjoint from ImageNet-C
(`tokalign/data.py: CBAR_CORRUPTIONS`), severities 3 and 5, so 20 cells, with the same image indices as the course splits:
design `perm[2000:4000]`, held-out `heldout3 = perm[40000:42000]`, plus the clean cell. Arms: `neo`, `tok`, `place_b6`,
`tok+place_b6` (forward-only, `run_forward_only.py --cells <CBAR_CELLS>`; scoring unchanged, the mean is over the 20 cells).
No choice is made on these images: block 6, λ, the priors and the gate thresholds are those of the report.

Prediction. The pixel site helps on the noise-like corruptions (blue noise, brownish noise, plasma noise, perlin noise,
single-frequency greyscale), block 6 on the structured ones (caustic refraction, sparkles, inverse sparkles, checkerboard
cutout, cocentric sine waves), and the pair beats either site. Because these shifts are not in the design sweep, the gain
should be smaller than on ImageNet-C (+5.1 design, +4.9 held-out over NEO).

Design bar (before the held-out run): on the 20 design cells, `tok+place_b6` ≥ NEO + 1.0 pt and ≥ max(`tok`, `place_b6`) +
0.3 pt. If it fails, the held-out run is not started and the design result is reported as a negative result.

**T5 (heldout3, scored once):** `tok+place_b6` ≥ NEO + 1.0 pt on the 20 cells and ≥ the better single site. Reported
either way: per-corruption gains, `place_b6` alone, clean harm, and the comparison with the design values.

## Test 6: a second gradient host (SAR)

Arms of `run_gradient_host.py --method sar`: `sar`, `tok+sar`, `sar+place_b6`, `tok+sar+place_b6` (SAR with NEO applied
at scoring, as for ETA), ViT-B/16, 30 ImageNet-C cells plus clean. Design bar: `tok+sar+place_b6` ≥ `tok+sar` + 0.3 pt on
the 30 design cells and clean harm against `sar` ≥ −0.5 pt; if it fails, the held-out run is not started.

**T6 (heldout3, scored once):** `tok+sar+place_b6` ≥ `tok+sar` + 0.3 pt on the 30 cells (the margin of T4 for ETA).
Reported either way: `sar+place_b6` alone, per-family gains, clean harm.

## Test 7: wall-clock cost of the two-site stack

`scripts/cost/time_batches.py` on the six timing cells of the report's cost table (design images, no labels), arms `neo`,
`stack_k16_fp32_stats32` (the pixel-site stack of the cost table), `twosite_k16_fp32_stats32` (the two-site stack at the
same refresh settings), `twosite_k1`, `eta`, `eta+twosite`. No claim is tested. Expectation: the two-site stack at k=16 runs
within 1.35 × NEO's time (the pixel-site stack is 1.18 ×, ETA 2.41 ×). The accuracies of the timed variants are scored from
the saved logits and reported next to the times.

## Verdict rule

T5 and T6 are judged separately, each PASS or FAIL by its own margin; no re-scoring, no additional arms on `heldout3`. The
report states every outcome as it is. Run order: `scripts/cvpr/course_extra.sh` (research repository) parts `timing`,
`sar_design` (with the bar), `cbar_design` (waits for the data marker, then the bar), `cbar_heldout`, `sar_heldout`;
outputs under `data/derived/course_two_site/` with the tag `_extra` (timing) or `_cbar`; scored files copied into
`results/scores/` as `timing_design_vitb_extra`, `sar_design_vitb`, `sar_heldout3_vitb`, `forward_only_design_vitb_cbar`,
`forward_only_heldout3_vitb_cbar`.

## Outcome (recorded as the parts finish; design bars applied mechanically by `scripts/cvpr/course_extra_bar.py`)

- **T7 (timing, 2026-11-09 14:10).** `timing_design_vitb_extra`: NEO 266 ms/batch; pixel-site stack k=16 f32 stats32 311 ms
  (1.17×, 64.92 %); two-site stack k=16 f32 stats32 315 ms (1.19×, 65.93 %); two-site k=1 f64 540 ms (2.03×, 66.37 %); ETA
  638 ms (2.40×, 61.16 %); ETA with the two sites 879 ms (3.30×, 66.55 %). Within the 1.35× expectation. Report Table S7.
- **T6 design bar (2026-11-09 16:00).** `sar_design_vitb` (the six arms of the earlier SAR run came out identical to the
  committed file; the scored file was replaced by the twelve-arm one): `tok+sar+place_b6` 66.76 vs `tok+sar` 62.69, +4.07
  (bar ≥ +0.3: passes); clean harm against `sar` +0.20 (bar ≥ −0.5: passes). **Design bar PASS**; the `heldout3` SAR run is
  queued (part `sar_heldout`, after the ImageNet-C-bar parts). With NEO: 62.81 / 64.72 / 65.91 / 67.68 (+2.95 over
  `tok+sar+neo`; clean −0.30). Per family (+NEO, two sites over `tok+sar+neo`): noise +0.2, blur +4.5, weather +2.4,
  digital +4.0. Report Table S10 and Section 3.
- The job script's `sar_design` part finished its work but its completion marker failed (the script file was edited while
  the part was running, and bash read the edited file); the queue logged the part as FAILED for that reason only.
- **T6 (heldout3, 2026-11-09 17:50, scored once).** `sar_heldout3_vitb`: `tok+sar+place_b6` 67.18 vs `tok+sar` 63.54, **+3.63
  (≥ +0.3: PASS)**; clean 85.20 vs `sar` 85.45 (−0.25). `sar+place_b6` alone 65.06 (+3.92 over `sar`). With NEO: 63.47 / 65.22 /
  66.42 / 68.21 (+2.98 over `tok+sar+neo`; clean +0.15). Per family (+NEO, two sites over `tok+sar+neo`): noise +0.1, blur
  +4.3, weather +2.8, digital +4.0. Report Table S10 (lower block), Section 3 and the contributions use the held-out gain.
- **T5 design bar (2026-11-09 18:07).** `forward_only_design_vitb_cbar` (20 cells): `tok+place_b6` 68.73 vs `neo` 66.75, +1.98
  (bar ≥ +1.0: passes); vs best single `place_b6` 68.38, +0.35 (bar ≥ +0.3: passes). **Design bar PASS**; the `heldout3` run
  started. The first attempt of this part failed before any cell was scored because the archive extracts to
  `extracted/cbar_download/`; re-run from `scripts/cvpr/course_cbar.sh` with that root.
- **T5 (heldout3, scored once).** `forward_only_heldout3_vitb_cbar` (18:24): `tok+place_b6` 69.84 vs `neo` 67.85, **+1.98 (≥ +1.0)**; vs the
  better single site `place_b6` 69.54, **+0.29 (≥ 0)**: **PASS**. Clean +0.40. Per corruption (mean of severities 3 and 5) the
  pair is the best arm on 5 of 10 (blue noise, brownish noise, concentric sine waves, Perlin noise, single-frequency greyscale),
  below the better single site on 5 (caustic refraction −0.20, checkerboard cutout −1.43, inverse sparkles −6.48, plasma noise
  −4.27, sparkles −0.38) and below NEO on checkerboard cutout. The prediction held for plasma noise and single-frequency
  greyscale (pixel site) and for caustic refraction, sparkles and inverse sparkles (block 6); blue, brownish and Perlin noise
  favour block 6, concentric sine waves the pixel site, and checkerboard cutout is hurt by both maps. Design +1.98 / +0.35,
  held-out +1.98 / +0.29. Report Table S11, Supplement G and Section 3.
