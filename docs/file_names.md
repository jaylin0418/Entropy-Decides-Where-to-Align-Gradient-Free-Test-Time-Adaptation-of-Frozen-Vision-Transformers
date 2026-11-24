# File names at the time of the runs

The run and score scripts were ported from the research codebase with their computation unchanged. The run records and
the documents in `docs/` use the names on the right; the arm names in the code keep the prefixes `exit_` (the
labeled-search cascade) and `rule_` (the calibrated cascade), and `trunk` in the score files means the main block.

| now | at the time of the runs (research codebase) |
|---|---|
| `tokalign/aligner.py` | `scripts/spa/probe_online.py` (`Aligner`, `Running`, `sqrt_psd`, `load_source`) |
| `scripts/episodic/run_forward_only.py`, `score_forward_only.py` | `scripts/spa/probe_online.py`, `score_probe.py` |
| `scripts/episodic/run_gradient_host.py`, `score_gradient_host.py` | `scripts/spa/probe_eta.py`, `score_eta.py` |
| `scripts/episodic/run_foa.py`, `score_foa.py`, `run_pea.py`, `score_pea.py` | `scripts/spa/probe_foa.py`, `score_foa.py`, `probe_pea.py`, `score_pea.py` |
| `scripts/streams/run_streams.py`, `score_streams.py` | `scripts/spa/probe_streams.py`, `score_streams.py` |
| `scripts/cost/time_batches.py`, `score_timing.py` | `scripts/spa/bench_wallclock.py`, `score_bench.py` |
| `scripts/compute_source_stats.py` (`--blocks`) | `scripts/spa/source_stats.py` |
| `scripts/block_alignment/{run_block_alignment,score_block_alignment}.py` | `scripts/regprobe/{probe_a,score_a}.py` |
| `results/scores/forward_only_design_vitb`, `eta_design_vitb`, `sar_design_vitb` | `results/spa/spa1.json`, `spa2.json`, `spa2_sar.json` |
| `results/scores/foa_design_vitb`, `foa_design_extended_vitb` | `results/spa/spa3.json`, `spa3_n6000.json` |
| `results/scores/pea_design_vitb{,_src20k}` | `results/spa/spa2_pea{,_src20k}.json` |
| `results/scores/timing_design_vitb` | `results/spa/bench_summary_e5c.json` |
| `results/scores/forward_only_design_vitb_ablations`, `patch_filters_design_vitb` | `results/spa/spa1_e6.json`, `results/wiener/wr1.json` |
| `results/scores/block_alignment_design_vitb` | `results/regprobe/probe_a.json` |
| the where-to-align, labeled-search, calibrated-cascade and model-dataset runs | made with this repository's scripts (research job scripts `scripts/cvpr/course*.sh`), outputs under `data/derived/course_two_site/` |
| `--split heldout` / `heldout2` / `heldout3` / `heldout4` | the first to fourth held-out sets of the report |

## The code's terms against the report's

| in the code, docs/ and the score files | in the report |
|---|---|
| `two_site`, `base path`, `pair`, `pair_gated`, `tok+place_b<k>` | the main path (TokAlign, LayerAlign after the main block, NEO, gated) |
| `trunk`, `t<b>` | the main block b |
| `second opinion`, `so`, `b<e>b<l>`, `shallower` / `deeper` | the two extra branches after blocks e < l (a1 < a2 in the report) |
| `routed`, `routed_fraction`, `exit_theta` | the unsure images, their share, and the threshold theta |
| `exit_` arms (`exit_b5b9_gated`) | the cascade with the blocks of the labeled dev search (5 and 9), a comparison only |
| `rule_` arms (`rule_t6_b4b11_gated`) | the calibrated cascade |
| `mix_` arms | the control that runs the branches on every image |
| `hier` | the two-level map of Section G, not kept |
| `design` | the dev images |
| `mean_30_cells` | the mean over the cells of the file, whatever their number (20 on ImageNet-C-bar, 3 on the natural shifts) |
| `data/derived/course_two_site`, `scripts/cvpr/course*.sh`, GB10 | the directory, job scripts and machine of the research repository where the runs were made, named in the run records and in results/number_inventory.csv, not part of this repository |
