# Results provenance

Which score files and which steps of `reproduce_paper.sh` stand behind every table and figure of the report.

`results/number_inventory.csv` lists every number printed in the report (one row per occurrence): its macro key in
`paper/numbers.tex`, the `results/paper_numbers.json` path or formula, the scored file and command that produced it, the
steps of `reproduce_paper.sh` that regenerate it, and its status. Per table and figure:

The held-out plans in this directory quote the numbers of their time: the ImageNet-C-bar gain against the main path
without its gate (+1.24, 70.23 against 68.99) and the first timing run (412 ms, 1.51x, NEO 273 ms). The report uses the
gated main path of the same run (+1.20, 69.03) and the later timing run (405 ms, 1.49x, 271 ms, step C6).

| table / figure | `paper_numbers.json` keys | scores (`results/scores/`) | steps |
|---|---|---|---|
| Table 1, the calibrated cascade on `heldout4` (with its ablations and the random control); Table 2, the calibration from few images (from the saved dev predictions, and the small-sample held-out runs) | `rule/held4/*`, `rule/random`, `rule/timing`, `rule/fewshot/*`, `exit/held4/*` | `forward_only_heldout4_vitb_{rule,cbar_rule,rule_abl,cbar_rule_abl,rule_small,cbar_rule_small,rule_random,cbar_rule_random,exit,cbar_exit}`, `forward_only_heldout4_vitb_fewshot{,_k10,_k1}_<cell>`, `timing_design_vitb_ablations` | S5 E38 E39 E52–E57 E64 E65 E69 C6 → R39 R45 R46 R56 R64 → P1 P2 P4 |
| Table 3, across models and shifts (the 3 x 3 grid; ETA with the main path and with the cascade inside) | `two_site/held3_*`, `rule/*`, `ca/*`, `ca/etastack/*` (computed in `paper/make_numbers.py across()`) | `forward_only_heldout3_{vitb,vits}_rule`, `eta_heldout3_{vitb,vits}{,_stack}`, `forward_only_heldout4_vitb_cbar_rule`, `*_<model>_ca_<dataset>{,_stack}`, `*_<model>_cbar_ca{,_stack}` | as Table S1 and the rule files |
| Figure 1, the depth sweep and the entropy | `two_site/sweep_vitb`, `exit/site_choice/{inc,cbar}/label_free` | `forward_only_design_vitb_{two_site,sweep}`, `eta_design_vitb_sweep`, `site_choice_design_vitb{,_cbar}` | S5 E19 E24 E25 E34 E35 → R22 R24 R25 A6 A7 → P1 P2 |
| Figure 2, the method (calibration panel from `site_choice_design_vitb`) | `exit/site_choice/inc/label_free` | `site_choice_design_vitb` | A6 → P2 |
| Figure S1 and Algorithm S1, one batch through the cascade | images and gate scores recomputed from the cached dev images; inset from the saved logits (`paper/figures/example_numbers.json`) | `forward_only_design_vitb_exit` | E36 → R39 → P5 |
| Figure S2, the sweep by family; Section C, the main path per corruption | `two_site/sweep_vitb`, `two_site/held3_vitb/per_cell_*` | as Figure 1; `forward_only_heldout3_vitb` | E20 → R22 → P2 |
| Figure S3 and Table S1, the entropy across models and shifts; the natural shifts in full | `exit/site_choice/*`, `rule/vits`, `ca/*` | `site_choice_design_{vitb,vitb_cbar,vits_rule,<model>_ca_<dataset>}`, `forward_only_{design,heldout}_<model>_ca_<dataset>`, `eta_heldout_<model>_ca_<dataset>`, `calib_<model>_<dataset>` | E58 E66 E67 E68 → A11 A13 R47 R53 R54 R55 → P1 P2 |
| Figure S4, the drift by depth | `two_site/{drift_vitb,sweep_vitb}` | `drift_by_depth_design_vitb` | F9 → R26 → P1 P2 |
| Section C, one block per shift against per-cell rules and oracles | `two_site/{design_vitb,selection_vitb}`, `exit/site_choice/inc` | `forward_only_design_vitb_two_site`, `site_selection_design_vitb`, `site_choice_design_vitb` | E19 F9 → R22 R26 R27 A6 → P1 P2 |
| Section D, what the calibration picks; ViT-S/16 on its own images | `rule/picks`, `exit/site_choice/*`, `rule/vits` | `site_choice_design_vitb{,_cbar,_cbar_t7}`, `forward_only_design_vits_rule_sweep`, `site_choice_design_vits_rule{,_t6}`, `forward_only_heldout3_vits_rule` | S6 E58 E59 → A6 A7 A11 A12 R47 R48 → P1 P2 |
| Figure S5, why these two blocks; the labelled search | `exit/site_choice/*/second_opinions`, `exit/blocks/*` | `site_choice_design_vitb{,_cbar}`, `forward_only_design_vitb_{select,cbar_select}` | E34 E35 → A6 A7 → P1 P2 |
| Figure S6, how many images the calibration needs | `rule/sample_size/{inc,cbar}` | `rule_sample_size_design_vitb{,_cbar}` | A9 A10 → P1 P2 |
| Table S2, the cascade per corruption; Table S3, over image sets and the stream (with ETA, SAR and the main path inside ETA on the same streams) | `rule/held4/*`, `rule/slices`, `rule/streams` | `forward_only_heldout4_vitb_{rule,cbar_rule}`, `forward_only_{heldout,heldout2,heldout3}_vitb_rule`, `forward_only_heldout3_vitb_cbar_rule`, `forward_only_design_vitb{,_cbar}_rule`, `streams_heldout4_vitb_{rule,eta}` | E52 E53 E60–E63 T6 T8 → R45 R46 R49 R50 R51 R68 → P1 P2 |
| Table S4, ablations and controls (with the views control, the pair the labelled dev search ranks first and two pairs fixed by hand) | `rule/held4/*`, `rule/random`, `rule/views`, `exit/controls` | as Table 1; `forward_only_heldout4_vitb_{rule_ctl,cbar_rule_ctl}`, `forward_only_heldout4_vitb_{rule_views,cbar_rule_views}`, `forward_only_heldout4_vitb_{exitctl,cbar_exitctl}` | E48 E49 E77–E80 → R43 R66 R67 R69 R70 → P1 P2 |
| Section E, the main path inside ETA and SAR | `two_site/{held3_vitb,held3_vits,slices_vitb}`, `extra/{sar_design,sar_held3}` | `forward_only_heldout3_{vitb,vits}`, `eta_heldout3_{vitb,vits}`, `forward_only_{heldout,heldout2}_vitb_two_site`, `eta_{heldout,heldout2}_vitb_two_site`, `sar_design_vitb`, `sar_heldout3_vitb` | S5 S6 E14 E20–E23 E27 E30–E33 → R4 R22 R23 R32 R35–R38 → P1 P2 |
| Table S5, intervals of the held-out gains, the correlation without blocks 1 to 3, the entropy saved by the pick | `rule/review` | `review_checks` | A15 → P1 P2 |
| Figure S7, accuracy against compute and time | `rule/timing`, `rule/timing_others`, `paper/compute_cost.py` | `timing_design_vitb_ablations`, `timing_design_vitb_others` | C6 C7 → R64 R65 → P1 P2 |
| Table S6, other TTA methods | `design_vitb/*`, `rule/slices/design`, `rule/timing`, `rule/timing_others` | `forward_only_design_vitb`, `eta_design_vitb`, `sar_design_vitb`, `foa_design_extended_vitb`, `pea_design_vitb_src20k`, `forward_only_design_vitb_rule`, `timing_design_vitb_ablations`, `timing_design_vitb_others` | S1 E1 E10 E14 E16 E18 C6 C7 → R1 R3 R4 R6 R8 R64 R65 → P1 P2 |
| Section G, other maps (every block re-centered; the two-level map) | `hier/design_vitb`, `block_alignment` | `forward_only_design_vitb_hier`, `block_alignment_design_vitb` | S7 E26 F8 → R20 R29 → P1 P2 |
| Section H, the source-image sensitivity; FOA and PEA settings | `design_vitb/ablation`, `design_vitb/{foa_*,pea_*}`, `source` | `forward_only_design_vitb_{src50,src300,ablations}`, `foa_design_vitb`, `pea_design_vitb`, `patch_filters_design_vitb` (the unadapted model's accuracy) | S3 S4 E5–E7 E15 E17 → R1 R5 R7 → P1 P2 |
