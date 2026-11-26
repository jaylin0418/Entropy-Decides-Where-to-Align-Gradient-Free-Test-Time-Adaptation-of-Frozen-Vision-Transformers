#!/usr/bin/env bash
# Every number, table and figure of the course report, in run order, from the repository root.
#   IMAGENET_VAL  ImageNet val in class-folder layout (<wnid>/ILSVRC2012_val_*.JPEG)
#   IMAGENET_C    ImageNet-C (<corruption>/<severity>/<wnid>/*.JPEG)
#   IMAGENET_C_BAR  ImageNet-C-bar, the authors' archive extracted: the cbar_download directory (same layout as ImageNet-C), for E28 E29
#   PY_TIMM1      a Python whose environment has timm 1.0.15 and cma (FOA needs timm >= 0.9), for E15 E16
#   IMAGENET_R, IMAGENET_SKETCH  ImageNet-R and ImageNet-Sketch extracted (class folders), for the natural-shift pairs
# GPU time of the original runs on one GB10 (from their run records): about 25 GPU-hours for the TokAlign study, half of it
# FOA, PEA and the recurring/mixed streams, plus about 17 GPU-hours for the where-to-align study (main-path dev run, depth
# sweeps, drift by depth, heldout3 of both backbones; the ETA depth sweep alone is 10.5 hours), about 1.1 GPU-hours for the
# two-level (between-image / within-image) alignment and the decomposition of the token statistics on the design cells, and
# about 3.6 GPU-hours for the SAR runs with the main-path arms (E14 dev, E27 heldout3), 35 minutes for the two ImageNet-C-bar runs
# (E28 E29, mostly image decoding), 10 minutes for the main-path timing run (C2) and 3.3 GPU-hours for the two further held-out
# slices (E30 E31 forward-only 46 minutes each, E32 E33 ETA 53 minutes each), and several more GPU-hours for the cascade and its
# calibration (E34 to E39, E44 to E80), whose per-step times are in their run records.
# Steps [S] [E] [T] [C] [F] need a GPU unless marked CPU; [R] and [A] are CPU only; [P] builds the report. Step labels are stable
# (gaps are steps of the earlier studies that the report no longer uses); results/number_inventory.csv names the steps behind every number.
set -euo pipefail
: "${IMAGENET_VAL:?set IMAGENET_VAL}"; : "${IMAGENET_C:?set IMAGENET_C}"; : "${IMAGENET_C_BAR:?set IMAGENET_C_BAR}"; : "${PY_TIMM1:?set PY_TIMM1}"; : "${IMAGENET_R:?set IMAGENET_R}"; : "${IMAGENET_SKETCH:?set IMAGENET_SKETCH}"
D="--imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C --cache-dir cache --out-dir runs"
V="--imagenet-val $IMAGENET_VAL"
O=results/scores
FWD="neo tok headcov tok+headcov"
FWD4="neo tok place_b6 tok+place_b6"

# [S] source statistics (3,000 clean images of split 'source'; 50 / 300 for the ablation)
python scripts/compute_source_stats.py --model vitb --imagenet-val $IMAGENET_VAL --out stats/vitb.pt                           # S1
python scripts/compute_source_stats.py --model vits --imagenet-val $IMAGENET_VAL --out stats/vits.pt                           # S2
python scripts/compute_source_stats.py --model vitb --n-src 50 --imagenet-val $IMAGENET_VAL --out stats/vitb_n50.pt            # S3
python scripts/compute_source_stats.py --model vitb --n-src 300 --imagenet-val $IMAGENET_VAL --out stats/vitb_n300.pt          # S4

python scripts/compute_source_stats.py --model vitb --blocks --imagenet-val $IMAGENET_VAL --out stats/vitb_blocks.pt                # S5 (adds the per-block patch-token statistics)
python scripts/compute_source_stats.py --model vits --blocks --imagenet-val $IMAGENET_VAL --out stats/vits_blocks.pt                # S6
SB=stats/vitb_blocks.pt; SS=stats/vits_blocks.pt
python scripts/compute_source_stats.py --model vitb --blocks --imagenet-val $IMAGENET_VAL --out stats/vitb_hier.pt                  # S7 (as S5, plus the between-image / within-image parts; the pooled statistics are identical)
HB=stats/vitb_hier.pt
HIER="hier_px hier_b6 hier_px+hier_b6 tok+hier_b6 within_b6 between_b6 hier_b6_n0x4 hier_b6_n0d4"
TWO="neo tok headcov tok+headcov place_b6 tok+place_b6 tok+place_b6+headcov"
TWOETA="eta tok+eta eta+place_b6 tok+eta+place_b6"

# [E] episodic runs (one stream per cell)
python scripts/episodic/run_forward_only.py --split design --arms $FWD --stats stats/vitb.pt $D                 # E1
python scripts/episodic/run_forward_only.py --model vits --split design --arms $FWD --stats stats/vits.pt $D    # E3
python scripts/episodic/run_forward_only.py --split design --arms tok+headcov --stats stats/vitb_n50.pt --tag _src50 $D     # E5
python scripts/episodic/run_forward_only.py --split design --arms tok+headcov --stats stats/vitb_n300.pt --tag _src300 $D   # E6
python scripts/episodic/run_forward_only.py --split design --arms stack_lam0 stack_lam0.3 stack_n0x16 stack_headn0x4 tok_only_nohead \
    --stats stats/vitb.pt --tag _ablations $D                                                                   # E7
python scripts/episodic/run_gradient_host.py --method eta --split design --stats stats/vitb.pt $D               # E10
python scripts/episodic/run_gradient_host.py --method eta --model vits --split design --stats stats/vits.pt $D  # E12
python scripts/episodic/run_gradient_host.py --method sar --split design --arms sar tok+sar sar+place_b6 tok+sar+place_b6 --stats $SB $D  # E14 (the two-site arms were added 2026-11-09; needs S5)
$PY_TIMM1 scripts/episodic/run_foa.py --n-eval 2000 --stats stats/vitb.pt $D                                    # E15
$PY_TIMM1 scripts/episodic/run_foa.py --n-eval 6000 --stats stats/vitb.pt $D                                    # E16
python scripts/episodic/run_pea.py --stats stats/vitb.pt $D                                                     # E17
python scripts/episodic/run_pea.py --src-split pea_source --n-src 20000 --tag _src20k --stats stats/vitb.pt $D  # E18

# where to align (the report's Section 4 and Supplement C): the two-site design run, heldout3 of both backbones, the depth sweeps
python scripts/episodic/run_forward_only.py --split design --arms $TWO --stats $SB --tag _two_site $D                       # E19
python scripts/episodic/run_forward_only.py --split heldout3 --arms $TWO --stats $SB $D                                     # E20
python scripts/episodic/run_gradient_host.py --method eta --split heldout3 --arms $TWOETA --stats $SB $D                    # E21
python scripts/episodic/run_forward_only.py --model vits --split heldout3 --arms $TWO --stats $SS $D                        # E22
python scripts/episodic/run_gradient_host.py --method eta --model vits --split heldout3 --arms $TWOETA --stats $SS $D       # E23
python scripts/episodic/run_gradient_host.py --method sar --split heldout3 --arms sar tok+sar sar+place_b6 tok+sar+place_b6 --stats $SB $D  # E27 (SAR with the two sites, heldout3)
CBARCELLS=$(python -c "from tokalign import data; print(' '.join(data.CBAR_CELLS))"); DBAR="--imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C_BAR --cache-dir cache --out-dir runs"
python scripts/episodic/run_forward_only.py --split design   --arms $FWD4 --cells clean $CBARCELLS --stats $SB --tag _cbar $DBAR   # E28 (ImageNet-C-bar, ten other corruptions)
python scripts/episodic/run_forward_only.py --split heldout3 --arms $FWD4 --cells clean $CBARCELLS --stats $SB --tag _cbar $DBAR   # E29
for sl in heldout heldout2; do                                                                                                  # E30 E31 (forward-only) E32 E33 (ETA): the heldout3 arms on the two other held-out slices
  python scripts/episodic/run_forward_only.py --split $sl --arms $TWO --stats $SB --tag _two_site $D
  python scripts/episodic/run_gradient_host.py --method eta --split $sl --arms $TWOETA --stats $SB --tag _two_site $D
done
# the cascade study: the pixel site plus one block for every block (design), the early-exit arms (design), the pre-registered heldout4 test
PXB="tok+place_b1 tok+place_b2 tok+place_b3 tok+place_b4 tok+place_b5 tok+place_b7 tok+place_b8 tok+place_b9 tok+place_b10 tok+place_b11 tok+place_b12"
PB="place_b1 place_b2 place_b3 place_b4 place_b5 place_b7 place_b8 place_b9 place_b10 place_b11 place_b12"
EXIT="exit_b9 exit_b5b9 exit_b5b9_gated pair_gated"
EXIT4="neo tok place_b6 tok+place_b6 tok+place_b6+headcov pair_gated exit_b9_gated exit_b5b9_gated"
python scripts/episodic/run_forward_only.py --split design --arms $PXB --stats $SB --tag _select $D                          # E34
python scripts/episodic/run_forward_only.py --split design --arms $PB $PXB --cells clean $CBARCELLS --stats $SB --tag _cbar_select $DBAR  # E35
python scripts/episodic/run_forward_only.py --split design --arms $EXIT --stats $SB --tag _exit $D                           # E36
python scripts/episodic/run_forward_only.py --split design --arms $EXIT --cells clean $CBARCELLS --stats $SB --tag _cbar_exit $DBAR  # E37
python scripts/episodic/run_forward_only.py --split heldout4 --arms $EXIT4 --stats $SB --tag _exit $D                        # E38 (heldout4, scored once)
python scripts/episodic/run_forward_only.py --split heldout4 --arms $EXIT4 --cells clean $CBARCELLS --stats $SB --tag _cbar_exit $DBAR  # E39 (heldout4, scored once)
python scripts/cost/time_batches.py --arms neo pair_k16_fp32_stats32 branch_k16_fp32_stats32 exit_b9_k16_fp32_stats32 exit_b5b9_k16_fp32_stats32 --stats $SB --tag _exit $D  # C3 (early-exit timing)
for sl in heldout heldout2 heldout3; do                                                                                       # E44 E45 E46 (the gated early-exit arms on the earlier held-out slices, after the heldout4 verdict)
  python scripts/episodic/run_forward_only.py --split $sl --arms pair_gated exit_b9_gated exit_b5b9_gated --stats $SB --tag _exit $D
done
python scripts/episodic/run_forward_only.py --split heldout3 --arms pair_gated exit_b9_gated exit_b5b9_gated --cells clean $CBARCELLS --stats $SB --tag _cbar_exit $DBAR  # E47
CTL="exit_b5b9_random_gated exit_b5b9_equal_gated mix_b5b9_gated"
python scripts/episodic/run_forward_only.py --split heldout4 --arms $CTL --stats $SB --tag _exitctl $D                        # E48 (equal-compute controls)
python scripts/episodic/run_forward_only.py --split heldout4 --arms $CTL --cells clean $CBARCELLS --stats $SB --tag _cbar_exitctl $DBAR  # E49
python scripts/episodic/run_forward_only.py --model vits --split design --arms tok+place_b6 neo pair_gated exit_b9_gated exit_b5b9_gated --stats $SS --tag _exit $D  # E50 (ViT-S/16)
python scripts/episodic/run_forward_only.py --model vits --split heldout3 --arms pair_gated exit_b9_gated exit_b5b9_gated --stats $SS --tag _exit $D        # E51
# the calibration of the headline ViT-B/16 blocks (CPU, from the dev logits above), checked before the held-out runs that use them
python scripts/analysis/site_choice.py --logits runs/logits --base forward_only_design_vitb_two_site --blocks forward_only_design_vitb_select $V --cells inc --out $O/site_choice_design_vitb.json        # A6
python scripts/analysis/site_choice.py --logits runs/logits --base forward_only_design_vitb_cbar --blocks forward_only_design_vitb_cbar_select $V --cells cbar --out $O/site_choice_design_vitb_cbar.json  # A7
python scripts/analysis/site_choice.py --logits runs/logits --base forward_only_design_vitb_cbar --blocks forward_only_design_vitb_cbar_select $V --cells cbar --trunk 7 --out $O/site_choice_design_vitb_cbar_t7.json  # A8 (the rule's own trunk on ImageNet-C-bar)
python scripts/analysis/rule_sample_size.py --logits runs/logits --base forward_only_design_vitb_two_site --blocks forward_only_design_vitb_select $V --cells inc --out $O/rule_sample_size_design_vitb.json        # A9 (how many unlabeled images the rule needs)
python scripts/analysis/rule_sample_size.py --logits runs/logits --base forward_only_design_vitb_cbar --blocks forward_only_design_vitb_cbar_select $V --cells cbar --out $O/rule_sample_size_design_vitb_cbar.json  # A10
python - <<'PY'   # A16 (CPU): the blocks E52 to E82 run (6 / 4, 11 and 7 / 5, 11, the 320-image 5 / 3, 7 and 7 / 5, 9, the 192-image 7 / 5, 8) are the calibration's outputs
import json
O = 'results/scores'; b = lambda a: int(a[4:])
def full(f):
    lf = json.load(open(f'{O}/{f}.json'))['label_free']; return (b(lf['trunk_pick_by_entropy']), *sorted(b(x) for x in lf['second_opinion_pick_pair'].split('|')))
def pooled(f, n):
    d = json.load(open(f'{O}/{f}.json'))['pooled_first_n_per_cell'][n]; return tuple(b(d[k]) for k in ('trunk', 'shallower', 'deeper'))
assert full('site_choice_design_vitb') == (6, 4, 11) and full('site_choice_design_vitb_cbar_t7') == (7, 5, 11)
assert pooled('rule_sample_size_design_vitb', '320') == (5, 3, 7) and pooled('rule_sample_size_design_vitb_cbar', '320') == (7, 5, 9)
assert pooled('rule_sample_size_design_vitb_cbar', '192') == (7, 5, 8)
PY
# the rule-chosen (calibrated) second-opinion stack, docs/heldout_test_plan_rule.md: arms named rule_t<T>_b<A>b<B>_gated
python scripts/episodic/run_forward_only.py --split heldout4 --arms rule_t6_b4b11_gated --stats $SB --tag _rule $D                                        # E52 (R1, scored once)
python scripts/episodic/run_forward_only.py --split heldout4 --arms rule_t6_b4b11_gated rule_t7_b5b11_gated tok+place_b7 --cells clean $CBARCELLS --stats $SB --tag _cbar_rule --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C_BAR --cache-dir cache --out-dir runs  # E53 (R2)
python scripts/episodic/run_forward_only.py --split heldout4 --arms rule_t6_b4_gated rule_t6_b11_gated --stats $SB --tag _rule_abl $D                        # E54 (ablation)
python scripts/episodic/run_forward_only.py --split heldout4 --arms rule_t6_b4_gated rule_t6_b11_gated --cells clean $CBARCELLS --stats $SB --tag _cbar_rule_abl --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C_BAR --cache-dir cache --out-dir runs  # E55
python scripts/episodic/run_forward_only.py --split heldout4 --arms rule_t5_b3b7_gated tok+place_b5 --stats $SB --tag _rule_small $D                       # E56 (R4, the small-sample calibration)
python scripts/episodic/run_forward_only.py --split heldout4 --arms rule_t7_b5b9_gated --cells clean $CBARCELLS --stats $SB --tag _cbar_rule_small --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C_BAR --cache-dir cache --out-dir runs  # E57 (R5)
python scripts/episodic/run_forward_only.py --model vits --split design --arms neo tok place_b6 tok+place_b6 tok+place_b1 tok+place_b2 tok+place_b3 tok+place_b4 tok+place_b5 tok+place_b7 tok+place_b8 tok+place_b9 tok+place_b10 tok+place_b11 tok+place_b12 --stats $SS --tag _rule_sweep $D  # E58 (ViT-S/16 sweep for its own calibration)
python scripts/analysis/site_choice.py --logits runs/logits --base forward_only_design_vits_rule_sweep --blocks forward_only_design_vits_rule_sweep $V --cells inc --trunk 6 --out $O/site_choice_design_vits_rule_t6.json  # A11 (ViT-S/16: the trunk pick)
# a pick of TokAlign alone ('px') would need an arm without LayerAlign, which no pair needed: the [4:] slices below assume a block
VITS_TRUNK=$(python -c "import json; print(json.load(open('$O/site_choice_design_vits_rule_t6.json'))['label_free']['trunk_pick_by_entropy'][4:])")
python scripts/analysis/site_choice.py --logits runs/logits --base forward_only_design_vits_rule_sweep --blocks forward_only_design_vits_rule_sweep $V --cells inc --trunk $VITS_TRUNK --out $O/site_choice_design_vits_rule.json  # A12 (ViT-S/16: the two branch blocks given that trunk)
read -r VT VA1 VA2 < <(python -c "import json; d=json.load(open('$O/site_choice_design_vits_rule.json')); lo, hi = (a[4:] for a in d['label_free']['second_opinion_pick_pair'].split('|')); print(d['trunk'][4:], lo, hi)")
python scripts/episodic/run_forward_only.py --model vits --split heldout3 --arms rule_t${VT}_b${VA1}b${VA2}_gated rule_t${VT}_b${VA1}_gated rule_t${VT}_b${VA2}_gated --exit-theta 1.5 --stats $SS --tag _rule $D   # E59 (R3: the picks of A11 A12, blocks 6 / 4, 11, theta 1.5 nat as on ViT-B/16)
for s in heldout heldout2 heldout3; do python scripts/episodic/run_forward_only.py --split $s --arms rule_t6_b4b11_gated --stats $SB --tag _rule $D; done   # E60 E61 E62 (the other slices)
python scripts/episodic/run_forward_only.py --split design --arms rule_t6_b4b11_gated --stats $SB --tag _rule $D   # E81 (the dev cells: Table S3's dev row and the cascade in Table S6)
python scripts/episodic/run_forward_only.py --split design --arms rule_t6_b4b11_gated --cells clean $CBARCELLS --stats $SB --tag _cbar_rule --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C_BAR --cache-dir cache --out-dir runs   # E82 (ImageNet-C-bar dev cells, Table S3)
python scripts/episodic/run_forward_only.py --split heldout3 --arms rule_t6_b4b11_gated --cells clean $CBARCELLS --stats $SB --tag _cbar_rule --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C_BAR --cache-dir cache --out-dir runs  # E63
python scripts/episodic/run_forward_only.py --split heldout4 --arms rule_t6_b4b11_random_gated --stats $SB --tag _rule_random $D                          # E64 (random-routing control)
python scripts/episodic/run_forward_only.py --split heldout4 --arms rule_t6_b4b11_random_gated --cells clean $CBARCELLS --stats $SB --tag _cbar_rule_random --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C_BAR --cache-dir cache --out-dir runs  # E65
python scripts/streams/run_streams.py --split heldout4 --arms neo pair_gated exit_b5b9_gated --streams cont5 cleanret --stats $SB --tag _exit $D   # T5 (the second-opinion stack on streams)
python scripts/streams/run_streams.py --split heldout4 --arms neo pair_gated exit_b5b9_gated rule_t6_b4b11_gated --streams cont5 cleanret --stats $SB --tag _rule $D  # T6 (the calibrated stack on streams)
python scripts/episodic/run_forward_only.py --split heldout4 --arms rule_t6_b3b9_gated rule_t6_b1b12_gated rule_t6_b5b7_gated --exit-theta 1.5 --stats $SB --tag _rule_ctl $D   # E77 (Table S4: the pair the labeled dev search ranks first and two pairs fixed by hand)
python scripts/episodic/run_forward_only.py --split heldout4 --arms rule_t6_b3b9_gated rule_t6_b1b12_gated rule_t6_b5b7_gated --exit-theta 1.5 --cells clean $CBARCELLS --stats $SB --tag _cbar_rule_ctl --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C_BAR --cache-dir cache --out-dir runs  # E78
python scripts/streams/run_streams.py --split heldout4 --arms neo eta sar twosite+eta_gated --streams cont5 cleanret --stats $SB --tag _eta $D  # T8 (ETA, SAR and the main path inside ETA on the same streams, weights carried across the cells)
python scripts/episodic/run_forward_only.py --split heldout4 --arms rule_t6_views_gated --exit-theta 1.5 --stats $SB --tag _rule_views $D   # E79 (Table S4: the main path on three views of the same unsure images instead of two more blocks)
python scripts/episodic/run_forward_only.py --split heldout4 --arms rule_t6_views_gated --exit-theta 1.5 --cells clean $CBARCELLS --stats $SB --tag _cbar_rule_views --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C_BAR --cache-dir cache --out-dir runs  # E80
python scripts/episodic/run_forward_only.py --split design --arms place_b1 place_b2 place_b3 place_b4 place_b5 place_b7 place_b8 \
    place_b9 place_b10 place_b11 place_b12 --stats $SB --tag _sweep $D                                                      # E24
python scripts/episodic/run_gradient_host.py --method eta --split design --stats $SB --tag _sweep $D \
    --arms eta+place_b1 eta+place_b2 eta+place_b3 eta+place_b4 eta+place_b5 eta+place_b6 eta+place_b7 eta+place_b8 eta+place_b9 \
    eta+place_b10 eta+place_b11 eta+place_b12                                                                               # E25
# what to align at a site (the report's Section G, the two-level map): the two-level map and its one-part variants
python scripts/episodic/run_forward_only.py --split design --arms $HIER --stats $HB --tag _hier $D                           # E26

# [T] streams (one host and one set of EMA statistics per stream, no resets)

python scripts/streams/run_streams.py --split heldout3 --streams cont5 cleanret \
    --arms neo stack_gated twosite_gated eta tok+eta_gated twosite+eta_gated --stats $SB $D                                 # T7

# [C] wall-clock (one run; the ratios hold within that run)
python scripts/cost/time_batches.py --arms neo stack_k1 stack_k16_fp32 stack_k16_fp32_stats32 eta --stats stats/vitb.pt $D  # C1
python scripts/cost/time_batches.py --arms neo stack_k16_fp32_stats32 twosite_k16_fp32_stats32 twosite_k1 eta eta+twosite --stats $SB --tag _extra $D  # C2 (two-site stack; needs S5)

# [F] the drift and the decomposition of the token statistics at every site (GPU, no labels)
python scripts/block_alignment/run_block_alignment.py $D                                                       # F8
python scripts/analysis/drift_by_depth.py --split design --stats $SB $D                                        # F9 (drift at every site)
python scripts/analysis/decompose_by_depth.py --split design --stats $HB $D                                    # F10 (between-image / within-image parts at every site)

# [R] scoring (CPU; labels are read only here)
for r in design_vitb design_vits design_vitb_src50 design_vitb_src300 design_vitb_ablations; do
  python scripts/episodic/score_forward_only.py --runs runs/forward_only_$r.json $V --out $O/forward_only_$r.json   # R1
done
for r in design_vitb:vitb:design_vitb design_vits:vits:design_vits; do
  IFS=: read -r name model fo <<< "$r"
  python scripts/episodic/score_gradient_host.py --runs runs/eta_$name.json --stats stats/$model.pt $V \
      --forward-only $O/forward_only_$fo.json --out $O/eta_$name.json                                           # R3
done
python scripts/episodic/score_gradient_host.py --runs runs/sar_design_vitb.json --stats $SB $V \
    --forward-only $O/forward_only_design_vitb.json --out $O/sar_design_vitb.json                               # R4
REF="--forward-only $O/forward_only_design_vitb.json --gradient-host $O/eta_design_vitb.json"
python scripts/episodic/score_foa.py --runs runs/foa_design_vitb.json $V $REF --out $O/foa_design_vitb.json   # R5
python scripts/episodic/score_foa.py --runs runs/foa_design_extended_vitb.json $V $REF --out $O/foa_design_extended_vitb.json  # R6
python scripts/episodic/score_pea.py --runs runs/pea_design_vitb.json $V $REF --out $O/pea_design_vitb.json   # R7
python scripts/episodic/score_pea.py --runs runs/pea_design_vitb_src20k.json $V $REF --out $O/pea_design_vitb_src20k.json  # R8
python scripts/cost/score_timing.py --runs runs/timing_design_vitb.json $V --out $O/timing_design_vitb.json    # R10
python scripts/cost/score_timing.py --runs runs/timing_design_vitb_extra.json $V --out $O/timing_design_vitb_extra.json  # R31
python scripts/block_alignment/score_block_alignment.py --runs runs/block_alignment_design_vitb.json \
    --diag runs/block_alignment_design_vitb_diag.json $V                                                        # R20
# where to align: scoring and the model-free analyses
for r in design_vitb_two_site heldout3_vitb heldout3_vits; do
  python scripts/episodic/score_forward_only.py --runs runs/forward_only_$r.json $V --out $O/forward_only_$r.json   # R22
done
python scripts/episodic/score_forward_only.py --runs runs/forward_only_design_vitb_sweep.json $V \
    --neo-from $O/forward_only_design_vitb_two_site.json --out $O/forward_only_design_vitb_sweep.json              # R24 (no neo arm in the sweep)
python scripts/episodic/score_gradient_host.py --runs runs/eta_heldout3_vitb.json --stats $SB $V --out $O/eta_heldout3_vitb.json   # R23
python scripts/episodic/score_gradient_host.py --runs runs/eta_heldout3_vits.json --stats $SS $V --out $O/eta_heldout3_vits.json   # R23
python scripts/episodic/score_gradient_host.py --runs runs/sar_heldout3_vitb.json --stats $SB $V --out $O/sar_heldout3_vitb.json   # R32
python scripts/episodic/score_forward_only.py --runs runs/forward_only_design_vitb_cbar.json $V --out $O/forward_only_design_vitb_cbar.json      # R33
python scripts/episodic/score_forward_only.py --runs runs/forward_only_heldout3_vitb_cbar.json $V --out $O/forward_only_heldout3_vitb_cbar.json  # R34
for sl in heldout heldout2; do                                                                                                  # R35 R36 R37 R38 (the two other held-out slices)
  python scripts/episodic/score_forward_only.py --runs runs/forward_only_${sl}_vitb_two_site.json $V --out $O/forward_only_${sl}_vitb_two_site.json
  python scripts/episodic/score_gradient_host.py --runs runs/eta_${sl}_vitb_two_site.json --stats $SB $V --out $O/eta_${sl}_vitb_two_site.json
done
for r in design_vitb_select design_vitb_cbar_select design_vitb_exit design_vitb_cbar_exit heldout4_vitb_exit heldout4_vitb_cbar_exit; do   # R39 (six files)
  python scripts/episodic/score_forward_only.py --runs runs/forward_only_$r.json $V --out $O/forward_only_$r.json
done
python scripts/cost/score_timing.py --runs runs/timing_design_vitb_exit.json $V --out $O/timing_design_vitb_exit.json          # R40
for r in heldout_vitb_two_site:heldout heldout2_vitb_two_site:heldout2 heldout3_vitb:heldout3; do                             # R42 (the range runs, NEO reference from each slice's earlier file)
  IFS=: read ref sl <<< "$r"
  python scripts/episodic/score_forward_only.py --runs runs/forward_only_${sl}_vitb_exit.json $V --neo-from $O/forward_only_$ref.json --out $O/forward_only_${sl}_vitb_exit.json
done
python scripts/episodic/score_forward_only.py --runs runs/forward_only_heldout3_vitb_cbar_exit.json $V --neo-from $O/forward_only_heldout3_vitb_cbar.json --out $O/forward_only_heldout3_vitb_cbar_exit.json
python scripts/episodic/score_forward_only.py --runs runs/forward_only_heldout4_vitb_exitctl.json $V --neo-from $O/forward_only_heldout4_vitb_exit.json --out $O/forward_only_heldout4_vitb_exitctl.json       # R43
python scripts/streams/score_streams.py --runs runs/streams_heldout4_vitb_exit.json $V --out $O/streams_heldout4_vitb_exit.json    # R44
for t in rule:forward_only_heldout4_vitb_exit rule_abl:forward_only_heldout4_vitb_exit rule_small:forward_only_heldout4_vitb_exit rule_random:forward_only_heldout4_vitb_exit; do   # R45 (the rule arms on heldout4, NEO from the exit file)
  python scripts/episodic/score_forward_only.py --runs runs/forward_only_heldout4_vitb_${t%%:*}.json $V --neo-from $O/${t##*:}.json --out $O/forward_only_heldout4_vitb_${t%%:*}.json; done
for t in cbar_rule cbar_rule_abl cbar_rule_small cbar_rule_random; do python scripts/episodic/score_forward_only.py --runs runs/forward_only_heldout4_vitb_$t.json $V --neo-from $O/forward_only_heldout4_vitb_cbar_exit.json --out $O/forward_only_heldout4_vitb_$t.json; done  # R46
python scripts/episodic/score_forward_only.py --runs runs/forward_only_design_vits_rule_sweep.json $V --out $O/forward_only_design_vits_rule_sweep.json                         # R47
python scripts/episodic/score_forward_only.py --runs runs/forward_only_heldout3_vits_rule.json $V --neo-from $O/forward_only_heldout3_vits.json --out $O/forward_only_heldout3_vits_rule.json  # R48
for r in heldout_vitb_two_site:heldout heldout2_vitb_two_site:heldout2 heldout3_vitb:heldout3; do python scripts/episodic/score_forward_only.py --runs runs/forward_only_${r##*:}_vitb_rule.json $V --neo-from $O/forward_only_${r%%:*}.json --out $O/forward_only_${r##*:}_vitb_rule.json; done  # R49
python scripts/episodic/score_forward_only.py --runs runs/forward_only_heldout3_vitb_cbar_rule.json $V --neo-from $O/forward_only_heldout3_vitb_cbar.json --out $O/forward_only_heldout3_vitb_cbar_rule.json  # R50
python scripts/episodic/score_forward_only.py --runs runs/forward_only_design_vitb_rule.json $V --neo-from $O/forward_only_design_vitb_two_site.json --out $O/forward_only_design_vitb_rule.json  # R71 (E81)
python scripts/episodic/score_forward_only.py --runs runs/forward_only_design_vitb_cbar_rule.json $V --neo-from $O/forward_only_design_vitb_cbar.json --out $O/forward_only_design_vitb_cbar_rule.json  # R72 (E82)
python scripts/streams/score_streams.py --runs runs/streams_heldout4_vitb_rule.json $V --out $O/streams_heldout4_vitb_rule.json    # R51
python scripts/episodic/score_forward_only.py --runs runs/forward_only_heldout4_vitb_rule_ctl.json $V --neo-from $O/forward_only_heldout4_vitb_exit.json --out $O/forward_only_heldout4_vitb_rule_ctl.json  # R66
python scripts/episodic/score_forward_only.py --runs runs/forward_only_heldout4_vitb_cbar_rule_ctl.json $V --neo-from $O/forward_only_heldout4_vitb_cbar_exit.json --out $O/forward_only_heldout4_vitb_cbar_rule_ctl.json  # R67
python scripts/streams/score_streams.py --runs runs/streams_heldout4_vitb_eta.json $V --out $O/streams_heldout4_vitb_eta.json    # R68 (ETA and SAR scored alone and with NEO on the recorded head input)
python scripts/episodic/score_forward_only.py --runs runs/forward_only_heldout4_vitb_rule_views.json $V --neo-from $O/forward_only_heldout4_vitb_exit.json --out $O/forward_only_heldout4_vitb_rule_views.json  # R69
python scripts/episodic/score_forward_only.py --runs runs/forward_only_heldout4_vitb_cbar_rule_views.json $V --neo-from $O/forward_only_heldout4_vitb_cbar_exit.json --out $O/forward_only_heldout4_vitb_cbar_rule_views.json  # R70
python scripts/episodic/score_forward_only.py --runs runs/forward_only_heldout4_vitb_cbar_exitctl.json $V --neo-from $O/forward_only_heldout4_vitb_cbar_exit.json --out $O/forward_only_heldout4_vitb_cbar_exitctl.json
python scripts/episodic/score_forward_only.py --runs runs/forward_only_design_vits_exit.json $V --out $O/forward_only_design_vits_exit.json
python scripts/episodic/score_forward_only.py --runs runs/forward_only_heldout3_vits_exit.json $V --neo-from $O/forward_only_heldout3_vits.json --out $O/forward_only_heldout3_vits_exit.json
# analyses from the saved logits (CPU): which block per cell or per image, the calibration, the sample-size study
python scripts/episodic/score_gradient_host.py --runs runs/eta_design_vitb_sweep.json --stats $SB $V --out $O/eta_design_vitb_sweep.json  # R25
python scripts/analysis/score_drift_by_depth.py --runs runs/drift_by_depth_design_vitb.json --out $O/drift_by_depth_design_vitb.json  # R26
python scripts/analysis/site_selection.py --two-site $O/forward_only_design_vitb_two_site.json --sweep $O/forward_only_design_vitb_sweep.json \
    --drift $O/drift_by_depth_design_vitb.json --out $O/site_selection_design_vitb.json                                   # R27
python scripts/streams/score_streams.py --runs runs/streams_heldout3_vitb.json $V --out $O/streams_heldout3_vitb.json   # R28
python scripts/episodic/score_forward_only.py --runs runs/forward_only_design_vitb_hier.json $V \
    --neo-from $O/forward_only_design_vitb_two_site.json --out $O/forward_only_design_vitb_hier.json                # R29 (no neo arm in the two-level run)
python scripts/analysis/score_decompose_by_depth.py --runs runs/decompose_by_depth_design_vitb.json --out $O/decompose_by_depth_design_vitb.json  # R30

# the calibrated stack on more models and shifts (docs/heldout_test_plan_models_datasets.md): the same procedure for every
# model-dataset pair, no label at any step. Models: vitb vits deitb deits; datasets: inc (ImageNet-C) r (ImageNet-R) sketch (ImageNet-Sketch).
python scripts/compute_source_stats.py --model deitb --blocks --imagenet-val $IMAGENET_VAL --out stats/deitb_blocks.pt           # S8
python scripts/compute_source_stats.py --model deits --blocks --imagenet-val $IMAGENET_VAL --out stats/deits_blocks.pt           # S9
SWEEP="neo tok place_b6 $(for k in 1 2 3 4 5 6 7 8 9 10 11 12; do printf 'tok+place_b%s ' $k; done)"
root_of() { case "$1" in inc) echo "$IMAGENET_C" ;; r) echo "$IMAGENET_R" ;; sketch) echo "$IMAGENET_SKETCH" ;; esac; }
dsflag() { [ "$1" = inc ] && echo "" || echo "--dataset $1"; }
heldsplit() { [ "$1" = inc ] && echo "heldout4" || echo "heldout"; }
for pair in vitb_r vitb_sketch vits_r vits_sketch deitb_inc deitb_r deitb_sketch deits_inc deits_r deits_sketch; do                 # E66 (ten design sweeps)
  m=${pair%%_*}; ds=${pair#*_}
  python scripts/episodic/run_forward_only.py --model $m --split design $(dsflag $ds) --arms $SWEEP --stats stats/${m}_blocks.pt --tag _ca_$ds --imagenet-val $IMAGENET_VAL --imagenet-c $(root_of $ds) --cache-dir cache --out-dir runs
  python scripts/episodic/score_forward_only.py --runs runs/forward_only_design_${m}_ca_$ds.json $V --out $O/forward_only_design_${m}_ca_$ds.json   # R53
  SC="scripts/analysis/site_choice.py --logits runs/logits --base forward_only_design_${m}_ca_$ds --blocks forward_only_design_${m}_ca_$ds $V --cells $ds"
  python $SC --trunk 6 --out $O/site_choice_design_${m}_ca_${ds}_t6.json                                                            # A13 (the calibration, CPU, no labels)
  T=$(python -c "import json; print(json.load(open('$O/site_choice_design_${m}_ca_${ds}_t6.json'))['label_free']['trunk_pick_by_entropy'][4:])")
  python $SC --trunk $T --out $O/site_choice_design_${m}_ca_$ds.json
  python - <<PYEOF
import json
d = json.load(open('$O/site_choice_design_${m}_ca_$ds.json')); lf = d['label_free']
lo, hi = (int(a[4:]) for a in lf['second_opinion_pick_pair'].split('|'))
cfg = {'model': '$m', 'dataset': '$ds', 'trunk': int(d['trunk'][4:]), 'shallower': lo, 'deeper': hi, 'theta': lf['trunk_entropy_quantile']['0.65'],
       'theta_rule': '0.65 quantile of the base path entropy on the design images', 'trunk_best_by_accuracy': lf['trunk_best_by_accuracy'],
       'routed_share_at_1.5': lf['routed_share_at_theta'], 'arm': f"rule_t{int(d['trunk'][4:])}_b{lo}b{hi}_gated"}
json.dump(cfg, open('$O/calib_${m}_$ds.json', 'w'), indent=1)
PYEOF
  read -r T A1 A2 TH < <(python -c "import json; c=json.load(open('$O/calib_${m}_$ds.json')); print(c['trunk'], c['shallower'], c['deeper'], c['theta'])")
  split=$(heldsplit $ds)
  python scripts/episodic/run_forward_only.py --model $m --split $split $(dsflag $ds) --arms neo tok tok+place_b$T rule_t${T}_b${A1}b${A2}_gated rule_t${T}_b${A1}_gated rule_t${T}_b${A2}_gated \
      --exit-theta $TH --stats stats/${m}_blocks.pt --tag _ca_$ds --imagenet-val $IMAGENET_VAL --imagenet-c $(root_of $ds) --cache-dir cache --out-dir runs   # E67 (held-out, scored once)
  python scripts/episodic/score_forward_only.py --runs runs/forward_only_${split}_${m}_ca_$ds.json $V --out $O/forward_only_${split}_${m}_ca_$ds.json       # R54
  python scripts/episodic/run_gradient_host.py --method eta --model $m --split $split $(dsflag $ds) --arms eta tok+eta+place_b$T --stats stats/${m}_blocks.pt --tag _ca_$ds \
      --imagenet-val $IMAGENET_VAL --imagenet-c $(root_of $ds) --cache-dir cache --out-dir runs                                      # E68 (ETA on the same cells)
  python scripts/episodic/score_gradient_host.py --runs runs/eta_${split}_${m}_ca_$ds.json --stats stats/${m}_blocks.pt $V --out $O/eta_${split}_${m}_ca_$ds.json   # R55
done

# ImageNet-C-bar for the across-models grid (Table 3; addendum 4): the same five steps on the 20 ImageNet-C-bar cells
CBARCELLS=$(python -c "from tokalign import data; print(' '.join(data.CBAR_CELLS))")
for m in vits deitb deits; do
  python scripts/episodic/run_forward_only.py --model $m --split design --arms $SWEEP --cells clean $CBARCELLS --stats stats/${m}_blocks.pt --tag _cbar_ca \
      --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C_BAR --cache-dir cache --out-dir runs                                      # E71 (the design sweep)
  python scripts/episodic/score_forward_only.py --runs runs/forward_only_design_${m}_cbar_ca.json $V --out $O/forward_only_design_${m}_cbar_ca.json   # R58
  SC="scripts/analysis/site_choice.py --logits runs/logits --base forward_only_design_${m}_cbar_ca --blocks forward_only_design_${m}_cbar_ca --imagenet-val $IMAGENET_VAL --cells cbar"
  python $SC --trunk 6 --out $O/site_choice_design_${m}_cbar_ca_t6.json                                                              # A14 (the calibration)
  T=$(python -c "import json; print(json.load(open('$O/site_choice_design_${m}_cbar_ca_t6.json'))['label_free']['trunk_pick_by_entropy'][4:])")
  python $SC --trunk $T --out $O/site_choice_design_${m}_cbar_ca.json
  python - <<PYEOF
import json
d = json.load(open('$O/site_choice_design_${m}_cbar_ca.json')); lf = d['label_free']
lo, hi = (int(a[4:]) for a in lf['second_opinion_pick_pair'].split('|'))
cfg = {'model': '$m', 'dataset': 'cbar', 'trunk': int(d['trunk'][4:]), 'shallower': lo, 'deeper': hi, 'theta': lf['trunk_entropy_quantile']['0.65'],
       'theta_rule': '0.65 quantile of the base path entropy on the design images', 'trunk_best_by_accuracy': lf['trunk_best_by_accuracy'],
       'routed_share_at_1.5': lf['routed_share_at_theta'], 'arm': f"rule_t{int(d['trunk'][4:])}_b{lo}b{hi}_gated"}
json.dump(cfg, open('$O/calib_${m}_cbar.json', 'w'), indent=1)
PYEOF
  read -r T A1 A2 TH < <(python -c "import json; c=json.load(open('$O/calib_${m}_cbar.json')); print(c['trunk'], c['shallower'], c['deeper'], c['theta'])")
  python scripts/episodic/run_forward_only.py --model $m --split heldout4 --arms neo tok tok+place_b$T rule_t${T}_b${A1}b${A2}_gated rule_t${T}_b${A1}_gated rule_t${T}_b${A2}_gated \
      --exit-theta $TH --cells clean $CBARCELLS --stats stats/${m}_blocks.pt --tag _cbar_ca --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C_BAR --cache-dir cache --out-dir runs   # E72 (held-out, scored once)
  python scripts/episodic/score_forward_only.py --runs runs/forward_only_heldout4_${m}_cbar_ca.json $V --out $O/forward_only_heldout4_${m}_cbar_ca.json   # R59
done
for m in vitb vits deitb deits; do                                                                               # ETA on the ImageNet-C-bar held-out cells (ViT-B/16: 7 / 5, 11, theta 1.5)
  if [ $m = vitb ]; then T=7; A1=5; A2=11; TH=1.5; else read -r T A1 A2 TH < <(python -c "import json; c=json.load(open('$O/calib_${m}_cbar.json')); print(c['trunk'], c['shallower'], c['deeper'], c['theta'])"); fi
  python scripts/episodic/run_gradient_host.py --method eta --model $m --split heldout4 --arms eta tok+eta+place_b$T --cells clean $CBARCELLS --stats stats/${m}_blocks.pt --tag _cbar_ca \
      --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C_BAR --cache-dir cache --out-dir runs                                      # E73
  python scripts/episodic/score_gradient_host.py --runs runs/eta_heldout4_${m}_cbar_ca.json --stats stats/${m}_blocks.pt $V --out $O/eta_heldout4_${m}_cbar_ca.json   # R60
  python scripts/episodic/run_gradient_host.py --method eta --model $m --split heldout4 --arms tok+eta+rule_t${T}_b${A1}b${A2} --exit-theta $TH --cells clean $CBARCELLS --stats stats/${m}_blocks.pt --tag _cbar_ca_stack \
      --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C_BAR --cache-dir cache --out-dir runs                                      # E74 (the stack inside ETA)
  python scripts/episodic/score_gradient_host.py --runs runs/eta_heldout4_${m}_cbar_ca_stack.json --stats stats/${m}_blocks.pt $V --no-headcov --out $O/eta_heldout4_${m}_cbar_ca_stack.json   # R61
done

# the stack inside ETA on the other cells of the grid (Table 3, last column; addendum 3): ViT-B/16 and ViT-S/16 on ImageNet-C heldout3
# (6 / 4, 11, theta 1.5), DeiT-B/16 and DeiT-S/16 on ImageNet-C heldout4, ImageNet-R and ImageNet-Sketch heldout, ViT-B/16 and ViT-S/16 on Sketch
for m in vitb vits; do
  python scripts/episodic/run_gradient_host.py --method eta --model $m --split heldout3 --arms tok+eta+rule_t6_b4b11 --exit-theta 1.5 --stats stats/${m}_blocks.pt --tag _stack \
      --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C --cache-dir cache --out-dir runs                                          # E75
  python scripts/episodic/score_gradient_host.py --runs runs/eta_heldout3_${m}_stack.json --stats stats/${m}_blocks.pt $V --no-headcov --out $O/eta_heldout3_${m}_stack.json   # R62
done
for pair in deitb:inc deitb:r deitb:sketch deits:inc deits:r deits:sketch vitb:r vitb:sketch vits:r vits:sketch; do
  m=${pair%%:*}; ds=${pair##*:}; split=$(heldsplit $ds)
  read -r T A1 A2 TH < <(python -c "import json; c=json.load(open('$O/calib_${m}_$ds.json')); print(c['trunk'], c['shallower'], c['deeper'], c['theta'])")
  python scripts/episodic/run_gradient_host.py --method eta --model $m --split $split $(dsflag $ds) --arms tok+eta+rule_t${T}_b${A1}b${A2} --exit-theta $TH --stats stats/${m}_blocks.pt --tag _ca_${ds}_stack \
      --imagenet-val $IMAGENET_VAL --imagenet-c $(root_of $ds) --cache-dir cache --out-dir runs                                      # E76
  python scripts/episodic/score_gradient_host.py --runs runs/eta_${split}_${m}_ca_${ds}_stack.json --stats stats/${m}_blocks.pt $V --no-headcov --out $O/eta_${split}_${m}_ca_${ds}_stack.json   # R63
done

# one calibration per shift from its first K design batches (K = 3, 10, 1: 192, 640, 64 images), held out (Table 2; addendum 5): the picks of
# each cell's first K batches on its heldout4 images, one run and one scored file per cell (K = 3 without the k suffix in the file name)
for bench in inc cbar; do
  if [ $bench = inc ]; then root=$IMAGENET_C; ss=$O/rule_sample_size_design_vitb.json; neo=$O/forward_only_heldout4_vitb_exit.json
  else root=$IMAGENET_C_BAR; ss=$O/rule_sample_size_design_vitb_cbar.json; neo=$O/forward_only_heldout4_vitb_cbar_exit.json; fi
  for K in 3 10 1; do
    if [ $K = 3 ]; then suffix=fewshot; else suffix=fewshot_k$K; fi
    python -c "
import json
for cell, (t, e, l) in json.load(open('$ss'))['per_cell_first_k_batches']['$K']['picks'].items(): print(cell, t[4:], e[4:], l[4:])" | while read -r cell T E L; do
      python scripts/episodic/run_forward_only.py --split heldout4 --cells $cell --arms tok+place_b$T rule_t${T}_b${E}b${L}_gated --exit-theta 1.5 --stats $SB --tag _${suffix}_$cell \
          --imagenet-val $IMAGENET_VAL --imagenet-c $root --cache-dir cache --out-dir runs                                           # E69 (one run per cell and K)
      python scripts/episodic/score_forward_only.py --runs runs/forward_only_heldout4_vitb_${suffix}_$cell.json $V --neo-from $neo --out $O/forward_only_heldout4_vitb_${suffix}_$cell.json   # R56
    done
  done
done
python scripts/episodic/run_forward_only.py --split heldout4 --cells clean $CBARCELLS --arms rule_t7_b5b8_gated --exit-theta 1.5 --stats $SB --tag _cbar_rule_p192 \
    --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C_BAR --cache-dir cache --out-dir runs                                        # E70 (the pooled 192-per-cell C-bar pick)
python scripts/episodic/score_forward_only.py --runs runs/forward_only_heldout4_vitb_cbar_rule_p192.json $V --neo-from $O/forward_only_heldout4_vitb_cbar_exit.json --out $O/forward_only_heldout4_vitb_cbar_rule_p192.json   # R57

# the time columns of Table 1 and Table S6 (an idle GPU, the shared one measures the contention): every row of Table 1 in one run,
# then PEA and FOA (time_others.py, the timm 1.0.15 environment of E15 E16); score_timing.py gives ms per batch and the ratio to NEO
python scripts/cost/time_batches.py --arms neo tok_k16_fp32_stats32 pair_k16_fp32_stats32 exit_b5b9_k16_fp32_stats32 rule_t6_b4b11_k16_fp32_stats32 \
    rule_t6_b4_k16_fp32_stats32 rule_t6_b11_k16_fp32_stats32 rule_t6_b4b11_random_k16_fp32_stats32 eta sar --repeats 3 --stats $SB --tag _ablations $D   # C6
python scripts/cost/score_timing.py --runs runs/timing_design_vitb_ablations.json $V --out $O/timing_design_vitb_ablations.json   # R64
$PY_TIMM1 scripts/cost/time_others.py --repeats 2 --stats $SB $D                                                                           # C7 (PEA, PEA with views, FOA)
python scripts/cost/score_timing.py --runs runs/timing_design_vitb_others.json $V --out $O/timing_design_vitb_others.json         # R65

# [P] numbers, figures and the report
python scripts/analysis/review_checks.py --logits runs/logits --imagenet-val $IMAGENET_VAL --scores results/scores --out results/scores/review_checks.json  # A15 (CPU, labels only to score: intervals of the held-out gains, the correlation without blocks 1-3, the threshold, every pair of branch blocks, Table S5)
python scripts/make_paper_numbers.py                                                                            # P1
python paper/make_numbers.py && python paper/make_figures.py                                                    # P2
python paper/make_example_figures.py --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C --cache-dir cache --stats stats/vitb.pt --logits runs/logits  # P5 (CPU, model-free; Figure S1 from the design images)
python scripts/analysis/number_inventory.py                                                                     # P3 (results/number_inventory.csv)
(cd paper/report && pdflatex report && bibtex report && pdflatex report && pdflatex report)                     # P4
