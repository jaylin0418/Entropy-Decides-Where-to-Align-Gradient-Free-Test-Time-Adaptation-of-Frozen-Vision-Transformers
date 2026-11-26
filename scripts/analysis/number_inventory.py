"""Rebuilds results/number_inventory.csv: one row per number printed in the report (paper/report/report.tex), with its macro key, the results/paper_numbers.json path or formula behind it, the file and command that
produced it, the steps of reproduce_paper.sh that regenerate it, and its verification status. Macro occurrences are read
from the .tex files and valued from paper/numbers.tex; literal numbers (settings, cost accounting, quotations, bounds
stated in words) are classified by the tables below. Provenance per macro key comes from PROVENANCE (prefix rules) and,
for the keys of the pixel-site study, from the committed inventory itself (hand-written when those runs were ported).

    python scripts/analysis/number_inventory.py            # rewrites results/number_inventory.csv
    python scripts/analysis/number_inventory.py --check    # exit 1 if the committed file differs
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'results' / 'number_inventory.csv'
TEX = {'report': ROOT / 'paper' / 'report' / 'report.tex'}
FIELDS = ['paper', 'line', 'section', 'table_or_figure', 'printed', 'macro_key', 'value (results/paper_numbers.json path or formula)',
          'research result file', 'research commands', 'steps here (reproduce_paper.sh)', 'status']
# ---- provenance of the where-to-align study (runs made with this repository's scripts; prefix -> columns)
RUN = 'run with this repository\'s scripts on the course checkpoint (GPU, 2026-11-08/09, data/derived/course_two_site); scored file committed'
HRUN = 'run with this repository\'s scripts on the course checkpoint (GPU, 2026-11-09, data/derived/course_two_site, tag _hier); scored file committed'
CPU = 'computed by this repository\'s analysis script from the committed scores (CPU, model-free)'
ERUN = 'run with this repository\'s scripts on the course checkpoint (GPU, 2026-11-09, data/derived/course_two_site, tag _extra); scored file committed'
CRUN = ('run with this repository\'s scripts on the course checkpoint on ImageNet-C-bar (the authors\' archive; GPU, 2026-11-09, '
        'data/derived/course_two_site, tag _cbar; the heldout3 run scored once); scored file committed')
SRUN = ('run with this repository\'s scripts on the course checkpoint (GPU, 2026-11-09, data/derived/course_two_site; the six arms of the earlier SAR '
        'run came out identical to the earlier scored file, the main-path arms were added); scored file committed')
TS = [  # (key prefix, value path or formula, scored files, run commands, steps, status)
    ('cost.ex.', 'extra/timing/<arm>/{ms_per_batch, x_neo, acc_6cells} (six timing cells, one run, main-path arms)', 'timing_design_vitb_extra',
     'time_batches.py --arms neo stack_k16_fp32_stats32 twosite_k16_fp32_stats32 twosite_k1 eta eta+twosite --tag _extra; score_timing.py', 'S5 C2 R31 P1 P2', ERUN),
    ('cb.d.', 'extra/cbar/design/{episodic, clean, per_cell} (20 ImageNet-C-bar cells; per-corruption rows = mean of severities 3 and 5)', 'forward_only_design_vitb_cbar',
     'run_forward_only.py --split design --arms neo tok place_b6 tok+place_b6 --cells clean <ImageNet-C-bar cells> --imagenet-c $IMAGENET_C_BAR --tag _cbar; score_forward_only.py', 'S5 E28 R33 P1 P2', CRUN),
    ('cb.h.', 'extra/cbar/held3/{episodic, clean, per_cell} (heldout3, scored once)', 'forward_only_heldout3_vitb_cbar',
     'run_forward_only.py --split heldout3 --arms neo tok place_b6 tok+place_b6 --cells clean <ImageNet-C-bar cells> --imagenet-c $IMAGENET_C_BAR --tag _cbar; score_forward_only.py', 'S5 E29 R34 P1 P2', CRUN),
    ('cb.n.', 'extra/cbar/design/{cells, n_cells} (counts)', 'forward_only_design_vitb_cbar', 'as cb.d.*', 'R33 P1 P2', CPU),
    ('ex.h.', 'exit/held4/{inc,cbar}/{episodic, clean, per_cell, routed}: the pre-registered heldout4 test of the gated early-exit stack (both benchmarks, scored once)',
     'forward_only_heldout4_vitb_exit, forward_only_heldout4_vitb_cbar_exit',
     'run_forward_only.py --split heldout4 --arms neo tok place_b6 tok+place_b6 tok+place_b6+headcov pair_gated exit_b9_gated exit_b5b9_gated --tag _exit | --cells clean <ImageNet-C-bar cells> --tag _cbar_exit; score_forward_only.py', 'S5 E38 E39 R39 P1 P2',
     "run with this repository's scripts on the course checkpoint (GPU, 2026-11-10, data/derived/course_two_site, tag _exit; heldout4 scored once); scored file committed"),
    ('ex.st.', 'exit/streams/table: the cascade on the continual and clean-return streams of heldout4 (scored once)', 'streams_heldout4_vitb_exit',
     'run_streams.py --split heldout4 --arms neo pair_gated exit_b5b9_gated --streams cont5 cleanret --tag _exit; score_streams.py', 'S5 T5 R44 P1 P2',
     "run with this repository's scripts on the course checkpoint (GPU, 2026-11-10, data/derived/course_two_site, tag _exit; streams, scored once); scored file committed"),
    ('sc.so.', 'site_choice/{inc,cbar}/second_opinions: why blocks 5 and 9 (agreement, rescue, harm, single and paired routed mixtures on the dev split, from the saved logits) and the cost in forward passes of the pair counted in blocks',
     'site_choice_design_vitb, site_choice_design_vitb_cbar', 'site_choice.py (second_opinions)', 'S5 E19 E28 E34 E35 A6 A7 P1 P2',
     "computed by this repository's analysis script from the saved logits of the dev runs (CPU, model-free)"),
    ('ex.ctl.', 'exit/controls/{inc,cbar}: equal-compute controls of the cascade on heldout4 (random routing, equal weights, every image routed), scored once',
     'forward_only_heldout4_vitb_exitctl, forward_only_heldout4_vitb_cbar_exitctl',
     'run_forward_only.py --split heldout4 --arms exit_b5b9_random_gated exit_b5b9_equal_gated mix_b5b9_gated --tag _exitctl | --cells clean <ImageNet-C-bar cells> --tag _cbar_exitctl; score_forward_only.py --neo-from', 'S5 E48 E49 R43 P1 P2',
     "run with this repository's scripts on the course checkpoint (GPU, 2026-11-10, data/derived/course_two_site, tag _exitctl; scored once); scored file committed"),
    ('ex.vs.', 'exit/vits/{design,held3}: the cascade on ViT-S/16 with the ViT-B/16 blocks and threshold (heldout3 scored once)',
     'forward_only_design_vits_exit, forward_only_heldout3_vits_exit, forward_only_heldout3_vits',
     'run_forward_only.py --model vits --split design --arms tok+place_b6 neo pair_gated exit_b9_gated exit_b5b9_gated --tag _exit; --split heldout3 --arms pair_gated exit_b9_gated exit_b5b9_gated --tag _exit; score_forward_only.py', 'S5 E50 E51 R43 P1 P2',
     "run with this repository's scripts on the course checkpoint (GPU, 2026-11-10, data/derived/course_two_site, tag _exit; ViT-S/16); scored file committed"),
    ('rl.h.', 'rule/held4/{inc,cbar,inc_abl,cbar_abl,inc_small,cbar_small,inc_ctl,cbar_ctl}: the rule-chosen (calibrated) cascade, its ablations, the small-sample calibration and three control pairs (3 and 9, 1 and 12, 5 and 7) on heldout4, each file scored once (docs/heldout_test_plan_rule.md)',
     'forward_only_heldout4_vitb_{rule,cbar_rule,rule_abl,cbar_rule_abl,rule_small,cbar_rule_small,rule_ctl,cbar_rule_ctl}',
     'run_forward_only.py --split heldout4 --arms rule_t6_b4b11_gated | rule_t6_b4b11_gated rule_t7_b5b11_gated tok+place_b7 | rule_t6_b4_gated rule_t6_b11_gated | rule_t5_b3b7_gated tok+place_b5 | rule_t7_b5b9_gated | rule_t6_b3b9_gated rule_t6_b1b12_gated rule_t6_b5b7_gated (--cells clean <ImageNet-C-bar cells>); score_forward_only.py --neo-from', 'S5 E52 E53 E54 E55 E56 E57 E77 E78 R45 R46 R66 R67 P1 P2',
     "run with this repository's scripts on the course checkpoint (GPU, 2026-11-10/11 and 2026-11-22, data/derived/course_two_site, tags _rule _cbar_rule _rule_abl _cbar_rule_abl _rule_small _cbar_rule_small _rule_ctl _cbar_rule_ctl; heldout4 scored once); scored file committed"),
    ('rl.vs.', 'rule/vits: ViT-S/16 calibrated by the rule on its own dev sweep (A11, A12) and run once on heldout3', 'forward_only_design_vits_rule_sweep, site_choice_design_vits_rule_t6, site_choice_design_vits_rule, forward_only_heldout3_vits_rule',
     'run_forward_only.py --model vits --split design --arms neo tok place_b6 tok+place_b1..b12 --tag _rule_sweep; site_choice.py --trunk 6 | --trunk <pick>; run_forward_only.py --model vits --split heldout3 --arms <rule arms> --tag _rule', 'S5 E58 A11 A12 E59 R47 R48 P1 P2',
     "run with this repository's scripts on the course checkpoint (GPU, 2026-11-10/11, data/derived/course_two_site, tag _rule_sweep _rule; ViT-S/16, heldout3 scored once); scored file committed"),
    ('rl.sl.', 'rule/slices: the calibrated cascade (trunk 6, blocks 4 and 11) on the other held-out image sets, ImageNet-C-bar heldout3 and the dev split, scored once',
     'forward_only_{heldout,heldout2,heldout3}_vitb_rule, forward_only_heldout3_vitb_cbar_rule, forward_only_design_vitb_rule, forward_only_design_vitb_cbar_rule',
     'run_forward_only.py --split heldout|heldout2|heldout3|design --arms rule_t6_b4b11_gated --tag _rule (| --cells <ImageNet-C-bar cells> --tag _cbar_rule); score_forward_only.py --neo-from', 'S5 E60 E61 E62 E63 E81 E82 R49 R50 R71 R72 P1 P2',
     "run with this repository's scripts on the course checkpoint (GPU, 2026-11-11, data/derived/course_two_site, tags _rule _cbar_rule; each set scored once); scored file committed"),
    ('rl.ctl.', 'rule/random/{inc,cbar} and rule/views/{inc,cbar}: the calibrated cascade with 35 % of the images routed at random, and the main path on three views (identity, flip, center crop) of the same unsure images instead of two more blocks (heldout4, scored once)', 'forward_only_heldout4_vitb_rule_random, forward_only_heldout4_vitb_cbar_rule_random, forward_only_heldout4_vitb_rule_views, forward_only_heldout4_vitb_cbar_rule_views',
     'run_forward_only.py --split heldout4 --arms rule_t6_b4b11_random_gated --tag _rule_random | _cbar_rule_random; --arms rule_t6_views_gated --tag _rule_views | _cbar_rule_views; score_forward_only.py --neo-from', 'S5 E64 E65 E79 E80 R45 R46 R69 R70 P1 P2',
     "run with this repository's scripts on the course checkpoint (GPU, 2026-11-11 and 2026-11-22, data/derived/course_two_site; scored once); scored files committed"),
    ('rl.st.', 'rule/streams/table: the calibrated cascade, and ETA, SAR and the main path inside ETA (weights carried across the cells, scored alone and with NEO on the recorded head input), on the continual and clean-return streams of heldout4 (scored once)', 'streams_heldout4_vitb_rule, streams_heldout4_vitb_eta',
     'run_streams.py --split heldout4 --arms neo pair_gated exit_b5b9_gated rule_t6_b4b11_gated --streams cont5 cleanret --tag _rule | --arms neo eta sar twosite+eta_gated --tag _eta; score_streams.py', 'S5 T6 T8 R51 R68 P1 P2',
     "run with this repository's scripts on the course checkpoint (GPU, 2026-11-11 and 2026-11-22, data/derived/course_two_site, tags _rule _eta; streams, scored once); scored files committed"),
    ('rl.macs.', 'analytic compute in GMACs per image (paper/compute_cost.py, one forward pass = 17.56 GMACs: one d x d multiply per token per map, the branches on the unsure share of the held-out run, a backward pass as two forward passes, PEA two passes with a map after every block, FOA 28 forward passes), for the rows of Table 1 and Table S6', 'forward_only_heldout4_vitb_{rule,rule_abl,rule_random}, forward_only_heldout4_vitb_exit (unsure shares)', 'forward_only_heldout4_vitb_rule; forward_only_heldout4_vitb_rule_abl; forward_only_heldout4_vitb_rule_random; forward_only_heldout4_vitb_exit', 'E38 E52 E54 E64 R38 R45 P1 P2', 'analytic'),
    ('rl.cost.', 'rule/timing and rule/timing_others: wall-clock of every row of Table 1 (NEO, TokAlign, the main path, blocks 5/9, the calibrated cascade, each extra branch alone, random routing, ETA, SAR) and of PEA and FOA on the six timing cells (one run each, the GPU idle)', 'timing_design_vitb_ablations, timing_design_vitb_others',
     'time_batches.py --arms neo tok pair exit_b5b9 rule_t6_b4b11 rule_t6_b4 rule_t6_b11 rule_t6_b4b11_random (k16_fp32_stats32) eta sar --tag _ablations; time_others.py; score_timing.py', 'S5 C6 R64 C7 R65 P1 P2',
     "run with this repository's scripts on the course checkpoint (GPU, 2026-11-16, data/derived/course_two_site, tags _ablations _others, the GPU idle); scored files committed"),
    ('ca.', 'ca/<model>_<dataset>: the calibrated cascade on ViT-S/16, DeiT-B/16 and DeiT-S/16 and on ImageNet-C-bar, ImageNet-R and ImageNet-Sketch (dev sweep, label-free calibration, held-out run, ETA on the same cells; docs/heldout_test_plan_models_datasets.md)',
     'forward_only_design_<model>_ca_<dataset>, site_choice_design_<model>_ca_<dataset>{,_t6}, calib_<model>_<dataset>, forward_only_<split>_<model>_ca_<dataset>, eta_<split>_<model>_ca_<dataset>',
     'scripts/cvpr/course_ca.sh (research repository): compute_source_stats.py --blocks; run_forward_only.py --split design --dataset r|sketch --arms neo tok place_b6 tok+place_b1..b12; site_choice.py; run_forward_only.py --split heldout|heldout4 --exit-theta; run_gradient_host.py --method eta', 'S5 E66 R53 E76 R63 P1 P2',
     "run with this repository's scripts on the course checkpoint (GPU, 2026-11-11, data/derived/course_two_site, tags _ca_inc _ca_r _ca_sketch; held-out cells scored once); scored file committed"),
    ('ax.', 'the cross-model table: per pair the gains over NEO of the main path, the calibrated cascade and ETA on the same held-out cells (from two_site/held3_*, rule/*, ca/*)', 'forward_only_heldout3_{vitb,vits}, forward_only_heldout4_vitb_cbar_rule, forward_only_heldout{,4}_<model>_ca_<dataset>, eta_heldout{3,4,}_*',
     'paper/make_numbers.py (across)', 'P1 P2', "computed by paper/make_numbers.py from the committed scores (CPU)"),
    ('cost.cfg.', 'the six timing cells (scripts/cost/time_batches.py CELLS): mean top-1 of the cascade (rule/slices/design/per_cell) and the main path (two_site/design_vitb/per_cell) in the reported setting, against rule/timing/<arm>/acc_6cells of the timed setting', 'forward_only_design_vitb_rule; forward_only_design_vitb_two_site; timing_design_vitb_ablations',
     'paper/make_numbers.py (rule)', 'S5 E19 R22 E81 R71 C6 R64 P1 P2', "computed by paper/make_numbers.py from the committed scores (CPU)"),
    ('cost.timed.', 'rule/timing/<arm>/acc_6cells: top-1 of the timed setting (map refreshed every 16 batches, float32) on the six timing cells', 'timing_design_vitb_ablations',
     'time_batches.py (C6); score_timing.py (R64)', 'S5 C6 R64 P1 P2', "computed by paper/make_numbers.py from the committed scores (CPU)"),
    ('abl.src3000', 'design_vitb/ablation/src3000: the tok+headcov arm of the dev run with the 3,000-image source statistics (the reference of the src50 / src300 sensitivity)', 'forward_only_design_vitb',
     'run_forward_only.py --split design --arms tok+headcov (E1)', 'S1 E1 R1 P1 P2', "computed by paper/make_numbers.py from the committed scores (CPU)"),
    ('sg.', 'block 1 among the 13 placements on the eight corruption pairs (Section C.2): its rank by accuracy and by mean entropy from the label_free entries of exit/site_choice/{inc,cbar}, rule/vits/pick_t6 and ca/<pair>/signal', 'site_choice_design_vitb, site_choice_design_vitb_cbar_t7, site_choice_design_vits_rule, site_choice_design_{vits_cbar_ca,deitb_ca_inc,deitb_cbar_ca,deits_ca_inc,deits_cbar_ca}_t6',
     'paper/make_numbers.py (across)', 'P1 P2', "computed by paper/make_numbers.py from the committed scores (CPU)"),
    ('cost.mainpath.', 'paper/compute_cost.py MAP: two maps of 196 x 768 x 768 multiply-accumulates per image against 17.56 GMACs, in percent', '',
     'paper/compute_cost.py', 'P1 P2', 'analytic'),
    ('rl.pick.', 'rule/picks/{inc,cbar_t6,cbar_t7}: the main block and the pair of branch blocks the entropy rule picks on the design logits, and the gain of the pick against the best labeled pair', 'site_choice_design_vitb, site_choice_design_vitb_cbar, site_choice_design_vitb_cbar_t7',
     'site_choice.py (label_free, --trunk 6 | 7)', 'S5 E19 E28 E34 E35 A6 A7 A8 P1 P2', "computed by this repository's analysis script from the saved logits of the dev runs (CPU, model-free)"),
    ('rl.ss.', 'rule/sample_size/{inc,cbar}: the rule applied to the first n images of every cell (pooled) and to the first K batches of each cell (one choice per shift), from the saved design logits', 'rule_sample_size_design_vitb, rule_sample_size_design_vitb_cbar',
     'rule_sample_size.py', 'S5 E19 E28 E34 E35 A9 A10 P1 P2', "computed by this repository's analysis script from the saved logits of the dev runs (CPU, model-free)"),
    ('rl.fs.', 'rule/fewshot/{inc,cbar}/k{3,10,1}: one calibration per shift from the first K dev batches of each cell (192, 640 or 64 images), its stack and main path run on the heldout4 images of that cell, one scored file per cell and K, means over the cells against the full-sample calibration', 'forward_only_heldout4_vitb_fewshot{,_k10,_k1}_<cell>, rule_sample_size_design_vitb{,_cbar}, forward_only_heldout4_vitb{,_cbar}_exit',
     'run_forward_only.py --split heldout4 --cells <cell> --arms tok+place_bT rule_tT_bEbL_gated --exit-theta 1.5 per cell and K; score_forward_only.py --neo-from', 'S5 E69 R56 P1 P2', 'run with this repository\'s scripts (GPU, 2026-11-12 to 2026-11-15, data/derived/course_two_site, tags _fewshot_<cell> _fewshot_k10_<cell> _fewshot_k1_<cell>); scored files committed'),
    ('rl.ss.cbar.p192', 'rule/pooled192_cbar: the ImageNet-C-bar pick from 192 images per cell pooled over the benchmark (7 / 5, 8) on heldout4, scored once', 'forward_only_heldout4_vitb_cbar_rule_p192',
     'run_forward_only.py --split heldout4 --arms rule_t7_b5b8_gated --exit-theta 1.5 (ImageNet-C-bar cells)', 'S5 E70 R57 P1 P2', 'run with this repository\'s scripts (GPU, 2026-11-12, data/derived/course_two_site, tag _cbar_rule_p192); scored file committed'),
    ('rc.', 'rule/review: paired bootstrap of the held-out gains over images and over cells, the Spearman correlation without blocks 1 to 3, the entropy the pick saves against TokAlign alone, the threshold sweep and the gated gain of every pair of branch blocks on the ViT-B/16 dev predictions, all from the saved logits of the runs above (scripts/analysis/review_checks.py, CPU, labels only to score)',
     'review_checks',
     'scripts/analysis/review_checks.py --logits runs/logits (the held-out and design logits of the runs above)', 'A15 P1 P2', 'run with this repository\'s scripts (CPU, 2026-11-22, from the saved logits in data/derived/course_two_site/runs/logits); scored file committed'),
    ('ex.sl.', 'exit/slices/{heldout,heldout2,heldout3,cbar_heldout3} and exit/held4: the gated early-exit arms on the four ImageNet-C held-out slices and two ImageNet-C-bar slices, each scored once',
     'forward_only_{heldout,heldout2,heldout3}_vitb_exit, forward_only_heldout3_vitb_cbar_exit, forward_only_heldout4_vitb_exit, forward_only_heldout4_vitb_cbar_exit',
     'run_forward_only.py --split heldout|heldout2|heldout3 --arms pair_gated exit_b9_gated exit_b5b9_gated --tag _exit (| --cells clean <ImageNet-C-bar cells> --tag _cbar_exit); score_forward_only.py --neo-from', 'S5 E44 E45 E46 E47 R42 P1 P2',
     "run with this repository's scripts on the course checkpoint (GPU, 2026-11-10, data/derived/course_two_site, tag _exit; each slice scored once, after the heldout4 verdict); scored file committed"),
    ('ex.d.', 'exit/design/{inc,cbar} and exit/blocks/{inc,cbar}: the early-exit arms and TokAlign plus LayerAlign after every block on the dev split',
     'forward_only_design_vitb_exit, forward_only_design_vitb_cbar_exit, forward_only_design_vitb_select, forward_only_design_vitb_cbar_select',
     'run_forward_only.py --split design --arms exit_b9 exit_b5b9 exit_b5b9_gated pair_gated --tag _exit | _cbar_exit; --arms tok+place_b1..b12 --tag _select | _cbar_select; score_forward_only.py', 'S5 E34 E35 E36 E37 R39 P1 P2',
     "run with this repository's scripts on the course checkpoint (GPU, 2026-11-09/10, data/derived/course_two_site, tags _select _exit); scored file committed"),
    ('ex.cost.', 'exit/timing: wall-clock of NEO, the pair, the branch mixture and the early-exit arms on the six timing cells (one run)', 'timing_design_vitb_exit',
     'time_batches.py --arms neo pair_k16_fp32_stats32 branch_k16_fp32_stats32 exit_b9_k16_fp32_stats32 exit_b5b9_k16_fp32_stats32 --tag _exit; score_timing.py', 'S5 C3 R40 P1 P2',
     "run with this repository's scripts on the course checkpoint (GPU, 2026-11-10, data/derived/course_two_site, tag _exit); scored file committed"),
    ('ex.theta', 'the early-exit threshold, a setting fixed on the dev split', '-', '-', 'P1 P2', 'setting'),
    ('sc.', 'site_choice/{inc,cbar}: per-cell and per-image choices of the block from the saved design logits (CPU, model-free given the logits)',
     'site_choice_design_vitb, site_choice_design_vitb_cbar', 'site_choice.py --base forward_only_design_vitb_two_site --blocks forward_only_design_vitb_select | the _cbar runs', 'S5 E19 E28 E34 E35 A6 A7 P1 P2',
     "computed by this repository's analysis script from the saved logits of the dev runs (CPU, model-free)"),
    ('ts.sl.', 'two_site/slices_vitb/{heldout, heldout2}/{episodic, clean} and two_site/held3_vitb/episodic: the heldout3 arms on the heldout and heldout2 slices, each scored once; mean and range over the three slices',
     'forward_only_{heldout,heldout2}_vitb_two_site, eta_{heldout,heldout2}_vitb_two_site, forward_only_heldout3_vitb, eta_heldout3_vitb',
     'run_forward_only.py --split heldout|heldout2 --arms neo tok headcov tok+headcov place_b6 tok+place_b6 tok+place_b6+headcov --tag _two_site; run_gradient_host.py --method eta --split heldout|heldout2 --arms eta tok+eta eta+place_b6 tok+eta+place_b6 --tag _two_site; score_*.py', 'S5 E30 E31 E32 E33 R35 R36 R37 R38 P1 P2',
     "run with this repository's scripts on the course checkpoint (GPU, 2026-11-09, data/derived/course_two_site, tag _two_site; each slice scored once); scored file committed"),
    ('sar.ts.h.', 'extra/sar_held3/{episodic, clean, per_cell} (heldout3, scored once)', 'sar_heldout3_vitb',
     'run_gradient_host.py --method sar --split heldout3 --arms sar tok+sar sar+place_b6 tok+sar+place_b6; score_gradient_host.py', 'S5 E27 R32 P1 P2', RUN),
    ('sar.ts.', 'design_vitb/{episodic, clean}/{sar, tok+sar, sar+place_b6, tok+sar+place_b6}{, +neo}; extra/sar_design/per_cell (per-family gains)', 'sar_design_vitb',
     'run_gradient_host.py --method sar --split design --arms sar tok+sar sar+place_b6 tok+sar+place_b6; score_gradient_host.py', 'S5 E14 R4 P1 P2', SRUN),
    ('ts.sw.frozen.', 'two_site/sweep_vitb/frozen_gain (mean over 30 cells of place_bk - neo; pixel = tok)', 'forward_only_design_vitb_sweep; forward_only_design_vitb_two_site',
     'run_forward_only.py --split design --arms place_b1..b12 --tag _sweep; run_forward_only.py --split design --arms neo tok ... --tag _two_site', 'S5 E19 E24 R22 R24 P1 P2', RUN),
    ('ts.sw.eta.', 'two_site/sweep_vitb/eta_gain (mean over 30 cells of eta+place_bk+neo - eta+neo; pixel = tok+eta+neo)', 'eta_design_vitb_sweep; eta_design_vitb',
     'run_gradient_host.py --method eta --split design --arms eta+place_b1..b12 --tag _sweep', 'S5 E25 R25 P1 P2', RUN),
    ('ts.sw.', 'two_site/sweep_vitb/frozen_per_cell (best placement per corruption, mean of severities 3 and 5)', 'forward_only_design_vitb_sweep; forward_only_design_vitb_two_site',
     'as ts.sw.frozen.*', 'S5 E19 E24 R22 R24 P1 P2', RUN),
    ('ts.dr.', 'two_site/drift_vitb/z_spec_first_k (z of the spectral drift, frozen pass, first 5 batches; best site from sweep_vitb)', 'drift_by_depth_design_vitb; forward_only_design_vitb_sweep',
     'drift_by_depth.py --split design; score_drift_by_depth.py', 'S5 F9 R26 P1 P2', RUN),
    ('ts.d.', 'two_site/design_vitb/{episodic, clean}', 'forward_only_design_vitb_two_site', 'run_forward_only.py --split design --arms neo tok headcov tok+headcov place_b6 tok+place_b6 tok+place_b6+headcov --tag _two_site', 'S5 E19 R22 P1 P2', RUN),
    ('ts.sel.', 'two_site/selection_vitb/{mean_30_cells, gain_over_neo, site_counts}', 'site_selection_design_vitb (from forward_only_design_vitb_two_site, _sweep and drift_by_depth_design_vitb)',
     'site_selection.py', 'R27 P1 P2', CPU),
    ('ts.hb.co.', 'two_site/held3_vitb/streams/<arm>/cont (cont5)', 'streams_heldout3_vitb', 'run_streams.py --split heldout3 --streams cont5 cleanret --arms neo stack_gated twosite_gated eta tok+eta_gated twosite+eta_gated', 'S5 T7 R28 P1 P2', RUN),
    ('ts.hb.cr.', 'two_site/held3_vitb/streams/<arm>/cleanret_clean', 'streams_heldout3_vitb', 'as ts.hb.co.*', 'S5 T7 R28 P1 P2', RUN),
    ('ts.hb.gate.', 'two_site/held3_vitb/gate_cleanret/twosite_gated', 'streams_heldout3_vitb', 'as ts.hb.co.*', 'S5 T7 R28 P1 P2', RUN),
    ('ts.hb.c5.', 'two_site/held3_vitb/cont5_seg', 'streams_heldout3_vitb', 'as ts.hb.co.*', 'S5 T7 R28 P1 P2', RUN),
    ('ts.hb.', 'two_site/held3_vitb/{episodic, clean, per_cell_episodic} (ETA arms: per_cell_eta)', 'forward_only_heldout3_vitb; eta_heldout3_vitb',
     'run_forward_only.py --split heldout3 --arms neo tok headcov tok+headcov place_b6 tok+place_b6 tok+place_b6+headcov; run_gradient_host.py --method eta --split heldout3 --arms eta tok+eta eta+place_b6 tok+eta+place_b6', 'S5 E20 E21 R22 R23 P1 P2', RUN),
    ('ts.hs.', 'two_site/held3_vits/{episodic, clean, per_cell_episodic} (ETA arms: per_cell_eta)', 'forward_only_heldout3_vits; eta_heldout3_vits',
     'run_forward_only.py --model vits --split heldout3 ...; run_gradient_host.py --method eta --model vits --split heldout3 ...', 'S6 E22 E23 R22 R23 P1 P2', RUN),
    ('ts.fam.', 'two_site/held3_vitb/per_cell_episodic and per_cell_eta (mean of severities 3 and 5)', 'forward_only_heldout3_vitb; eta_heldout3_vitb', 'as ts.hb.*', 'S5 E20 E21 R22 R23 P1 P2', RUN),
    ('ts.gainfam.', 'two_site/held3_vitb/per_cell_episodic (gain over neo, mean of severities 3 and 5)', 'forward_only_heldout3_vitb', 'as ts.hb.*', 'S5 E20 R22 P1 P2', RUN),
    ('ts.grp.', 'two_site/held3_vitb/per_cell_episodic (mean gain over neo of a corruption family)', 'forward_only_heldout3_vitb', 'as ts.hb.*', 'S5 E20 R22 P1 P2', RUN),
    ('hi.d.', 'hier/design_vitb/{episodic, clean} against two_site/design_vitb/{episodic, clean} (NEO, tok, place_b6, tok+place_b6)', 'forward_only_design_vitb_hier; forward_only_design_vitb_two_site',
     'run_forward_only.py --split design --arms hier_px hier_b6 hier_px+hier_b6 tok+hier_b6 within_b6 between_b6 hier_b6_n0x4 hier_b6_n0d4 --tag _hier; score_forward_only.py --neo-from forward_only_design_vitb_two_site.json', 'S7 E26 R29 P1 P2', HRUN),
    ('hi.grp.', 'hier/design_vitb/per_cell and two_site/design_vitb/per_cell (mean gain over neo of a corruption family, severities 3 and 5)', 'forward_only_design_vitb_hier; forward_only_design_vitb_two_site', 'as hi.d.*', 'S7 E26 R29 P1 P2', HRUN),
    ('hi.corr.', 'hier/design_vitb/per_cell minus two_site/design_vitb/per_cell (two-level arm against its pooled arm per corruption, mean of severities 3 and 5)', 'forward_only_design_vitb_hier; forward_only_design_vitb_two_site', 'as hi.d.*', 'S7 E26 R29 P1 P2', HRUN),
    ('rb.loo.', 'two_site/sweep_vitb/frozen_per_cell and neo_per_cell (block chosen on the dev sweep with one family or one corruption left out; gains on the family left out)', 'forward_only_design_vitb_sweep; forward_only_design_vitb_two_site',
     'paper/make_numbers.py robust() from results/paper_numbers.json', 'P1 P2', CPU),
    ('rb.gate.', 'two_site/design_vitb/per_cell (per-cell oracles of a gate per placement against tok+place_b6; cells where block 6 lowers the pair) and hier/decomp_vitb/mean (pixel within-image log trace ratio rule)', 'forward_only_design_vitb_two_site; decompose_by_depth_design_vitb',
     'paper/make_numbers.py robust() from results/paper_numbers.json', 'P1 P2', CPU),
    ('hi.dec.', 'hier/decomp_vitb/mean (log trace ratio of the within-image / between-image part to the source part, and the batch-mean shift; mean of severities 3 and 5)', 'decompose_by_depth_design_vitb',
     'decompose_by_depth.py --split design --stats stats/vitb_hier.pt; score_decompose_by_depth.py', 'S7 F10 R30 P1 P2', HRUN),
    ('ts.held.vs.design.max', 'max |two_site/held3_vitb/episodic - two_site/design_vitb/episodic| over the seven forward-only arms', 'forward_only_heldout3_vitb; forward_only_design_vitb_two_site', 'as ts.hb.* and ts.d.*', 'S5 E19 E20 R22 P1 P2', RUN),
]
# ---- literal numbers: category by printed text (the context of each is a setting unless listed here)
SETTING = ('hyper-parameter, data size or protocol constant of the runs (tokalign/aligner.py, tokalign/data.py, scripts/*; README Data preparation)')
LITERAL = {'66.3': ('quoted from another paper', 'FOA paper (66.3 % on 50,000-image domains)', 'citation'),
           '64.5': ('quoted from another paper', 'PEA paper, Table 1 (ViT-Base, ImageNet-C severity 5, continual, without augmentation)', 'citation'),
           '66.5': ('quoted from another paper', 'PEA paper, Table 1 (ViT-Base, ImageNet-C severity 5, continual, with augmentation)', 'citation'),
           '13.5': ('quoted from another paper', 'DDA (13.5 s per image)', 'citation'),
           '0.75': ('analytic cost', 'one 768x768 multiply per token and per image plus one eigendecomposition per refresh = about 0.12 GMACs per image = 0.75 % of 17.56 GMACs (scripts/analysis/frontier_and_drift.py MACS)', 'analytic'),
           '1.5': ('setting', 'the threshold theta = 1.5 nat, chosen with blocks 5 and 9 by the labeled dev search over ImageNet-C and ImageNet-C-bar (docs/heldout_test_plan_exit.md) before the quantile rule, used for the ViT-B/16 runs and ViT-S/16 on ImageNet-C (run_forward_only.py --exit-theta 1.5)', 'setting'),
           '28': ('cost accounting', 'FOA 27+1 forwards per batch', 'setting'), '25': ('bound stated in words', '50,000 / 2,000 = 25 times more batches', 'setting')}
COST_WORDS = ('forward-pass equivalents', 'forward passes', 'charged', 'MACs', 'multiply', 'multiplies')
SIGNAL_FILES = '; '.join(['site_choice_design_vitb', 'site_choice_design_vitb_cbar', 'site_choice_design_vits_rule_t6']
                         + sorted(p.stem for p in (ROOT / 'results' / 'scores').glob('site_choice_design_*_ca_*_t6.json')))
FIG_DATA = [  # figures: what paper/make_figures.py draws (rows with line 0)
    ('fig:whysites (Figure S5)', 'why two branches: agreement with the main path, single-branch gain per block, two-branch gains (exit/site_choice/*/second_opinions)', 'exit/site_choice/{inc,cbar}/second_opinions', 'site_choice_design_vitb; site_choice_design_vitb_cbar', 'A6 A7 P1 P2',
     "computed by this repository's analysis script from the saved logits of the dev runs (CPU, model-free)"),
    ('fig:framework (Figure S1)', 'one dev batch through the calibrated cascade: images, patches, covariances, gate scores, n (paper/figures/example_numbers.json); network rows schematic; inset = trunk entropy of the 2,000 dev images of the cell from the saved logits, routed share from forward_only_design_vitb_exit', 'paper/figures/example_numbers.json (dev images, source statistics, saved logits)', 'forward_only_design_vitb_two_site; forward_only_design_vitb_exit', 'S1 E19 E36 R39 P5',
     'recomputed model-free on the CPU from the cached dev images with tokalign/stages.py (the same map the episodic run applied)'),
    ('fig:depth (Figure 1)', 'lines: mean gain at every placement, frozen ViT and ETA; per-family lines', 'two_site/sweep_vitb', 'forward_only_design_vitb_sweep; forward_only_design_vitb_two_site; eta_design_vitb_sweep; eta_design_vitb', 'S5 E19 E24 E25 R22 R24 R25 P1 P2', RUN),
    ('fig:drift (Figure S4)', 'heat map: z of the spectral drift per corruption and placement; markers: best placement per corruption', 'two_site/drift_vitb; two_site/sweep_vitb', 'drift_by_depth_design_vitb; forward_only_design_vitb_sweep; forward_only_design_vitb_two_site', 'S5 F9 E19 E24 R22 R24 R26 P1 P2', RUN),
    ('fig:depthfam (Figure S2)', 'lines: mean gain at every placement per corruption family, frozen ViT', 'two_site/sweep_vitb', 'forward_only_design_vitb_sweep; forward_only_design_vitb_two_site', 'S5 E19 E24 R22 R24 P1 P2', RUN),
    ('fig:overview (Figure 2)', '(a) the calibration from the ImageNet-C design logits: mean entropy of the 13 placements, the entropy of the average with the main path per candidate block (exit/site_choice/inc/label_free); (b) one batch of the stack, schematic', 'exit/site_choice/inc/label_free', 'site_choice_design_vitb', 'S5 E19 E34 A6 P1 P2',
     "computed by this repository's analysis script from the saved logits of the dev runs (CPU, model-free)"),
    ('fig:frontier (Figure S7)', 'two scatters: top-1 on the six timing cells against compute in forward passes (paper/compute_cost.py) and against time per batch relative to NEO for NEO, TokAlign, the main path, the cascade, the random-image control, ETA, SAR, PEA, PEA with views and FOA (rule/timing, rule/timing_others)', 'timing_design_vitb_ablations, timing_design_vitb_others', 'timing_design_vitb_ablations; timing_design_vitb_others', 'S5 C6 R64 C7 R65 P1 P2', RUN),
    ('fig:signalgrid (Figure S3)', 'per model-dataset pair: accuracy and mean entropy of the 13 placements on the dev images, the pick and the most accurate (exit/site_choice/*, rule/vits/pick_t6, ca/*/signal)', 'exit/site_choice/{inc,cbar}/label_free, rule/vits/pick_t6, ca/<pair>/signal', SIGNAL_FILES, 'E19 E34 E35 E58 E66 A6 A7 A11 A13 P1 P2',
     "computed by this repository's analysis script from the saved logits of the dev runs (CPU, model-free)"),
    ('fig:calib (Figure S6)', 'gain of the calibrated cascade over the pair against the number of calibration images, pooled per benchmark and per shift (rule/sample_size/{inc,cbar})', 'rule/sample_size/{inc,cbar}', 'rule_sample_size_design_vitb; rule_sample_size_design_vitb_cbar', 'E19 E28 E34 E35 A9 A10 P1 P2',
     "computed by this repository's analysis script from the saved logits of the dev runs (CPU, model-free)"),
]
STRIP = [r'(?<!\\)%.*', r'\\(?:cite[pt]?|label|[cC]ref|ref|bibliography|bibliographystyle|input|includegraphics(?:\[[^\]]*\])?|setlength\{[^}]*\}|icml\w+|usepackage(?:\[[^\]]*\])?|documentclass|newcommand|renewcommand|setcounter|multicolumn|cmidrule(?:\([^)]*\))?|begin|end|providecommand)\{[^}]*\}',
         r'\\(?:vskip|hskip|vspace)\s*[-\d.]+\s*(?:in|pt|em)', r'\\(?:tabcolsep|itemsep)\}?\{?[^}]*\}', r'\\[A-Za-z]+\*?', r'\$\\\w+\$']
NUM = re.compile(r'(?<![A-Za-z0-9/\-@._^])(\d+(?:(?:\{,\}|,)\d{3})*(?:\.\d+)?)(?![A-Za-z0-9/])')


def macros():
    """key -> printed text, from paper/numbers.tex."""
    out = {}
    for m in re.finditer(r'\\expandafter\\def\\csname pn@([^\\]+?)\\endcsname\{(.*)\}$', (ROOT / 'paper' / 'numbers.tex').read_text(), re.M):
        out[m.group(1)] = m.group(2)
    return out


def old_provenance():
    """Provenance of the pixel-site study's macro keys, from the committed inventory (hand-written at the port)."""
    prov = {}
    if CSV.exists():
        for r in csv.DictReader(CSV.open()):
            if r['macro_key'] and r['macro_key'] not in prov and not r['macro_key'].startswith('ts.'):
                prov[r['macro_key']] = r
    return prov


def ts_provenance(key):
    for prefix, value, files, cmd, steps, status in TS:
        if key.startswith(prefix):
            return {'value (results/paper_numbers.json path or formula)': value, 'research result file': result_files(files),
                    'research commands': cmd, 'steps here (reproduce_paper.sh)': steps, 'status': status}
    raise KeyError(key)


def rows_of(paper, path, M, prov):
    rows, section, flt = [], 'preamble', ''
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw
        m = re.match(r'\\(?:section|subsection)\*?\{([^}]*)\}', line) or re.match(r'\\paragraph\{([^}]*)\}', line)
        if m:
            section = m.group(1).replace('\\tok{}', 'TokAlign').replace('\\neo{}', 'NEO')
        if '\\begin{abstract}' in line:
            section = 'abstract'
        if re.match(r'\\begin\{(table|figure)', line):
            flt = 'float'
        lm = re.search(r'\\label\{((?:tab|fig|alg):[^}]*)\}', line)
        if lm and flt:
            flt = lm.group(1)
        if re.match(r'\\end\{(table|figure)\}', line):
            flt = ''
        tf = '' if not flt or flt == 'float' else flt
        if '\\caption' in line and tf == '':
            tf = 'caption'
        for key in re.findall(r'\\pn\{([^}]*)\}', line):
            try:
                p = ts_provenance(key)                                        # the table above first, the committed file for the keys it does not cover
            except KeyError:
                p = prov[key]
            rows.append({'paper': paper, 'line': n, 'section': section, 'table_or_figure': tf, 'printed': M[key], 'macro_key': key,
                         **{f: p[f] for f in FIELDS[6:]}})
        text = re.sub(r'\\pn\{[^}]*\}', ' ', line)
        for pat in STRIP:
            text = re.sub(pat, ' ', text)
        if re.match(r'\s*\\?(begin|end|documentclass|usepackage|newcommand|setcounter|renewcommand|input|icml)', raw) or raw.lstrip().startswith('%'):
            continue
        if any(s in raw for s in ('\\begin{tabular}', '\\cmidrule', '\\multicolumn', '\\setlength', '\\includegraphics', '\\vskip', '\\icml', '\\printAffiliations')):
            continue
        for num in NUM.findall(text):
            printed = num
            cat = LITERAL.get(printed.replace('{,}', ''))
            if cat is None and printed in ('3', '6') and any(w in raw for w in COST_WORDS):
                cat = ('cost accounting', 'ETA charged a forward + a backward counted as two forwards (3); SAR two of each (6)', 'setting')
            if cat is None and printed == '1' and ('1\\%' in raw or 'under 1' in raw):
                cat = ('analytic cost', LITERAL['0.75'][1], 'analytic')
            value, src, status = cat if cat else ('setting', SETTING, 'setting')
            rows.append({'paper': paper, 'line': n, 'section': section, 'table_or_figure': tf, 'printed': printed, 'macro_key': '',
                         'value (results/paper_numbers.json path or formula)': value, 'research result file': src, 'research commands': '',
                         'steps here (reproduce_paper.sh)': '', 'status': status})
    return rows


def result_files(files):
    """The scored files behind a row as results/scores/<name>.json paths: names separated by '; ' or ', ' (a comma inside braces
    is a pattern, kept), parenthetical remarks dropped, no path when the row names no file."""
    names = [n.strip() for n in re.split(r';\s*|,\s+', re.sub(r'\s*\([^)]*\)', '', files)) if n.strip()]
    return '; '.join(f'results/scores/{n}.json' for n in names)


def build():
    M, prov = macros(), old_provenance()
    rows = rows_of('report', TEX['report'], M, prov)
    for paper, figs in (('report', FIG_DATA),):
        for tf, printed, value, files, steps, status in figs:
            rows.append({'paper': paper, 'line': 0, 'section': 'figure data (paper/make_figures.py)', 'table_or_figure': tf, 'printed': printed, 'macro_key': '',
                         'value (results/paper_numbers.json path or formula)': value, 'research result file': result_files(files),
                         'research commands': 'paper/make_example_figures.py' if 'framework' in tf else 'paper/make_figures.py', 'steps here (reproduce_paper.sh)': steps, 'status': status})
    return rows


def render(rows):
    buf = io.StringIO(); w = csv.DictWriter(buf, FIELDS, lineterminator='\n'); w.writeheader(); w.writerows(rows)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); args = ap.parse_args()
    text = render(build())
    if args.check:
        same = CSV.exists() and CSV.read_text() == text
        print('identical' if same else 'differs'); return 0 if same else 1
    CSV.write_text(text); n = text.count('\n') - 1
    print(f'{n} rows written to {CSV}')


if __name__ == '__main__':
    sys.exit(main())
