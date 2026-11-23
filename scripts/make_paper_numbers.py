"""Builds results/paper_numbers.json, the single source of every number in the two papers, from the scored run outputs
in results/scores/ (written by the score_* scripts, the analysis scripts and the FlexiViT replay), including the
where-to-align study (two-site design run, depth sweeps, drift by depth, site selection, heldout3 of both backbones). Nothing is computed
here except means of scored values and the selection of the reported arms. paper/make_numbers.py turns the result into
paper/numbers.tex and paper/make_figures.py into the figures.

    python scripts/make_paper_numbers.py            # -> results/paper_numbers.json
    python scripts/make_paper_numbers.py --check    # compare with the existing file instead of writing it
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / 'results' / 'scores'
OUT = ROOT / 'results' / 'paper_numbers.json'
FORWARD = ('neo', 'tok', 'headcov', 'tok+headcov')
WALLCLOCK = ('neo', 'stack_k1', 'stack_k16_fp32', 'stack_k16_fp32_stats32', 'eta')


def load(name):
    return json.loads((SCORES / f'{name}.json').read_text())


def pick(d, keys):
    return {k: d[k] for k in keys}


def build():
    """All numbers, read from results/scores/."""
    fo = {k: load(f'forward_only_{k}') for k in ('design_vitb', 'design_vits', 'design_vitb_src50', 'design_vitb_src300', 'design_vitb_ablations')}
    eta = {k: load(f'eta_{k}') for k in ('design_vitb', 'design_vits')}
    sar = load('sar_design_vitb')
    foa2k, foa6k = load('foa_design_vitb'), load('foa_design_extended_vitb')
    pea3k, pea20k = load('pea_design_vitb'), load('pea_design_vitb_src20k')
    timing, filters = load('timing_design_vitb'), load('patch_filters_design_vitb')
    abl = fo['design_vitb_ablations']['mean_30_cells']
    D = {}
    D['design_vitb'] = {
        'episodic': {**pick(fo['design_vitb']['mean_30_cells'], FORWARD), **eta['design_vitb']['mean_30_cells'], **sar['mean_30_cells']},
        'clean': {**pick(fo['design_vitb']['clean'], FORWARD), **eta['design_vitb']['clean'], **sar['clean']},
        'foa_2k': foa2k['mean_cells'], 'foa_6k': foa6k['mean_cells'], 'foa_refs_2k': foa2k['references_same_cells'],
        'foa_clean_2k': foa2k['clean'], 'foa_clean_6k': foa6k['clean'],
        'pea_3k': pea3k['mean_cells'], 'pea_20k': pea20k['mean_cells'], 'pea_clean_20k': pea20k['clean'],
        'ablation': {'src50': fo['design_vitb_src50']['mean_30_cells']['tok+headcov'], 'src300': fo['design_vitb_src300']['mean_30_cells']['tok+headcov'],
                     'src3000': fo['design_vitb']['mean_30_cells']['tok+headcov'],
                     **pick(abl, ('stack_lam0', 'stack_lam0.3', 'stack_n0x16', 'stack_headn0x4', 'tok_only_nohead'))},
        'wallclock': {**{a: {'ms_per_batch': timing[a]['ms_per_batch'], 'x_neo': timing[a]['x_neo'], 'acc_6cells': timing[a]['acc_cells']} for a in WALLCLOCK},
                      '_run': 'E5c, one run, 2026-10-28, results/spa/bench_summary_e5c.json'},
        'per_family_vs_neo_eta': eta['design_vitb']['per_family_vs_eta_neo'],
        'per_cell_episodic': pick(fo['design_vitb']['per_cell'], FORWARD)}
    D['design_vits'] = {'episodic': {**pick(fo['design_vits']['mean_30_cells'], FORWARD), **eta['design_vits']['mean_30_cells']},
                        'clean': {**pick(fo['design_vits']['clean'], FORWARD), **eta['design_vits']['clean']}}
    # the unadapted model (no adaptation at all) on the design cells, from the patch-filter study's identity filter (raw, without NEO)
    D['source'] = {'raw_design_vitb': filters['mean_30_cells_raw']['identity'], 'raw_design_vitb_clean': filters['per_cell_raw']['identity']['clean'],
                   'neo_design_vitb': fo['design_vitb']['mean_30_cells']['neo']}
    ba = load('block_alignment_design_vitb')
    D['block_alignment'] = {k: ba[k] for k in ('mean_30', 'clean', 'family')}
    D['two_site'] = two_site(eta['design_vitb'])
    D['exit'] = early_exit()
    D['hier'] = hier()
    D['extra'] = extra()
    D['rule'] = rule()
    D['ca'] = models_datasets()
    return D


TS_SITES = ('pixel',) + tuple(f'b{k}' for k in range(1, 13))


def two_site(eta_design):
    """Where to align: the two-site study on the course checkpoint (design sweep and drift, heldout3 for both backbones)."""
    d = load('forward_only_design_vitb_two_site')
    sw, swe = load('forward_only_design_vitb_sweep'), load('eta_design_vitb_sweep')
    h3, e3, s3 = load('forward_only_heldout3_vitb'), load('eta_heldout3_vitb'), load('streams_heldout3_vitb')
    h3s, e3s = load('forward_only_heldout3_vits'), load('eta_heldout3_vits')
    dr, sel = load('drift_by_depth_design_vitb'), load('site_selection_design_vitb')
    cells = [c for c in d['per_cell']['neo'] if c != 'clean']
    # the alignment at one site alone, per cell: the pixel site is TokAlign, block k the arm place_bk (the two-site run holds b6)
    frozen_cell = {s: (d if s in ('pixel', 'b6') else sw)['per_cell']['tok' if s == 'pixel' else f'place_{s}'] for s in TS_SITES}
    eta_cell = {s: eta_design['per_cell']['tok+eta+neo'] if s == 'pixel' else swe['per_cell'][f'eta+place_{s}+neo'] for s in TS_SITES}
    gain = lambda a, base: float(np.mean([a[c] - base[c] for c in cells]))
    st = s3['streams']
    seg = lambda name: {a: r['seg'] for a, r in st[name].items()}
    held = lambda fo, et: {'episodic': {**fo['mean_30_cells'], **et['mean_30_cells']}, 'clean': {**fo['clean'], **et['clean']},
                           'per_cell_episodic': fo['per_cell'], 'per_cell_eta': et['per_cell']}
    T = {'design_vitb': {'episodic': d['mean_30_cells'], 'clean': d['clean'], 'per_cell': d['per_cell']},
         'sweep_vitb': {'frozen_gain': {s: gain(frozen_cell[s], d['per_cell']['neo']) for s in TS_SITES},
                        'eta_gain': {s: gain(eta_cell[s], eta_design['per_cell']['eta+neo']) for s in TS_SITES},
                        'frozen_per_cell': frozen_cell, 'eta_per_cell': eta_cell,
                        'neo_per_cell': d['per_cell']['neo'], 'eta_neo_per_cell': eta_design['per_cell']['eta+neo']},
         'held3_vitb': {**held(h3, e3), 'streams': s3['table'], 'cont5_seg': seg('cont5'), 'cleanret_seg': seg('cleanret'),
                        'gate_cleanret': {a: r['gate_on_per_seg'] for a, r in st['cleanret'].items() if 'gate_on_per_seg' in r}},
         'held3_vits': held(h3s, e3s),
         'drift_vitb': {'k_batches': dr['k'], 'sites': dr['sites'],
                        'z_spec_first_k': {c: {s: dr['cells'][c]['frozen'][s]['spec']['z_first_k'] for s in dr['sites']} for c in dr['cells']},
                        'z_spec_first_k_after_tok': {c: {s: dr['cells'][c]['tok'][s]['spec']['z_first_k'] for s in dr['sites']} for c in dr['cells']},
                        'z_shift_first_k': {c: {s: dr['cells'][c]['frozen'][s]['shift']['z_first_k'] for s in dr['sites']} for c in dr['cells']}},
         'selection_vitb': pick(sel, ('k_batches', 'mean_30_cells', 'gain_over_neo', 'site_counts')),
         # the heldout3 arms repeated on the two other held-out slices (docs/heldout_test_plan_slices.md), each scored once
         'slices_vitb': {**{sl: held(load(f'forward_only_{sl}_vitb_two_site'), load(f'eta_{sl}_vitb_two_site')) for sl in ('heldout', 'heldout2')},
                         '_run': 'run_forward_only.py / run_gradient_host.py --method eta on the heldout and heldout2 splits with the heldout3 arms, '
                                 'tag _two_site, one run per slice scored once, 2026-11-09 (data/derived/course_two_site/runs/*_heldout{,2}_vitb_two_site.json); '
                                 'the neo, tok, headcov, tok+headcov arms of heldout equal the committed pixel-site file exactly, the ETA arms within 0.1 per cell'}}
    return T


def hier():
    """What to align at a site: the two-level (between-image / within-image) map and its one-part variants on the design
    cells (scored against the NEO arm of the two-site design run), and the label-free decomposition of the token statistics
    at every site into the two parts (per cell: mean over the batches of each statistic, and its z against the clean batches)."""
    h, dec = load('forward_only_design_vitb_hier'), load('decompose_by_depth_design_vitb')
    stat = lambda field: {c: {s: {col: dec['cells'][c][s][col][field] for col in dec['columns']} for s in dec['sites']} for c in dec['cells']}
    return {'design_vitb': {'episodic': h['mean_30_cells'], 'clean': h['clean'], 'per_cell': h['per_cell'], 'reference': h['reference']},
            'decomp_vitb': {'sites': dec['sites'], 'columns': dec['columns'], 'clean': dec['clean'], 'mean': stat('mean'), 'z': stat('z')}}


def early_exit():
    """The adaptive site choice per image (docs/heldout_test_plan_exit.md): the early-exit arms on the design split (ImageNet-C,
    ImageNet-C-bar) with the pixel-plus-one-block design runs they were chosen from, the heldout4 test (scored once, both
    benchmarks, one clean cell), and the timing run."""
    def fo(name):
        f = load(name)
        return {'episodic': f['mean_30_cells'], 'clean': f['clean'], 'per_cell': f['per_cell'], 'cells': f['cells'], 'n_cells': f['n_cells'],
                **({'routed': f['routed_fraction'], 'routed_mean': f['routed_fraction_mean']} if 'routed_fraction' in f else {})}
    t = load('timing_design_vitb_exit')
    return {'design': {'inc': fo('forward_only_design_vitb_exit'), 'cbar': fo('forward_only_design_vitb_cbar_exit')},
            'blocks': {'inc': fo('forward_only_design_vitb_select'), 'cbar': fo('forward_only_design_vitb_cbar_select')},
            'held4': {'inc': fo('forward_only_heldout4_vitb_exit'), 'cbar': fo('forward_only_heldout4_vitb_cbar_exit')},
            'timing': {a: {'ms_per_batch': t[a]['ms_per_batch'], 'x_neo': t[a]['x_neo'], 'acc_6cells': t[a]['acc_cells']} for a in t if isinstance(t[a], dict) and 'ms_per_batch' in t[a]},
            'slices': {**{sl: fo(f'forward_only_{sl}_vitb_exit') for sl in ('heldout', 'heldout2', 'heldout3')}, 'cbar_heldout3': fo('forward_only_heldout3_vitb_cbar_exit'),
                       '_run': 'the gated early-exit arms (pair_gated exit_b9_gated exit_b5b9_gated) on the three earlier ImageNet-C held-out slices and on ImageNet-C-bar heldout3, after the heldout4 verdict, for ranges (2026-11-10)'},
            'controls': {'inc': fo('forward_only_heldout4_vitb_exitctl'), 'cbar': fo('forward_only_heldout4_vitb_cbar_exitctl'),
                         '_run': 'run_forward_only.py --split heldout4 --arms exit_b5b9_random_gated exit_b5b9_equal_gated mix_b5b9_gated (equal-compute controls, scored once, 2026-11-10)'},
            'vits': {'design': fo('forward_only_design_vits_exit'), 'held3': fo('forward_only_heldout3_vits_exit'),
                     '_run': 'run_forward_only.py --model vits --arms tok+place_b6 neo pair_gated exit_b9_gated exit_b5b9_gated (design) and pair_gated exit_b9_gated exit_b5b9_gated (heldout3, scored once), sites and threshold from ViT-B/16, 2026-11-10'},
            'streams': {**pick(load('streams_heldout4_vitb_exit'), ('table',)),
                        '_run': 'run_streams.py --split heldout4 --arms neo pair_gated exit_b5b9_gated --streams cont5 cleanret --tag _exit (EMA statistics, gate), scored once, 2026-11-10'},
            'site_choice': {k: pick(load(f'site_choice_design_vitb{sfx}'), ('pair', 'oracle_per_cell', 'oracle_gain', 'per_cell_entropy_choice', 'per_cell_entropy_choice_gain', 'chosen_arm_counts', 'per_image_mixtures', 'fixed_mean', 'controls', 'second_opinions', 'label_free')) for k, sfx in (('inc', ''), ('cbar', '_cbar'))},
            '_run': 'run_forward_only.py --arms exit_b9 exit_b5b9 exit_b5b9_gated pair_gated (design, tag _exit / _cbar_exit) and --arms neo tok place_b6 '
                    'tok+place_b6 tok+place_b6+headcov pair_gated exit_b9_gated exit_b5b9_gated --split heldout4 (scored once), 2026-11-10; the pixel-plus-one-block '
                    'design runs tok+place_b1..b12 (tag _select / _cbar_select, 2026-11-09/10); time_batches.py --tag _exit'}


def rule():
    """The rule-chosen second-opinion stack (docs/heldout_test_plan_rule.md): the held-out runs that exist so far, the design
    analyses of the rule (picks per benchmark and trunk, and how many unlabeled images the rule needs)."""
    def fo(name):
        if not (SCORES / f'{name}.json').exists():
            return None
        f = load(name)
        return {'episodic': f['mean_30_cells'], 'clean': f['clean'], 'per_cell': f['per_cell'], 'cells': f['cells'], 'n_cells': f['n_cells'],
                'routed': f.get('routed_fraction_mean', {}), 'routed_per_cell': f.get('routed_fraction', {})}
    lf = lambda name: (lambda d: {k: d[k] for k in ('trunk_pick_by_entropy', 'trunk_best_by_accuracy', 'second_opinion_pick_shallower', 'second_opinion_pick_deeper',
                                                      'second_opinion_pick_pair', 'second_opinion_pick_pair_gain', 'second_opinion_best', 'second_opinion_best_gain',
                                                      'trunk_mean_entropy', 'trunk_accuracy', 'trunk_spearman_confidence_accuracy', 'trunk_entropy_quantile', 'routed_share_at_theta') if k in d})(load(name)['label_free'])
    ss = lambda name: pick(load(name), ('pair', 'full_rule', 'pooled_first_n_per_cell', 'per_cell_first_k_batches'))
    held = {k: fo(n) for k, n in (('inc', 'forward_only_heldout4_vitb_rule'), ('cbar', 'forward_only_heldout4_vitb_cbar_rule'),
                                 ('inc_abl', 'forward_only_heldout4_vitb_rule_abl'), ('cbar_abl', 'forward_only_heldout4_vitb_cbar_rule_abl'),
                                 ('inc_small', 'forward_only_heldout4_vitb_rule_small'), ('cbar_small', 'forward_only_heldout4_vitb_cbar_rule_small'),
                                 ('inc_ctl', 'forward_only_heldout4_vitb_rule_ctl'), ('cbar_ctl', 'forward_only_heldout4_vitb_cbar_rule_ctl'))}   # other pairs of branch blocks, after the review
    vits = {k: fo(n) for k, n in (('sweep', 'forward_only_design_vits_rule_sweep'), ('held3', 'forward_only_heldout3_vits_rule'))}
    for k, n in (('pick_t6', 'site_choice_design_vits_rule_t6'), ('pick', 'site_choice_design_vits_rule')):
        vits[k] = lf(n) if (SCORES / f'{n}.json').exists() else None
    slices = {k: fo(n) for k, n in (('heldout', 'forward_only_heldout_vitb_rule'), ('heldout2', 'forward_only_heldout2_vitb_rule'), ('heldout3', 'forward_only_heldout3_vitb_rule'),
                                   ('cbar_heldout3', 'forward_only_heldout3_vitb_cbar_rule'), ('design', 'forward_only_design_vitb_rule'), ('cbar_design', 'forward_only_design_vitb_cbar_rule'))}
    random = {k: fo(n) for k, n in (('inc', 'forward_only_heldout4_vitb_rule_random'), ('cbar', 'forward_only_heldout4_vitb_cbar_rule_random'))}
    views = {k: fo(n) for k, n in (('inc', 'forward_only_heldout4_vitb_rule_views'), ('cbar', 'forward_only_heldout4_vitb_cbar_rule_views'))}   # the main path on three views of the same unsure images (after the review)
    streams = pick(load('streams_heldout4_vitb_rule'), ('table',)) if (SCORES / 'streams_heldout4_vitb_rule.json').exists() else None
    streams_eta = pick(load('streams_heldout4_vitb_eta'), ('table',)) if (SCORES / 'streams_heldout4_vitb_eta.json').exists() else None   # ETA and SAR on the same streams, after the review
    # the timing run with every row of the report's first table (NEO, the pixel map alone, the base path, blocks 5/9, the calibrated
    # stack, each second opinion alone, random routing, ETA, SAR) replaces the four-arm run of the calibrated stack when it exists
    timing = next((load(n) for n in ('timing_design_vitb_ablations', 'timing_design_vitb_rule') if (SCORES / f'{n}.json').exists()), None)
    others = load('timing_design_vitb_others') if (SCORES / 'timing_design_vitb_others.json').exists() else None
    # one calibration per shift from its first K design batches (64, 192 or 640 images), held out (addendum 5 for K=3): one scored file
    # per cell, the picks of that cell's first K design batches run on its heldout4 images; NEO per cell from the heldout4 exit file
    fewshot = {}
    for bench, ssname, neoname in (('inc', 'rule_sample_size_design_vitb', 'forward_only_heldout4_vitb_exit'), ('cbar', 'rule_sample_size_design_vitb_cbar', 'forward_only_heldout4_vitb_cbar_exit')):
        first_k = load(ssname)['per_cell_first_k_batches']; neo = load(neoname)['per_cell']['neo']
        for K in ('1', '3', '10'):
            picks = first_k[K]['picks']; rows = {}
            for cell, (t, e, l) in picks.items():
                name = f'forward_only_heldout4_vitb_fewshot_{cell}' if K == '3' else f'forward_only_heldout4_vitb_fewshot_k{K}_{cell}'
                if not (SCORES / f'{name}.json').exists():
                    continue
                f = load(name); T, E, L = (x[4:] for x in (t, e, l))
                rows[cell] = {'picks': f'{T}/{E},{L}', 'neo': neo[cell], 'base': f['per_cell'][f'tok+place_b{T}'][cell], 'stack': f['per_cell'][f'rule_t{T}_b{E}b{L}_gated'][cell],
                              'routed': f.get('routed_fraction', {}).get(f'rule_t{T}_b{E}b{L}_gated', {}).get(cell)}
            if len(rows) == len(picks):
                fewshot.setdefault(bench, {})[f'k{K}'] = {'per_cell': rows, 'n_cells': len(rows), 'images_per_cell': first_k[K]['images_per_cell'],
                                                          'neo': sum(r['neo'] for r in rows.values()) / len(rows), 'base': sum(r['base'] for r in rows.values()) / len(rows),
                                                          'stack': sum(r['stack'] for r in rows.values()) / len(rows)}
    pooled192 = fo('forward_only_heldout4_vitb_cbar_rule_p192')
    return {'picks': {'inc': lf('site_choice_design_vitb'), 'cbar_t6': lf('site_choice_design_vitb_cbar'), 'cbar_t7': lf('site_choice_design_vitb_cbar_t7')},
            'sample_size': {'inc': ss('rule_sample_size_design_vitb'), 'cbar': ss('rule_sample_size_design_vitb_cbar')},
            'fewshot': fewshot, 'pooled192_cbar': pooled192,
            'review': load('review_checks') if (SCORES / 'review_checks.json').exists() else None,   # scripts/analysis/review_checks.py
            'held4': held, 'vits': vits, 'slices': slices, 'random': random, 'views': views, 'streams': streams, 'streams_eta': streams_eta,
            'timing': {a: {'ms_per_batch': t['ms_per_batch'], 'x_neo': t['x_neo'], 'acc_6cells': t['acc_cells']} for a, t in timing.items() if isinstance(t, dict) and 'x_neo' in t} if timing else None,
            'timing_others': {a: {'ms_per_batch': t['ms_per_batch'], 'x_neo': t['x_neo'], 'acc_6cells': t['acc_cells']} for a, t in others.items() if isinstance(t, dict) and 'x_neo' in t} if others else None,
            '_run': 'run_forward_only.py --split heldout4 --arms rule_t6_b4b11_gated (ImageNet-C) / rule_t6_b4b11_gated rule_t7_b5b11_gated tok+place_b7 (ImageNet-C-bar), '
                    'rule_t6_b4_gated rule_t6_b11_gated (ablation), rule_t5_b3b7_gated tok+place_b5 / rule_t7_b5b9_gated (small-sample picks); ViT-S/16 design sweep and heldout3; '
                    'each held-out file scored once (docs/heldout_test_plan_rule.md, 2026-11-10)'}


def models_datasets():
    """The calibrated stack on more models and natural shifts (docs/heldout_test_plan_models_datasets.md): per model-dataset
    pair the design sweep (accuracy and mean entropy of every version), the calibration file, the held-out run and the ETA run."""
    out = {}
    for model in ('vitb', 'vits', 'deitb', 'deits'):
        for ds in ('inc', 'cbar', 'r', 'sketch'):
            split = 'heldout4' if ds in ('inc', 'cbar') else 'heldout'
            tag = f'{model}_cbar_ca' if ds == 'cbar' else f'{model}_ca_{ds}'      # ImageNet-C-bar runs are tagged _cbar_ca (course_cbar2.sh)
            names = {'sweep': f'forward_only_design_{tag}', 'pick_t6': f'site_choice_design_{tag}_t6', 'pick': f'site_choice_design_{tag}',
                     'calib': f'calib_{model}_{ds}', 'held': f'forward_only_{split}_{tag}', 'eta': f'eta_{split}_{tag}'}
            if not (SCORES / f"{names['calib']}.json").exists():
                continue
            pair = {'calib': load(names['calib'])}
            sw = load(names['sweep']); pair['sweep'] = {'episodic': sw['mean_30_cells'], 'clean': sw['clean'], 'n_cells': sw['n_cells']}
            t6 = load(names['pick_t6'])['label_free']
            pair['signal'] = pick(t6, ('trunk_mean_entropy', 'trunk_accuracy', 'trunk_pick_by_entropy', 'trunk_best_by_accuracy', 'trunk_spearman_confidence_accuracy'))
            pk = load(names['pick'])['label_free']
            pair['second_opinions'] = pick(pk, ('second_opinion_pick_shallower', 'second_opinion_pick_deeper', 'second_opinion_pick_pair_gain', 'second_opinion_best', 'second_opinion_best_gain'))
            for k in ('held', 'eta'):
                if (SCORES / f"{names[k]}.json").exists():
                    f = load(names[k]); pair[k] = {'episodic': f['mean_30_cells'], 'clean': f['clean'], 'per_cell': f['per_cell'], 'routed': f.get('routed_fraction_mean', {}), 'n_cells': f['n_cells']}
            out[f'{model}_{ds}'] = pair
    # ETA with the calibrated stack inside (addendum 3) and ETA on ImageNet-C-bar for ViT-B/16 (addendum 4), on the held-out cells
    # of each pair's row in the across-models table (ViT-B/16 and ViT-S/16 on ImageNet-C: heldout3, their ETA comparison's split)
    stack_files = {'vitb_inc': 'eta_heldout3_vitb_stack', 'vits_inc': 'eta_heldout3_vits_stack', 'vitb_cbar': 'eta_heldout4_vitb_cbar_ca_stack'}
    for model in ('vits', 'deitb', 'deits'):
        stack_files[f'{model}_cbar'] = f'eta_heldout4_{model}_cbar_ca_stack'
    for model in ('deitb', 'deits'):
        stack_files[f'{model}_inc'] = f'eta_heldout4_{model}_ca_inc_stack'
    for model in ('vitb', 'vits', 'deitb', 'deits'):
        for ds in ('r', 'sketch'):
            stack_files[f'{model}_{ds}'] = f'eta_heldout_{model}_ca_{ds}_stack'
    out['etastack'] = {}
    for key, name in stack_files.items():
        if (SCORES / f'{name}.json').exists():
            f = load(name); out['etastack'][key] = {'episodic': f['mean_30_cells'], 'clean': f['clean'], 'routed': f.get('routed_fraction_mean', {}), 'n_cells': f['n_cells'], 'file': name}
    if (SCORES / 'eta_heldout4_vitb_cbar_ca.json').exists():
        f = load('eta_heldout4_vitb_cbar_ca'); out['eta_vitb_cbar'] = {'episodic': f['mean_30_cells'], 'clean': f['clean'], 'n_cells': f['n_cells']}
    out['_run'] = 'scripts/cvpr/course_ca.sh of the research repository: compute_source_stats.py --blocks; run_forward_only.py --split design (--dataset r|sketch) --arms neo tok place_b6 tok+place_b1..b12; site_choice.py --trunk 6 then --trunk <pick>; run_forward_only.py --split heldout4|heldout --arms neo tok tok+place_bT rule_tT_bEbL_gated rule_tT_bE_gated rule_tT_bL_gated --exit-theta <0.65 quantile>; run_gradient_host.py --method eta --arms eta tok+eta+place_bT (2026-11-11); ImageNet-C-bar via scripts/cvpr/course_cbar2.sh (tags _cbar_ca, heldout4); the stack inside ETA via course_etastack.sh / course_cbar2.sh etastack_* (--arms tok+eta+rule_tT_bEbL --exit-theta theta, tag _stack)'
    return out


def extra():
    """Supplementary tests of the two-site stack (docs/heldout_test_plan_extra.md): the wall-clock run with the two-site
    arms (six timing cells, one run), the per-cell accuracies of the SAR design run with the two-site arms (its means and clean
    accuracies are under design_vitb with the other hosts), the heldout3 SAR run (scored once), and the ImageNet-C-bar runs (ten
    corruptions disjoint from ImageNet-C, design and heldout3 splits)."""
    t, s, h = load('timing_design_vitb_extra'), load('sar_design_vitb'), load('sar_heldout3_vitb')
    cb = {k: load(f'forward_only_{k}_vitb_cbar') for k in ('design', 'heldout3')}
    cbar = {k: {'episodic': c['mean_30_cells'], 'clean': c['clean'], 'per_cell': c['per_cell'], 'cells': c['cells'], 'n_cells': c['n_cells']}
            for k, c in cb.items()}
    return {'timing': {a: {'ms_per_batch': t[a]['ms_per_batch'], 'x_neo': t[a]['x_neo'], 'acc_6cells': t[a]['acc_cells']} for a in t},
            'sar_design': {'per_cell': s['per_cell'], 'n_cells': s['n_cells']},
            'sar_held3': {'episodic': h['mean_30_cells'], 'clean': h['clean'], 'per_cell': h['per_cell'], 'n_cells': h['n_cells']},
            'cbar': {'design': cbar['design'], 'held3': cbar['heldout3'],
                     '_run': 'run_forward_only.py --arms neo tok place_b6 tok+place_b6 --cells clean <ImageNet-C-bar cells> --tag _cbar on the '
                             'design and heldout3 splits (the latter scored once), 2026-11-09; the key mean_30_cells of the scored files is the '
                             'mean over the 20 ImageNet-C-bar cells (10 corruptions x severities 3 and 5)'},
            '_run': 'time_batches.py --tag _extra, one run, 2026-11-09, data/derived/course_two_site/runs/timing_design_vitb_extra.json',
            '_sar_run': 'run_gradient_host.py --method sar --split design --arms sar tok+sar sar+place_b6 tok+sar+place_b6, one run, 2026-11-09, '
                        'data/derived/course_two_site/runs/sar_design_vitb.json; the six arms of the earlier SAR run came out identical to the earlier scored file',
            '_sar_held3_run': 'run_gradient_host.py --method sar --split heldout3, same arms, one run scored once, 2026-11-09, data/derived/course_two_site/runs/sar_heldout3_vitb.json'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', nargs='?', const=str(OUT), default=None, metavar='JSON',
                    help='compare with this file (default results/paper_numbers.json) instead of writing it')
    args = ap.parse_args()
    D = build()
    if args.check:
        old = json.loads(Path(args.check).read_text())
        diffs = compare(old, D)
        print('\n'.join(diffs) if diffs else 'identical')
        return 1 if any(not d.startswith('added') for d in diffs) else 0
    OUT.write_text(json.dumps(D, indent=1) + '\n')
    print(f'wrote {OUT}')


def compare(old, new, path=''):
    """Exact comparison: every key of ``old`` must exist in ``new`` with the identical value; keys only in ``new`` are
    reported as additions."""
    out = []
    if isinstance(old, dict) and isinstance(new, dict):
        for k in old:
            out += compare(old[k], new[k], f'{path}/{k}') if k in new else [f'missing {path}/{k}']
        out += [f'added {path}/{k}' for k in new if k not in old]
    elif old != new and not (isinstance(old, float) and isinstance(new, float) and np.isnan(old) and np.isnan(new)):
        out.append(f'differs {path}: {old!r} -> {new!r}')
    return out


if __name__ == '__main__':
    sys.exit(main())
