"""Writes paper/numbers.tex: every result number printed in the course report (paper/report/report.tex) and the long
paper (paper/paper.tex) as a macro \\pn{<key>}, formatted from results/paper_numbers.json. Differences, ranges and
counts are computed here from the stored values; the format of each number is the one the papers print. Hyper-
parameters, data sizes and numbers quoted from other papers stay literal in the .tex files (see
results/number_inventory.csv for the source of every number).
Usage: python3 paper/make_numbers.py
"""
import json
import sys
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
J = json.loads((ROOT / 'results' / 'paper_numbers.json').read_text())
OUT = Path(__file__).resolve().parent / 'numbers.tex'
CORR = ('gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow',
        'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression')
N = {}


def v(path):
    x = J
    for k in path.split('/'):
        x = x[k]
    return x


def fmt(x, spec):
    """Python format spec; 'up' + spec rounds half up on the shortest decimal form of x (how a value read off a table
    by hand is rounded)."""
    if spec.startswith('up'):
        q = Decimal(1).scaleb(-int(spec[-2]))
        return str(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))
    s = format(x, spec)
    return re.sub(r'^-(0(\.0+)?)$', r'+\1' if '+' in spec else r'\1', s)   # no negative zero


def put(key, x, spec='.2f'):
    assert key not in N, key
    N[key] = fmt(x, spec)


def r1(x):
    return float(f'{x:.1f}')


VITB_GMACS = 17.56   # multiply-accumulates of one ViT-B/16 forward pass at 224 px, in G (counted once with fvcore; a constant of the architecture)


def build():
    """Fills N with every printed number (key -> text)."""
    N.clear()
    # ---------------- Table: TokAlign on five hosts (design cells)
    d, ds = 'design_vitb', 'design_vits'
    hosts = {  # key: (alone, with TokAlign, clean alone, clean with)
        'neo': (f'{d}/episodic/neo', f'{d}/episodic/tok', f'{d}/clean/neo', f'{d}/clean/tok'),
        'eta': (f'{d}/episodic/eta+neo', f'{d}/episodic/tok+eta+neo', f'{d}/clean/eta+neo', f'{d}/clean/tok+eta+neo'),
        'sar': (f'{d}/episodic/sar+neo', f'{d}/episodic/tok+sar+neo', f'{d}/clean/sar+neo', f'{d}/clean/tok+sar+neo'),
        'foa': (f'{d}/foa_6k/foa', f'{d}/foa_6k/tok+foa', f'{d}/foa_clean_6k/foa', f'{d}/foa_clean_6k/tok+foa'),
        'pea': (f'{d}/pea_20k/pea', f'{d}/pea_20k/tok+pea', f'{d}/pea_clean_20k/pea', f'{d}/pea_clean_20k/tok+pea'),
        'peaviews': (f'{d}/pea_20k/pea_aug', f'{d}/pea_20k/tok+pea_aug', f'{d}/pea_clean_20k/pea_aug', f'{d}/pea_clean_20k/tok+pea_aug'),
        'sneo': (f'{ds}/episodic/neo', f'{ds}/episodic/tok', f'{ds}/clean/neo', f'{ds}/clean/tok'),
        'seta': (f'{ds}/episodic/eta+neo', f'{ds}/episodic/tok+eta+neo', f'{ds}/clean/eta+neo', f'{ds}/clean/tok+eta+neo'),
    }
    gain = {h: v(p[1]) - v(p[0]) for h, p in hosts.items()}
    cdelta = {h: v(p[3]) - v(p[2]) for h, p in hosts.items()}
    for h, p in hosts.items():
        put(f'u.{h}.alone', v(p[0])); put(f'u.{h}.tok', v(p[1])); put(f'u.{h}.gain', gain[h], '+.2f'); put(f'u.{h}.clean', cdelta[h], '+.2f')
    put('u.peaviews.gain.abs', gain['peaviews'])
    put('u.sneo.gain.abs', gain['sneo']); put('u.seta.gain.abs', gain['seta'])
    noviews = ('neo', 'eta', 'sar', 'foa', 'pea')
    put('u.range.lo', min(gain[h] for h in noviews), '.1f'); put('u.range.hi', max(gain[h] for h in noviews), '.1f')
    put('u.cleanmax', max(abs(c) for c in cdelta.values()))

    # ---------------- cost: wall-clock of NEO and ETA (design timing cells) and the MACs of one ViT-B/16 forward pass
    put('cost.gmacs', VITB_GMACS)
    from compute_cost import MAP
    put('cost.mainpath.pct', 100 * 2 * MAP, '.1f')                       # the main path's two maps, in percent of a forward pass
    wc = v('design_vitb/wallclock')
    for k, a in (('neo', 'neo'), ('k1', 'stack_k1'), ('k16', 'stack_k16_fp32'), ('k16s32', 'stack_k16_fp32_stats32'), ('eta', 'eta')):
        put(f'cost.{k}.ms', wc[a]['ms_per_batch'], '.0f'); put(f'cost.{k}.x', wc[a]['x_neo']); put(f'cost.{k}.acc', wc[a]['acc_6cells'])

    # ---------------- ablations
    ab = v('design_vitb/ablation')
    for k in ('src50', 'src300', 'src3000', 'stack_lam0', 'stack_lam0.3', 'stack_n0x16', 'stack_headn0x4', 'tok_only_nohead'):
        put(f'abl.{k}', ab[k])
    src = v('source/raw_design_vitb')
    put('abl.source', src); put('abl.lam0.cost', ab['src3000'] - ab['stack_lam0'])

    # ---------------- baselines (FOA, PEA)
    put('foa.2k', v('design_vitb/foa_2k/foa'), '.1f'); put('foa.2k.neo', v('design_vitb/foa_refs_2k/spa1/neo'), '.1f'); put('foa.6k', v('design_vitb/foa_6k/foa'), '.1f')
    for k, p in (('pea.3k', 'pea_3k/pea'), ('pea.20k', 'pea_20k/pea'), ('pea.3k.views', 'pea_3k/pea_aug'), ('pea.20k.views', 'pea_20k/pea_aug'), ('pea.clean', 'pea_clean_20k/pea')):
        put(k, v(f'design_vitb/{p}'), '.1f')
    put('pea.source', src, '.1f'); put('pea.source.clean', v('source/raw_design_vitb_clean'), '.1f')

    # ---------------- directions that did not work: multi-block mean alignment
    put('blk.all', v('block_alignment/mean_30/mb_all_all')); put('blk.neo', v('block_alignment/mean_30/neo'))
    put('blk.late.gain', v('block_alignment/mean_30/mb_all_late') - v('block_alignment/mean_30/neo'))

    two_site()
    hier()
    robust()
    extra()
    early_exit()
    rule()
    models_datasets()
    across()
    # one-decimal, unsigned copies of three held-out gains that the main text quotes next to one-decimal gains
    for key in ('sar.ts.h.neo.gain.px', 'cb.h.gain.twosite', 'cb.h.twosite.over.best'):
        N[key + '1'] = fmt(float(N[key]), '.1f')
    return N


SITES = ('pixel',) + tuple(f'b{k}' for k in range(1, 13))
GROUPS = {'noise': ('gaussian_noise', 'shot_noise', 'impulse_noise'), 'blur': ('defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'),
          'weather': ('snow', 'frost', 'fog', 'brightness'), 'digital': ('contrast', 'elastic_transform', 'pixelate', 'jpeg_compression')}
CBAR_NAMES = {'blue_noise_sample': 'blue noise', 'brownish_noise': 'brownish noise', 'caustic_refraction': 'caustic refraction',
              'checkerboard_cutout': 'checkerboard cutout', 'cocentric_sine_waves': 'concentric sine waves', 'inverse_sparkles': 'inverse sparkles',
              'perlin_noise': 'Perlin noise', 'plasma_noise': 'plasma noise', 'single_frequency_greyscale': 'single-frequency greyscale',
              'sparkles': 'sparkles'}
NAMES = {'gaussian_noise': 'Gaussian noise', 'shot_noise': 'shot noise', 'impulse_noise': 'impulse noise', 'defocus_blur': 'defocus blur',
         'glass_blur': 'glass blur', 'motion_blur': 'motion blur', 'zoom_blur': 'zoom blur', 'snow': 'snow', 'frost': 'frost', 'fog': 'fog',
         'brightness': 'brightness', 'contrast': 'contrast', 'elastic_transform': 'elastic transform', 'pixelate': 'pixelate',
         'jpeg_compression': 'JPEG'}


def site_name(s):
    return 'the pixels' if s == 'pixel' else f'block {s[1:]}'


def join_names(cos):
    n = [NAMES.get(c, c) for c in cos]
    return n[0] if len(n) == 1 else ', '.join(n[:-1]) + ' and ' + n[-1]


def spearman(x, y):
    rx, ry = (np.argsort(np.argsort(v)) for v in (np.asarray(x), np.asarray(y)))
    return float(np.corrcoef(rx, ry)[0, 1])


def two_site():
    """The where-to-align study: depth sweep, drift by depth, two-site design run and site selection, heldout3 results."""
    T = v('two_site')
    cells = [c for c in T['design_vitb']['per_cell']['neo'] if c != 'clean']
    fam_gain = lambda per_cell, base, co: np.mean([per_cell[f'{co}-{s}'] - base[f'{co}-{s}'] for s in (3, 5)])
    # ---- depth sweep (design cells): mean gain at every site, early / mid blocks, the best site per corruption
    for host, base in (('frozen', 'neo_per_cell'), ('eta', 'eta_neo_per_cell')):
        g = T['sweep_vitb'][f'{host}_gain']
        for s in SITES:
            put(f'ts.sw.{host}.{s}', g[s], '+.1f')
        early, mid = [g[f'b{k}'] for k in (1, 2, 3)], [g[f'b{k}'] for k in (5, 6, 7, 8)]
        put(f'ts.sw.{host}.early.lo', min(early), '.1f'); put(f'ts.sw.{host}.early.hi', max(early), '.1f')
        put(f'ts.sw.{host}.mid.lo', min(mid), '.1f'); put(f'ts.sw.{host}.mid.hi', max(mid), '.1f')
        best = max((s for s in SITES if s != 'pixel'), key=g.get)
        put(f'ts.sw.{host}.best.site', site_name(best), 's'); put(f'ts.sw.{host}.best', g[best], '+.1f')
        put(f'ts.sw.{host}.best.vs.b6', g[best] - g['b6'], '.1f')
    sw = T['sweep_vitb']
    best_site = {co: max(SITES, key=lambda s: fam_gain(sw['frozen_per_cell'][s], sw['neo_per_cell'], co)) for co in CORR}
    pixel_best = [co for co in CORR if best_site[co] == 'pixel']
    mid_best = [co for co in CORR if best_site[co] in ('b4', 'b5', 'b6', 'b7', 'b8')]
    put('ts.sw.pixelbest.n', len(pixel_best), 'd'); put('ts.sw.pixelbest.names', join_names(pixel_best), 's')
    put('ts.sw.midbest.n', len(mid_best), 'd')
    put('ts.sw.pixelbest.nonnoise.names', join_names([co for co in pixel_best if co not in GROUPS['noise']]), 's')
    early_best = [co for co in CORR if best_site[co] in ('b1', 'b2', 'b3')]
    put('ts.sw.earlybest.n', len(early_best), 'd')
    b6_neg = [co for co in CORR if fam_gain(sw['frozen_per_cell']['b6'], sw['neo_per_cell'], co) < 0]
    put('ts.sw.b6neg.n', len(b6_neg), 'd'); put('ts.sw.b6neg.names', join_names(b6_neg) if b6_neg else 'none', 's')
    # ---- drift by depth (design cells): z of the spectral drift over the first k batches, pixel against block 6
    dr = T['drift_vitb']; put('ts.dr.k', dr['k_batches'], 'd')
    zco = lambda co, s: np.mean([dr['z_spec_first_k'][f'{co}-{sev}'][s] for sev in (3, 5)])
    for grp, cos in (('pixelbest', pixel_best), ('midbest', mid_best)):
        for s in ('pixel', 'b6'):
            put(f'ts.dr.{grp}.{s}', np.mean([zco(co, s) for co in cos]) if cos else float('nan'), '.0f')
    dz = [zco(co, 'b6') - zco(co, 'pixel') for co in CORR]
    dg = [fam_gain(sw['frozen_per_cell']['b6'], sw['neo_per_cell'], co) - fam_gain(sw['frozen_per_cell']['pixel'], sw['neo_per_cell'], co) for co in CORR]
    put('ts.dr.spearman', spearman(dz, dg), '+.2f')
    put('ts.dr.pixelmax.n', sum(zco(co, 'pixel') > zco(co, 'b6') for co in CORR), 'd')
    put('ts.dr.b6max.names', join_names([co for co in CORR if zco(co, 'b6') > zco(co, 'pixel')]) or 'none', 's')
    put('ts.dr.agree.n', sum((a > 0) == (b > 0) for a, b in zip(dz, dg)), 'd')
    zat = lambda co, s: np.mean([dr['z_spec_first_k_after_tok'][f'{co}-{sev}'][s] for sev in (3, 5)])
    put('ts.dr.aftertok.b6.min', min(zat(co, 'b6') for co in CORR), '.0f')
    for co in CORR:
        put(f'ts.dr.{co}.pixel', zco(co, 'pixel'), '.0f'); put(f'ts.dr.{co}.b6', zco(co, 'b6'), '.0f')
        put(f'ts.dr.{co}.best', site_name(best_site[co]).replace('the pixels', 'pixels'), 's')
    # ---- two-site design run and site selection (design cells)
    d, sel = T['design_vitb'], T['selection_vitb']
    DA = {'neo': 'neo', 'tok': 'tok', 'hc': 'headcov', 'stack': 'tok+headcov', 'b6': 'place_b6', 'twosite': 'tok+place_b6', 'stack2': 'tok+place_b6+headcov'}
    for k, a in DA.items():
        put(f'ts.d.{k}', d['episodic'][a]); put(f'ts.d.clean.{k}', d['clean'][a] - d['clean']['neo'], '+.2f')
        if k != 'neo':
            put(f'ts.d.gain.{k}', d['episodic'][a] - d['episodic']['neo'], '+.2f')
    put('ts.d.twosite.x.tok', (d['episodic']['tok+place_b6'] - d['episodic']['neo']) / (d['episodic']['tok'] - d['episodic']['neo']), '.1f')
    put('ts.d.twosite.over.tok', d['episodic']['tok+place_b6'] - d['episodic']['tok'], '+.2f')
    for k, r in (('oracle2', 'oracle_pixel_or_b6'), ('oracleany', 'oracle_any_site'), ('drift2', 'select_pixel_or_b6_by_drift'), ('driftany', 'select_site_of_largest_drift')):
        put(f'ts.sel.{k}', sel['mean_30_cells'][r]); put(f'ts.sel.{k}.gain', sel['gain_over_neo'][r], '+.2f')
    put('ts.sel.twosite.over.oracle2', sel['mean_30_cells']['two_site'] - sel['mean_30_cells']['oracle_pixel_or_b6'], '+.2f')
    put('ts.sel.drift2.pixel.n', sel['site_counts']['select_pixel_or_b6_by_drift'].get('pixel', 0), 'd')
    put('ts.sel.drift2.b6.n', sel['site_counts']['select_pixel_or_b6_by_drift'].get('b6', 0), 'd')
    put('ts.sel.driftany.sites.n', len(sel['site_counts']['select_site_of_largest_drift']), 'd')
    # ---- heldout3, both backbones
    HA = {**DA, 'eta': 'eta', 'etaneo': 'eta+neo', 'etaneotok': 'tok+eta+neo', 'etaneob6': 'eta+place_b6+neo', 'etaneotwosite': 'tok+eta+place_b6+neo'}
    CO = {'neo': 'neo', 'stack': 'stack_gated', 'stack2': 'twosite_gated', 'etaneo': 'eta+neo', 'etaneotok': 'tok+eta_gated+neo', 'etaneotwosite': 'twosite+eta_gated+neo'}
    for b, blk in (('hb', 'held3_vitb'), ('hs', 'held3_vits')):
        H = T[blk]; ep, cl = H['episodic'], H['clean']
        for k, a in HA.items():
            put(f'ts.{b}.ep.{k}', ep[a]); put(f'ts.{b}.cl.{k}', cl[a])
        for k in ('tok', 'b6', 'twosite', 'stack', 'stack2'):
            put(f'ts.{b}.gain.{k}', ep[HA[k]] - ep['neo'], '.1f')
        put(f'ts.{b}.gain.stack2.over.stack', ep['tok+place_b6+headcov'] - ep['tok+headcov'], '.1f')
        put(f'ts.{b}.gain.twosite.over.tok', ep['tok+place_b6'] - ep['tok'], '.1f')
        put(f'ts.{b}.twosite.x.tok', (ep['tok+place_b6'] - ep['neo']) / (ep['tok'] - ep['neo']), '.1f')
        put(f'ts.{b}.gain.etab6', ep['eta+place_b6+neo'] - ep['eta+neo'], '.1f')
        put(f'ts.{b}.gain.etatok', ep['tok+eta+neo'] - ep['eta+neo'], '.1f')
        put(f'ts.{b}.gain.etatwosite', ep['tok+eta+place_b6+neo'] - ep['tok+eta+neo'], '.1f')
        put(f'ts.{b}.gain.etatwosite.over.etaneo', ep['tok+eta+place_b6+neo'] - ep['eta+neo'], '.1f')
        put(f'ts.{b}.stack2.vs.etaneo', ep['tok+place_b6+headcov'] - ep['eta+neo'], '+.1f')
        put(f'ts.{b}.clean.stack2', cl['tok+place_b6+headcov'] - cl['neo'], '+.2f'); put(f'ts.{b}.clean.twosite', cl['tok+place_b6'] - cl['neo'], '+.2f')
        put(f'ts.{b}.clean.b6', cl['place_b6'] - cl['neo'], '+.2f')
        pc = H['per_cell_episodic']; hc = [c for c in pc['neo'] if c != 'clean']
        put(f'ts.{b}.oracle2', float(np.mean([max(pc['tok'][c], pc['place_b6'][c]) for c in hc])))
        put(f'ts.{b}.twosite.over.oracle2', ep['tok+place_b6'] - float(np.mean([max(pc['tok'][c], pc['place_b6'][c]) for c in hc])), '+.2f')
    H = T['held3_vitb']; co = H['streams']
    for k, a in CO.items():
        put(f'ts.hb.co.{k}', co[a]['cont']); put(f'ts.hb.cr.{k}', co[a]['cleanret_clean'])
    put('ts.hb.co.gain.stack2', co['twosite_gated']['cont'] - co['neo']['cont'], '.1f')
    put('ts.hb.co.gain.stack', co['stack_gated']['cont'] - co['neo']['cont'], '.1f')
    put('ts.hb.co.gain.stack2.over.stack', co['twosite_gated']['cont'] - co['stack_gated']['cont'], '.1f')
    put('ts.hb.co.gain.etatwosite', co['twosite+eta_gated+neo']['cont'] - co['tok+eta_gated+neo']['cont'], '.1f')
    put('ts.hb.co.gain.etatok', co['tok+eta_gated+neo']['cont'] - co['eta+neo']['cont'], '.1f')
    put('ts.hb.co.stack2.vs.etaneo', co['twosite_gated']['cont'] - co['eta+neo']['cont'], '+.1f')
    put('ts.hb.cr.stack2.vs.neo', co['twosite_gated']['cleanret_clean'] - co['neo']['cleanret_clean'], '+.2f')
    gate = H['gate_cleanret']['twosite_gated']
    put('ts.hb.gate.clean.batches', round(max(g for k, g in gate.items() if k.startswith('clean@')) * 32), 'd')
    put('ts.hb.gate.corr.min', min(g for k, g in gate.items() if not k.startswith('clean@')), '.2f')
    seg = H['cont5_seg']
    put('ts.hb.c5.stack2.wins', sum(seg['twosite_gated'][k] > seg['stack_gated'][k] for k in seg['neo']), 'd')
    put('ts.hb.c5.etatwosite.wins', sum(seg['twosite+eta_gated+neo'][k] > seg['tok+eta_gated+neo'][k] for k in seg['neo']), 'd')
    # held-out against design (the frozen arms)
    put('ts.held.vs.design.max', max(abs(H['episodic'][a] - d['episodic'][a]) for a in DA.values()), '.1f')
    # ---- the same arms on the two other held-out slices: per-slice values and the mean and range over the three slices
    SL = {'h1': T['slices_vitb']['heldout'], 'h2': T['slices_vitb']['heldout2']}
    for sl, Hs in SL.items():
        for k, a in HA.items():
            put(f'ts.sl.{sl}.ep.{k}', Hs['episodic'][a]); put(f'ts.sl.{sl}.cl.{k}', Hs['clean'][a])
    three = [T['held3_vitb'], SL['h1'], SL['h2']]
    GAINS = {'twosite': ('tok+place_b6', 'neo'), 'tok': ('tok', 'neo'), 'b6': ('place_b6', 'neo'), 'stack2': ('tok+place_b6+headcov', 'neo'),
             'stack2.over.stack': ('tok+place_b6+headcov', 'tok+headcov'), 'twosite.over.tok': ('tok+place_b6', 'tok'),
             'etatwosite': ('tok+eta+place_b6+neo', 'tok+eta+neo'), 'etatok': ('tok+eta+neo', 'eta+neo'), 'etab6': ('eta+place_b6+neo', 'eta+neo'),
             'etatwosite.over.etaneo': ('tok+eta+place_b6+neo', 'eta+neo')}
    for g, (a, b) in GAINS.items():
        vals = [Hs['episodic'][a] - Hs['episodic'][b] for Hs in three]
        put(f'ts.sl.gain.{g}.mean', np.mean(vals), '+.1f'); put(f'ts.sl.gain.{g}.lo', min(vals), '+.1f'); put(f'ts.sl.gain.{g}.hi', max(vals), '+.1f')
        put(f'ts.sl.gain.{g}.min', min(vals), '.1f')
    for k, a in (('twosite', 'tok+place_b6'), ('stack2', 'tok+place_b6+headcov'), ('etatwosite', 'tok+eta+place_b6+neo')):
        base = 'eta+neo' if k.startswith('eta') else 'neo'
        vals = [Hs['clean'][a] - Hs['clean'][base] for Hs in three]
        put(f'ts.sl.clean.{k}.lo', min(vals), '+.2f'); put(f'ts.sl.clean.{k}.hi', max(vals), '+.2f')
    put('ts.sl.n', len(three), 'd')
    put('ts.sl.ep.twosite.lo', min(Hs['episodic']['tok+place_b6'] for Hs in three)); put('ts.sl.ep.twosite.hi', max(Hs['episodic']['tok+place_b6'] for Hs in three))
    # ---- per corruption on heldout3 (ViT-B/16, mean of severities 3 and 5)
    FC = {'neo': ('per_cell_episodic', 'neo'), 'tok': ('per_cell_episodic', 'tok'), 'b6': ('per_cell_episodic', 'place_b6'),
          'twosite': ('per_cell_episodic', 'tok+place_b6'), 'etaneo': ('per_cell_eta', 'eta+neo'), 'etaneotwosite': ('per_cell_eta', 'tok+eta+place_b6+neo')}
    f3 = lambda blk, a, co: np.mean([H[blk][a][f'{co}-{s}'] for s in (3, 5)])
    for c in CORR:
        for k, (blk, a) in FC.items():
            put(f'ts.fam.{c}.{k}', f3(blk, a, c), '.1f')
    for k, (blk, a) in FC.items():
        put(f'ts.fam.mean.{k}', H['episodic'][a])
    gains = {k: {c: f3('per_cell_episodic', HA[k], c) - f3('per_cell_episodic', 'neo', c) for c in CORR} for k in ('tok', 'b6', 'twosite')}
    for k in gains:
        for c in CORR:
            put(f'ts.gainfam.{c}.{k}', gains[k][c], '+.1f')
    for grp, cos in GROUPS.items():
        for k in gains:
            put(f'ts.grp.{grp}.{k}', np.mean([gains[k][c] for c in cos]), '+.1f')
    put('ts.fam.twosite.wins', sum(gains['twosite'][c] >= max(gains['tok'][c], gains['b6'][c]) for c in CORR), 'd')
    hb6neg = [c for c in CORR if gains['b6'][c] < 0]
    put('ts.fam.b6neg.n', len(hb6neg), 'd'); put('ts.fam.b6neg.names', join_names(hb6neg) if hb6neg else 'none', 's')
    htokneg = [c for c in CORR if gains['tok'][c] < 0]
    put('ts.fam.tokneg.n', len(htokneg), 'd'); put('ts.fam.tokneg.names', join_names(htokneg) if htokneg else 'none', 's')
    best_h = max(CORR, key=lambda c: gains['twosite'][c]); worst_h = min(CORR, key=lambda c: gains['twosite'][c])
    put('ts.fam.twosite.best.name', NAMES[best_h], 's'); put('ts.fam.twosite.best', gains['twosite'][best_h], '+.1f')
    put('ts.fam.twosite.worst.name', NAMES[worst_h], 's'); put('ts.fam.twosite.worst', gains['twosite'][worst_h], '+.1f')


def hier():
    """What to align at a site: the two-level map and its one-part variants on the design cells, against the pooled arms of
    the two-site design run, and the decomposition of the token statistics at the pixels and at block 6."""
    H, T = v('hier'), v('two_site')
    ep, cl, pc = H['design_vitb']['episodic'], H['design_vitb']['clean'], H['design_vitb']['per_cell']
    pep, pcl, ppc = T['design_vitb']['episodic'], T['design_vitb']['clean'], T['design_vitb']['per_cell']
    HA = {'hierb6': 'hier_b6', 'withinb6': 'within_b6', 'betweenb6': 'between_b6', 'hierpx': 'hier_px', 'hierpair': 'hier_px+hier_b6',
          'tokhierb6': 'tok+hier_b6', 'n0x4': 'hier_b6_n0x4', 'n0d4': 'hier_b6_n0d4'}
    PA = {'neo': 'neo', 'tok': 'tok', 'b6': 'place_b6', 'twosite': 'tok+place_b6'}
    cell = {**{k: pc[a] for k, a in HA.items()}, **{k: ppc[a] for k, a in PA.items()}}
    mean = {**{k: ep[a] for k, a in HA.items()}, **{k: pep[a] for k, a in PA.items()}}
    clean = {**{k: cl[a] for k, a in HA.items()}, **{k: pcl[a] for k, a in PA.items()}}
    for k in HA:
        put(f'hi.d.{k}', mean[k]); put(f'hi.d.gain.{k}', mean[k] - mean['neo'], '+.2f'); put(f'hi.d.clean.{k}', clean[k] - clean['neo'], '+.2f')
    for k, base in (('hierb6', 'b6'), ('withinb6', 'b6'), ('betweenb6', 'b6'), ('hierpx', 'tok'), ('hierpair', 'twosite'), ('tokhierb6', 'twosite'),
                    ('n0x4', 'hierb6'), ('n0d4', 'hierb6'), ('hierb6', 'withinb6')):
        put(f'hi.d.{k}.vs.{base}', mean[k] - mean[base], '+.2f')
    put('hi.d.hierb6.vs.b6.abs', abs(mean['hierb6'] - mean['b6']), '.2f')                      # the size of the gap, unsigned
    # per corruption (mean of severities 3 and 5) and per family: gains over NEO, and the two-level arm against its pooled arm
    g = lambda k, co: np.mean([cell[k][f'{co}-{s}'] - cell['neo'][f'{co}-{s}'] for s in (3, 5)])
    for k in ('hierb6', 'withinb6', 'betweenb6', 'hierpx', 'hierpair', 'b6', 'tok', 'twosite'):
        for grp, cos in GROUPS.items():
            put(f'hi.grp.{grp}.{k}', np.mean([g(k, co) for co in cos]), '+.1f')
    for k, base in (('hierb6', 'b6'), ('hierpx', 'tok'), ('hierpair', 'twosite')):
        d = {co: g(k, co) - g(base, co) for co in CORR}
        for co in CORR:
            put(f'hi.corr.{co}.{k}', d[co], '+.1f')
        best, worst = max(CORR, key=d.get), min(CORR, key=d.get)
        put(f'hi.corr.{k}.best.name', NAMES[best], 's'); put(f'hi.corr.{k}.best', d[best], '+.1f')
        put(f'hi.corr.{k}.worst.name', NAMES[worst], 's'); put(f'hi.corr.{k}.worst', d[worst], '+.1f')
        put(f'hi.corr.{k}.wins', sum(x > 0 for x in d.values()), 'd')
    # the decomposition: log ratio of the within-image and between-image trace to the source, mean of severities 3 and 5
    M = H['decomp_vitb']['mean']
    lr = lambda site, part, co: np.mean([M[f'{co}-{s}'][site][f'{part}_logtrace'] for s in (3, 5)])
    for site, tag in (('pixel', 'px'), ('b6', 'b6')):
        for part in ('within', 'between'):
            vals = {co: lr(site, part, co) for co in CORR}
            for co in CORR:
                put(f'hi.dec.{tag}.{part}.{co}', vals[co], '+.2f')
            for grp, cos in GROUPS.items():
                put(f'hi.dec.{tag}.{part}.{grp}', np.mean([vals[co] for co in cos]), '+.2f')
            put(f'hi.dec.{tag}.{part}.min', min(vals.values()), '+.2f'); put(f'hi.dec.{tag}.{part}.max', max(vals.values()), '+.2f')
            put(f'hi.dec.{tag}.{part}.shrink.lo', 100 * (1 - np.exp(max(vals.values()))), '.0f')     # per cent of the source trace lost
            put(f'hi.dec.{tag}.{part}.shrink.hi', 100 * (1 - np.exp(min(vals.values()))), '.0f')
            put(f'hi.dec.{tag}.{part}.neg.n', sum(x < 0 for x in vals.values()), 'd')
        sh = {co: np.mean([M[f'{co}-{s}'][site]['shift'] for s in (3, 5)]) for co in CORR}
        for co in CORR:
            put(f'hi.dec.{tag}.shift.{co}', sh[co], '.2f')
    put('hi.dec.px.within.noise.x', np.exp(lr('pixel', 'within', 'gaussian_noise')), '.2f')
    put('hi.dec.px.within.contrast.x', 1 / np.exp(np.mean([lr('pixel', 'within', co) for co in ('contrast',)])), '.0f')
    moved = [co for co in CORR if np.mean([M[f'{co}-{s}']['pixel']['shift'] for s in (3, 5)]) > 0.5]
    put('hi.dec.px.shift.names', join_names(moved) if moved else 'none', 's'); put('hi.dec.px.shift.n', len(moved), 'd')


def robust():
    """Is the pair tuned to the corruptions it was chosen on, and would one gate per site help: leave-one-family-out and
    leave-one-corruption-out choice of the block site from the design sweep, the per-cell oracles of a gate per site, and a
    label-free rule that switches block 6 off when the pixel within-image part inflates (all design cells, ViT-B/16 with NEO)."""
    T, H = v('two_site'), v('hier')
    sw, pc, dec = T['sweep_vitb'], T['design_vitb']['per_cell'], H['decomp_vitb']['mean']
    cells = [c for c in pc['neo'] if c != 'clean']
    blocks = [s for s in SITES if s != 'pixel']
    gain = lambda s, cs: np.mean([sw['frozen_per_cell'][s][c] - sw['neo_per_cell'][c] for c in cs])
    of = lambda cos: [c for c in cells if c.rsplit('-', 1)[0] in cos]
    # ---- leave one family out: choose the block on the other three families, then look at the family left out
    chosen = {}
    for grp, cos in GROUPS.items():
        train, test = [c for c in cells if c not in of(cos)], of(cos)
        chosen[grp] = max(blocks, key=lambda s: gain(s, train))
        best = max(blocks, key=lambda s: gain(s, test))
        put(f'rb.loo.{grp}.site', site_name(chosen[grp]), 's'); put(f'rb.loo.{grp}.b6', gain('b6', test), '+.1f')
        put(f'rb.loo.{grp}.best.site', site_name(best), 's'); put(f'rb.loo.{grp}.best', gain(best, test), '+.1f')
        put(f'rb.loo.{grp}.best.vs.b6', gain(best, test) - gain('b6', test), '.1f')
    put('rb.loo.fam.b6.n', sum(s == 'b6' for s in chosen.values()), 'd'); put('rb.loo.fam.n', len(GROUPS), 'd')
    # ---- leave one corruption out
    loo = {co: max(blocks, key=lambda s: gain(s, [c for c in cells if c.rsplit('-', 1)[0] != co])) for co in CORR}
    put('rb.loo.corr.b6.n', sum(s == 'b6' for s in loo.values()), 'd'); put('rb.loo.corr.n', len(CORR), 'd')
    # ---- a gate per site: per-cell oracles against the fixed pair, and the cells where block 6 hurts with the pixel map in front
    pair = np.mean([pc['tok+place_b6'][c] for c in cells])
    orc = lambda arms: np.mean([max(pc[a][c] for a in arms) for c in cells]) - pair
    put('rb.gate.oracle.b6', orc(('tok', 'tok+place_b6')), '+.2f'); put('rb.gate.oracle.px', orc(('place_b6', 'tok+place_b6')), '+.2f')
    put('rb.gate.oracle.both', orc(('neo', 'tok', 'place_b6', 'tok+place_b6')), '+.2f')
    hurt = sorted((c for c in cells if pc['tok+place_b6'][c] < pc['tok'][c]), key=lambda c: pc['tok+place_b6'][c] - pc['tok'][c])
    cell_name = lambda c: f"{NAMES[c.rsplit('-', 1)[0]]} {c.rsplit('-', 1)[1]}"
    put('rb.gate.b6hurts.n', len(hurt), 'd'); put('rb.gate.b6hurts.names', join_names([cell_name(c) for c in hurt]), 's')
    put('rb.gate.b6hurts.worst', pc['tok+place_b6'][hurt[0]] - pc['tok'][hurt[0]], '+.2f'); put('rb.gate.b6hurts.worst.name', cell_name(hurt[0]), 's')
    # ---- a label-free rule: block 6 off when the pixel within-image part is inflated (noise), threshold chosen on the design cells
    within = lambda c: dec[c]['pixel']['within_logtrace']
    rule = lambda t: np.mean([pc['tok'][c] if within(c) > t else pc['tok+place_b6'][c] for c in cells]) - pair
    thr = max((0.0, 0.03, 0.05, 0.08, 0.10, 0.15), key=rule)
    put('rb.gate.rule.thr', thr, '.2f'); put('rb.gate.rule.gain', rule(thr), '+.2f')
    put('rb.gate.rule.off.n', sum(within(c) > thr for c in cells), 'd')


def extra():
    """The wall-clock run with the two-site stack: ms per batch, ratio to NEO and six-cell top-1 of every timed arm."""
    wc = v('extra/timing')
    for k, a in (('neo', 'neo'), ('stack', 'stack_k16_fp32_stats32'), ('twosite16', 'twosite_k16_fp32_stats32'), ('twosite1', 'twosite_k1'),
                 ('eta', 'eta'), ('etatwosite', 'eta+twosite')):
        put(f'cost.ex.{k}.ms', wc[a]['ms_per_batch'], '.0f'); put(f'cost.ex.{k}.x', wc[a]['x_neo']); put(f'cost.ex.{k}.acc', wc[a]['acc_6cells'])
    # the two sites inside SAR (design cells, then heldout3): without and with NEO at the head, as the SAR rows of the host table
    H = v('extra/sar_held3')
    for d, pc, pre in ((v('design_vitb'), v('extra/sar_design/per_cell'), 'sar.ts.'), (H, H['per_cell'], 'sar.ts.h.')):
        for k, suf in ((pre, ''), (pre + 'neo.', '+neo')):
            al, px, b6, two = f'sar{suf}', f'tok+sar{suf}', f'sar+place_b6{suf}', f'tok+sar+place_b6{suf}'
            for n, a in (('alone', al), ('px', px), ('b6', b6), ('twosite', two)):
                put(k + n, d['episodic'][a]); put(k + 'clean.' + n, d['clean'][a] - d['clean'][al], '+.2f')
            put(k + 'gain.px', d['episodic'][two] - d['episodic'][px], '+.2f'); put(k + 'gain.alone', d['episodic'][two] - d['episodic'][al], '+.2f')
            put(k + 'gain.b6', d['episodic'][b6] - d['episodic'][al], '+.2f'); put(k + 'gain.tok', d['episodic'][px] - d['episodic'][al], '+.2f')
            for grp, cos in GROUPS.items():
                put(k + f'fam.{grp}', np.mean([pc[two][f'{co}-{s}'] - pc[px][f'{co}-{s}'] for co in cos for s in (3, 5)]), '+.1f')

    # ImageNet-C-bar (ten corruptions never seen when the sites were chosen): design and heldout3, four forward-only arms
    CB = v('extra/cbar')
    corrs = sorted({c.rsplit('-', 1)[0] for c in CB['design']['cells'] if c != 'clean'}, key=CBAR_NAMES.__getitem__)
    put('cb.n.corr', len(corrs), 'd'); put('cb.n.cells', CB['design']['n_cells'], 'd')
    A = {'neo': 'neo', 'tok': 'tok', 'b6': 'place_b6', 'twosite': 'tok+place_b6'}
    for s, S in (('d', 'design'), ('h', 'held3')):
        ep, cl, pc = CB[S]['episodic'], CB[S]['clean'], CB[S]['per_cell']
        for k, a in A.items():
            put(f'cb.{s}.{k}', ep[a]); put(f'cb.{s}.clean.{k}', cl[a] - cl['neo'], '+.2f')
            if k != 'neo':
                put(f'cb.{s}.gain.{k}', ep[a] - ep['neo'], '+.2f')
        put(f'cb.{s}.twosite.over.best', ep['tok+place_b6'] - max(ep['tok'], ep['place_b6']), '+.2f')
        m = {co: {k: np.mean([pc[a][f'{co}-{sev}'] for sev in (3, 5)]) for k, a in A.items()} for co in corrs}
        for co in corrs:
            for k in A:
                put(f'cb.{s}.{co}.{k}', m[co][k])
        names = lambda cos: join_names([CBAR_NAMES[c] for c in cos]) if cos else 'none'
        loses = [co for co in corrs if m[co]['twosite'] < max(m[co]['tok'], m[co]['b6'])]
        neg = [co for co in corrs if m[co]['twosite'] < m[co]['neo']]
        best = lambda co: max(('tok', 'b6', 'twosite'), key=lambda k: m[co][k])
        put(f'cb.{s}.loses.n', len(loses), 'd'); put(f'cb.{s}.neg.n', len(neg), 'd'); put(f'cb.{s}.neg.names', names(neg), 's')
        put(f'cb.{s}.pxonly.names', names([co for co in corrs if best(co) == 'tok']), 's')
        put(f'cb.{s}.b6only.names', names([co for co in corrs if best(co) == 'b6']), 's')
        put(f'cb.{s}.pairbest.n', sum(best(co) == 'twosite' for co in corrs), 'd')


def early_exit():
    """Adaptive site choice per image: the early-exit arms on the design split and on heldout4 (both benchmarks), the pixel-plus-
    one-block design arms, and the timing run. Prefix ex."""
    E = v('exit'); TS = v('two_site'); CB = v('extra/cbar')
    A = {'neo': 'neo', 'tok': 'tok', 'b6': 'place_b6', 'pair': 'tok+place_b6', 'stack2': 'tok+place_b6+headcov', 'pairg': 'pair_gated',
         'b9': 'exit_b9', 'b5b9': 'exit_b5b9', 'b9g': 'exit_b9_gated', 'b5b9g': 'exit_b5b9_gated'}
    # design: the early-exit arms against the pair and NEO of the design runs of the same benchmark
    for bench, ref in (('inc', TS['design_vitb']), ('cb', CB['design'])):
        d = E['design']['inc' if bench == 'inc' else 'cbar']; pair, neo = ref['episodic']['tok+place_b6'], ref['episodic']['neo']
        put(f'ex.d.{bench}.pair', pair); put(f'ex.d.{bench}.neo', neo)
        for k in ('b9', 'b5b9', 'b5b9g', 'pairg'):
            a = A[k]; put(f'ex.d.{bench}.{k}', d['episodic'][a]); put(f'ex.d.{bench}.gain.{k}', d['episodic'][a] - pair, '+.2f')
            put(f'ex.d.{bench}.clean.{k}', d['clean'][a] - ref['clean']['neo'], '+.2f')
            if a in d.get('routed_mean', {}):
                put(f'ex.d.{bench}.routed.{k}', 100 * d['routed_mean'][a], '.0f')
        # the pixel site plus one block, every block (design): the best block and its mean, the per-cell oracle over the 13 arms
        blk = E['blocks']['inc' if bench == 'inc' else 'cbar']; cells = blk['cells']
        arms = {('px' if kk == 0 else f'b{kk}'): (ref['per_cell']['tok'] if kk == 0 else (ref['per_cell']['tok+place_b6'] if kk == 6 else blk['per_cell'][f'tok+place_b{kk}'])) for kk in range(13)}
        means = {n: float(np.mean([pc[c] for c in cells])) for n, pc in arms.items()}
        best = max(means, key=means.get); put(f'ex.d.{bench}.blocks.best', best.replace('b', 'block ').replace('px', 'the pixels alone'), 's'); put(f'ex.d.{bench}.blocks.best.mean', means[best])
        put(f'ex.d.{bench}.blocks.oracle', float(np.mean([max(pc[c] for pc in arms.values()) for c in cells])))
        put(f'ex.d.{bench}.blocks.oracle.gain', float(np.mean([max(pc[c] for pc in arms.values()) for c in cells])) - pair, '+.2f')
        for n, mval in means.items():
            put(f'ex.d.{bench}.blocks.{n}', mval)
    # heldout4, scored once: every arm, gains against the pair and NEO, one clean cell
    for bench in ('inc', 'cb'):
        h = E['held4']['inc' if bench == 'inc' else 'cbar']; ep, cl = h['episodic'], h['clean']
        for k, a in A.items():
            if a not in ep:
                continue
            put(f'ex.h.{bench}.{k}', ep[a]); put(f'ex.h.{bench}.gain.{k}', ep[a] - ep['tok+place_b6'], '+.2f'); put(f'ex.h.{bench}.gainneo.{k}', ep[a] - ep['neo'], '+.2f')
            put(f'ex.h.{bench}.gaing.{k}', ep[a] - ep['pair_gated'], '+.2f')                       # against the gated main path, the reference of the calibrated cascade's table
            if a in h.get('routed_mean', {}):
                put(f'ex.h.{bench}.routed.{k}', 100 * h['routed_mean'][a], '.0f')
        put(f'ex.h.{bench}.n', h['n_cells'], 'd')
    h = E['held4']['inc']
    for k, a in A.items():
        if a in h['clean']:
            put(f'ex.h.clean.{k}', h['clean'][a]); put(f'ex.h.clean.vsneo.{k}', h['clean'][a] - h['clean']['neo'], '+.2f')
    # per corruption on heldout4 (ImageNet-C, mean of severities 3 and 5): the method against the pair, and the routed fraction
    pc = h['per_cell']
    f3 = lambda a, co: np.mean([pc[a][f'{co}-{s}'] for s in (3, 5)])
    wins = 0
    for co in CORR:
        for k in ('neo', 'pair', 'b9g', 'b5b9g'):
            put(f'ex.h.fam.{co}.{k}', f3(A[k], co), '.1f')
        put(f'ex.h.fam.{co}.gain', f3(A['b5b9g'], co) - f3(A['pair'], co), '+.1f')
        put(f'ex.h.fam.{co}.routed', 100 * np.mean([h['routed'][A['b5b9g']][f'{co}-{s}'] for s in (3, 5)]), '.0f')
        wins += f3(A['b5b9g'], co) >= f3(A['pair'], co)
    put('ex.h.fam.wins', wins, 'd')
    neg = [co for co in CORR if f3(A['b5b9g'], co) < f3(A['pair'], co)]
    put('ex.h.fam.neg.n', len(neg), 'd'); put('ex.h.fam.neg.names', join_names(neg) if neg else 'none', 's')
    hc = E['held4']['cbar']['per_cell']; cb_cells = E['held4']['cbar']['cells']
    put('ex.h.cb.loses.n', sum(hc[A['b5b9g']][c] < hc[A['pair']][c] for c in cb_cells), 'd')
    cbc = sorted({c.rsplit('-', 1)[0] for c in cb_cells})
    f3c = lambda a, co: np.mean([hc[a][f'{co}-{sv}'] for sv in (3, 5)])
    lose = [co for co in cbc if f3c(A['b5b9g'], co) < f3c(A['pair'], co)]
    put('ex.h.cb.loses.corr.n', len(lose), 'd'); put('ex.h.cb.corr.n', len(cbc), 'd')
    # cost (design timing cells)
    for k, a in (('neo', 'neo'), ('pair', 'pair_k16_fp32_stats32'), ('branch', 'branch_k16_fp32_stats32'), ('b9', 'exit_b9_k16_fp32_stats32'), ('b5b9', 'exit_b5b9_k16_fp32_stats32')):
        t = E['timing'][a]; put(f'ex.cost.{k}.ms', t['ms_per_batch'], '.0f'); put(f'ex.cost.{k}.x', t['x_neo']); put(f'ex.cost.{k}.acc', t['acc_6cells'])
    put('ex.cost.b5b9.over.pair', E['timing']['exit_b5b9_k16_fp32_stats32']['ms_per_batch'] / E['timing']['pair_k16_fp32_stats32']['ms_per_batch'])
    put('ex.cost.b9.over.pair', E['timing']['exit_b9_k16_fp32_stats32']['ms_per_batch'] / E['timing']['pair_k16_fp32_stats32']['ms_per_batch'])
    put('ex.theta', 1.5, '.1f')
    # the cost of a set of second opinions in forward passes of the pair, from the blocks each branch runs at the routed share of
    # ImageNet-C (a branch at block k<6 runs blocks k+1..12, a branch at k>6 blocks 7..12, the pixel-only branch blocks 7..12)
    SCi, SCb = E['site_choice']['inc'], E['site_choice']['cbar']; f_r = SCi['controls']['routed_fraction']
    nblk = lambda a: 6 if a == 'px' else (12 - int(a[4:]) if int(a[4:]) < 6 else 6)
    pcost = lambda k: 1 + f_r * sum(nblk(a) for a in k.split('|')) / 12
    meang = {k: (SCi['second_opinions']['pairs'][k] + SCb['second_opinions']['pairs'][k]) / 2 for k in SCi['second_opinions']['pairs']}
    for lim, tag in ((1.40, 'lim'), (9.0, 'all')):
        best = max((k for k in meang if pcost(k) <= lim), key=meang.get)
        put(f'sc.so.best.{tag}', ' and '.join(x.replace('px+b', 'blocks ' if False else 'block ') for x in best.split('|')), 's')
        put(f'sc.so.best.{tag}.mean', meang[best], '+.2f'); put(f'sc.so.best.{tag}.cost', pcost(best), '.2f')
    put('sc.so.b5b9.mean', meang['px+b5|px+b9'], '+.2f'); put('sc.so.b5b9.cost', pcost('px+b5|px+b9'), '.2f'); put('sc.so.costlim', 1.40, '.1f')
    # the gated early-exit arms on the four ImageNet-C held-out slices (pair of each slice from its own run) and the two ImageNet-C-bar slices
    SLB = {'heldout': TS['slices_vitb']['heldout'], 'heldout2': TS['slices_vitb']['heldout2'], 'heldout3': TS['held3_vitb'], 'heldout4': E['held4']['inc']}
    SLE = {**E['slices'], 'heldout4': E['held4']['inc']}
    for k, a in (('b9g', 'exit_b9_gated'), ('b5b9g', 'exit_b5b9_gated'), ('pairg', 'pair_gated')):
        g = {sl: SLE[sl]['episodic'][a] - SLB[sl]['episodic']['tok+place_b6'] for sl in SLB}
        cl = {sl: SLE[sl]['clean'][a] - SLB[sl]['clean']['neo'] for sl in SLB}
        for sl in SLB:
            put(f'ex.sl.{sl}.{k}', SLE[sl]['episodic'][a]); put(f'ex.sl.{sl}.gain.{k}', g[sl], '+.2f'); put(f'ex.sl.{sl}.clean.{k}', cl[sl], '+.2f')
        put(f'ex.sl.gain.{k}.lo', min(g.values()), '+.2f'); put(f'ex.sl.gain.{k}.hi', max(g.values()), '+.2f'); put(f'ex.sl.gain.{k}.mean', np.mean(list(g.values())), '+.2f')
        put(f'ex.sl.clean.{k}.lo', min(cl.values()), '+.2f'); put(f'ex.sl.clean.{k}.hi', max(cl.values()), '+.2f')
        if a in E['slices']['heldout3'].get('routed_mean', {}):
            put(f'ex.sl.routed.{k}.lo', 100 * min(SLE[sl]['routed_mean'][a] for sl in SLB), '.0f'); put(f'ex.sl.routed.{k}.hi', 100 * max(SLE[sl]['routed_mean'][a] for sl in SLB), '.0f')
    for sl in SLB:
        put(f'ex.sl.{sl}.pair', SLB[sl]['episodic']['tok+place_b6']); put(f'ex.sl.{sl}.neo', SLB[sl]['episodic']['neo'])
    cb3 = E['slices']['cbar_heldout3']; cbp = CB['held3']['episodic']['tok+place_b6']
    for k, a in (('b9g', 'exit_b9_gated'), ('b5b9g', 'exit_b5b9_gated'), ('pairg', 'pair_gated')):
        put(f'ex.sl.cb3.{k}', cb3['episodic'][a]); put(f'ex.sl.cb3.gain.{k}', cb3['episodic'][a] - cbp, '+.2f')
    put('ex.sl.cb3.pair', cbp); put('ex.sl.n', len(SLB), 'd')
    # equal-compute controls on heldout4 (scored once), against the pair and the method of the heldout4 files
    for bench, key in (('inc', 'inc'), ('cb', 'cbar')):
        c, h = E['controls'][key], E['held4'][key]; pair, meth = h['episodic']['tok+place_b6'], h['episodic']['exit_b5b9_gated']
        for k, a in (('random', 'exit_b5b9_random_gated'), ('equal', 'exit_b5b9_equal_gated'), ('full', 'mix_b5b9_gated')):
            put(f'ex.ctl.{bench}.{k}', c['episodic'][a]); put(f'ex.ctl.{bench}.{k}.gain', c['episodic'][a] - pair, '+.2f')
            put(f'ex.ctl.{bench}.{k}.gaing', c['episodic'][a] - h['episodic']['pair_gated'], '+.2f')   # against the gated main path
            put(f'ex.ctl.{bench}.{k}.vsmethod', c['episodic'][a] - meth, '+.2f'); put(f'ex.ctl.{bench}.{k}.clean', c['clean'][a] - h['clean']['neo'], '+.2f')
            put(f'ex.ctl.{bench}.{k}.routed', 100 * c['routed_mean'][a], '.0f')
            put(f'ex.ctl.{bench}.{k}.cleanabs', c['clean'][a])
    # ViT-S/16 with the ViT-B/16 sites and threshold: design and heldout3 (scored once; NEO and the pair of heldout3 from Table 2's run)
    V, VS3 = E['vits'], TS['held3_vits']
    for k, a in (('pair', 'tok+place_b6'), ('neo', 'neo'), ('pairg', 'pair_gated'), ('b9g', 'exit_b9_gated'), ('b5b9g', 'exit_b5b9_gated')):
        put(f'ex.vs.d.{k}', V['design']['episodic'][a])
    for k, a in (('pairg', 'pair_gated'), ('b9g', 'exit_b9_gated'), ('b5b9g', 'exit_b5b9_gated')):
        put(f'ex.vs.d.gain.{k}', V['design']['episodic'][a] - V['design']['episodic']['tok+place_b6'], '+.2f')
        put(f'ex.vs.h.{k}', V['held3']['episodic'][a]); put(f'ex.vs.h.gain.{k}', V['held3']['episodic'][a] - VS3['episodic']['tok+place_b6'], '+.2f')
        put(f'ex.vs.h.clean.{k}', V['held3']['clean'][a] - VS3['clean']['neo'], '+.2f')
    put('ex.vs.h.pair', VS3['episodic']['tok+place_b6']); put('ex.vs.h.neo', VS3['episodic']['neo'])
    put('ex.vs.h.routed.b5b9g', 100 * V['held3']['routed_mean']['exit_b5b9_gated'], '.0f'); put('ex.vs.d.routed.b5b9g', 100 * V['design']['routed_mean']['exit_b5b9_gated'], '.0f')
    # the second-opinion stack on streams (heldout4, scored once): continual cont5 and the clean segments of cleanret
    ST = E['streams']['table']
    for k, a in (('neo', 'neo'), ('pairg', 'pair_gated'), ('b5b9g', 'exit_b5b9_gated')):
        put(f'ex.st.cont.{k}', ST[a]['cont']); put(f'ex.st.cr.{k}', ST[a]['cleanret_clean'])
    put('ex.st.cont.gain', ST['exit_b5b9_gated']['cont'] - ST['pair_gated']['cont'], '+.2f')
    put('ex.st.cont.gainneo', ST['exit_b5b9_gated']['cont'] - ST['neo']['cont'], '+.2f')
    put('ex.st.cr.vsneo', ST['exit_b5b9_gated']['cleanret_clean'] - ST['neo']['cleanret_clean'], '+.2f')
    # which block per cell or per image, from the saved design logits (site_choice.py)
    for bench, key in (('inc', 'inc'), ('cb', 'cbar')):
        sc = E['site_choice'][key]
        put(f'sc.{bench}.oracle.gain', sc['oracle_gain'], '+.2f'); put(f'sc.{bench}.choice.gain', sc['per_cell_entropy_choice_gain'], '+.2f')
        put(f'sc.{bench}.choice.gainneo', sc['per_cell_entropy_choice'] - sc['fixed_mean']['neo'], '+.2f')
        put(f'sc.{bench}.choice', sc['per_cell_entropy_choice']); put(f'sc.{bench}.oracle', sc['oracle_per_cell'])
        for name, m in sc['per_image_mixtures'].items():
            put(f'sc.{bench}.mix.{name}.full', m['full'] - sc['pair'], '+.2f'); put(f'sc.{bench}.mix.{name}.exit', m['early_exit'] - sc['pair'], '+.2f')
            put(f'sc.{bench}.mix.{name}.routed', 100 * m['routed_fraction'], '.0f')
        so = sc['second_opinions']; ps = so['per_site']
        for a, st in ps.items():
            k = a.replace('px+', '')
            put(f'sc.{bench}.so.{k}.agree', st['agree_routed'], '.0f'); put(f'sc.{bench}.so.{k}.rescue', st['rescue_routed'], '.1f')
            put(f'sc.{bench}.so.{k}.harm', st['harm_routed'], '.1f'); put(f'sc.{bench}.so.{k}.gain', st['gain_routed_mixture'], '+.2f')
        put(f'sc.{bench}.so.trunk.routed', so['trunk_acc_routed'], '.0f'); put(f'sc.{bench}.so.trunk.kept', so['trunk_acc_kept'], '.0f')
        pr = so['pairs']
        for a, b in (('px+b5', 'px+b9'), ('px+b3', 'px+b9'), ('px+b4', 'px+b9'), ('px+b5', 'px+b7'), ('px+b8', 'px+b9')):
            put(f"sc.{bench}.so.pair.{a.replace('px+', '')}{b.replace('px+', '')}", pr[f'{a}|{b}'], '+.2f')
        best = max(pr, key=pr.get); put(f'sc.{bench}.so.pair.best', ' and '.join(x.replace('px+b', 'block ').replace('px', 'the pixels alone') for x in best.split('|')), 's')
        put(f'sc.{bench}.so.pair.best.gain', pr[best], '+.2f')
        shallow, deep = ('px', 'px+b1', 'px+b2', 'px+b3', 'px+b4', 'px+b5'), ('px+b7', 'px+b8', 'px+b9', 'px+b10', 'px+b11', 'px+b12')
        top10 = sorted(pr, key=pr.get, reverse=True)[:10]
        put(f'sc.{bench}.so.pair.top10.opposite', sum((a in shallow) != (b in shallow) and (a in deep or b in deep) for a, b in (x.split('|') for x in top10)), 'd')
        for k2, ov in so['rescue_overlap_on_trunk_errors'].items():
            a, b = k2.split('|'); kk = f"{a.replace('px+', '')}{b.replace('px+', '')}"
            put(f'sc.{bench}.so.ov.{kk}.either', ov['either'], '.0f'); put(f'sc.{bench}.so.ov.{kk}.both', ov['both'], '.1f')
        lf = sc['label_free']; nm = lambda k: ' and '.join(x.replace('px+b', 'block ') for x in k.split('|'))
        put(f'sc.{bench}.lf.trunk.pick', nm(lf['trunk_pick_by_entropy']), 's'); put(f'sc.{bench}.lf.trunk.best', nm(lf['trunk_best_by_accuracy']), 's')
        put(f'sc.{bench}.lf.trunk.rho', lf['trunk_spearman_confidence_accuracy'], '+.2f')
        put(f'sc.{bench}.lf.so.pick', nm(lf['second_opinion_pick_by_entropy']), 's'); put(f'sc.{bench}.lf.so.pick.gain', lf['second_opinion_pick_gain'], '+.2f')
        put(f'sc.{bench}.lf.so.best', nm(lf['second_opinion_best']), 's'); put(f'sc.{bench}.lf.so.best.gain', lf['second_opinion_best_gain'], '+.2f')
        put(f'sc.{bench}.lf.so.share', 100 * lf['second_opinion_pick_gain'] / lf['second_opinion_best_gain'], '.0f')
        put(f'sc.{bench}.lf.so.rho', lf['second_opinion_spearman_confidence_gain'], '+.2f'); put(f'sc.{bench}.lf.so.rho.disagree', lf['second_opinion_spearman_disagreement_gain'], '+.2f')
        ctl = sc['controls']
        for k in ('entropy_routing', 'entropy_routing_equal_weights', 'random_routing_same_fraction', 'most_confident_routed_same_fraction',
                  'full_mixture_every_image', 'entropy_routing_b9_only', 'entropy_routing_b5_only', 'oracle_routing_trunk_errors'):
            put(f'sc.{bench}.ctl.{k}', ctl[k] - sc['pair'], '+.2f')


def rule():
    """The rule-chosen second-opinion stack: held-out results that exist (gains against the pair, NEO and the labelled 5+9 stack of
    the same slice), the rule's picks and the sample-size analysis. Prefix rl."""
    R = v('rule'); E = v('exit')
    base = {'inc': E['held4']['inc'], 'cbar': E['held4']['cbar']}
    names = {'rule_t6_b4b11_gated': 'b4b11', 'rule_t6_b4_gated': 'b4', 'rule_t6_b11_gated': 'b11', 'rule_t7_b5b11_gated': 'own', 'tok+place_b7': 'pair7',
             'rule_t5_b3b7_gated': 'small', 'rule_t7_b5b9_gated': 'small', 'tok+place_b5': 'pair5',
             'rule_t6_b3b9_gated': 'b3b9', 'rule_t6_b1b12_gated': 'b1b12', 'rule_t6_b5b7_gated': 'b5b7'}   # the pairs run as controls after the review
    for key, f in R['held4'].items():
        if f is None:
            continue
        bench = key.split('_')[0]; b = base[bench]
        pair, neo, meth = b['episodic']['pair_gated'], b['episodic']['neo'], b['episodic']['exit_b5b9_gated']   # the gated main path, as the cascade is gated
        # the full-sample main path of the benchmark: block 6 on ImageNet-C, block 7 on ImageNet-C-bar (its own calibration), the one
        # reference of every row of the calibration-size table
        fullmain = {'inc': pair, 'cbar': R['held4']['cbar']['episodic']['tok+place_b7']}
        for arm, acc in f['episodic'].items():
            k = f'rl.h.{bench}.{names[arm]}'
            put(k, acc); put(k + '.gain', acc - pair, '+.2f'); put(k + '.gainneo', acc - neo, '+.2f'); put(k + '.vs59', acc - meth, '+.2f')
            put(k + '.gainfull', acc - fullmain[bench], '+.2f')
            put(k + '.clean', f['clean'][arm]); put(k + '.cleand', f['clean'][arm] - b['clean']['neo'], '+.2f')
            if arm in f['routed']:
                put(k + '.routed', 100 * f['routed'][arm], '.0f')
    # ViT-S/16: the rule's own sweep, picks and held-out run (baselines of heldout3: Table 2's run and the 5+9 transfer)
    if R['vits']['held3'] is not None:
        b = v('two_site')['held3_vits']                                 # NEO and the pair of ViT-S/16 on heldout3 (the run of the pair's table)
        pair, neo = b['episodic']['tok+place_b6'], b['episodic']['neo']
        meth = E['vits']['held3']['episodic']['exit_b5b9_gated']        # the ViT-B/16 blocks carried over
        put('rl.vs.pair', pair); put('rl.vs.neo', neo); put('rl.vs.b5b9.gain', meth - pair, '+.2f')
        arms = list(R['vits']['held3']['episodic'])
        for arm, tag in zip(arms, ('both', 'lo', 'hi')):
            acc = R['vits']['held3']['episodic'][arm]; k = f'rl.vs.{tag}'
            put(k + '.arm', arm.replace('_', '\\_'), 's'); put(k, acc); put(k + '.gain', acc - pair, '+.2f'); put(k + '.gainneo', acc - neo, '+.2f'); put(k + '.vs59', acc - meth, '+.2f')
            put(k + '.clean', R['vits']['held3']['clean'][arm]); put(k + '.cleand', R['vits']['held3']['clean'][arm] - b['clean']['neo'], '+.2f')
            if arm in R['vits']['held3']['routed']:
                put(k + '.routed', 100 * R['vits']['held3']['routed'][arm], '.0f')
    if R['vits']['pick'] is not None:
        pk = R['vits']['pick']; put('rl.vs.pick.trunk', pk['trunk_pick_by_entropy'][4:], 's'); put('rl.vs.pick.lo', pk['second_opinion_pick_shallower'][4:], 's'); put('rl.vs.pick.hi', pk['second_opinion_pick_deeper'][4:], 's')
        put('rl.vs.pick.best', R['vits']['pick_t6']['trunk_best_by_accuracy'][4:], 's')
        put('rl.vs.theta65', pk['trunk_entropy_quantile']['0.65'])                                   # the quantile rule's threshold on ViT-S/16 (its run used 1.5 nat)
    # the other held-out image sets and the design split (gain over the pair of the same set), the random-routing control, streams, timing
    TSS = v('two_site')
    refs = {'heldout': TSS['slices_vitb']['heldout']['episodic'] if 'slices_vitb' in TSS and 'heldout' in TSS['slices_vitb'] else None,
            'heldout2': TSS['slices_vitb']['heldout2']['episodic'] if 'slices_vitb' in TSS and 'heldout2' in TSS['slices_vitb'] else None,
            'heldout3': TSS['held3_vitb']['episodic'], 'cbar_heldout3': v('extra/cbar')['held3']['episodic'],
            'design': TSS['design_vitb']['episodic'], 'cbar_design': v('extra/cbar')['design']['episodic'] if 'design' in v('extra/cbar') else None}
    gains = []
    for key, f in R['slices'].items():
        if f is None or refs.get(key) is None:
            continue
        ref = refs[key]; acc = f['episodic']['rule_t6_b4b11_gated']
        put(f'rl.sl.{key}.b4b11', acc); put(f'rl.sl.{key}.gain', acc - ref['tok+place_b6'], '+.2f'); put(f'rl.sl.{key}.gainneo', acc - ref['neo'], '+.2f')
        put(f'rl.sl.{key}.pair', ref['tok+place_b6']); put(f'rl.sl.{key}.neo', ref['neo'])
        if key in ('heldout', 'heldout2', 'heldout3'):
            gains.append(acc - ref['tok+place_b6'])
    inc4 = R['held4']['inc']['episodic']['rule_t6_b4b11_gated'] - base['inc']['episodic']['tok+place_b6']
    allg = gains + [inc4]
    put('rl.sl.gain.lo', min(allg), '+.2f'); put('rl.sl.gain.hi', max(allg), '+.2f'); put('rl.sl.gain.mean', sum(allg) / len(allg), '+.2f'); put('rl.sl.n', len(allg), 'd')
    for bench, f in R['random'].items():
        if f is None:
            continue
        b = base[bench]; acc = f['episodic']['rule_t6_b4b11_random_gated']
        put(f'rl.ctl.{bench}.random', acc); put(f'rl.ctl.{bench}.random.gain', acc - b['episodic']['pair_gated'], '+.2f'); put(f'rl.ctl.{bench}.random.routed', 100 * f['routed']['rule_t6_b4b11_random_gated'], '.0f')
        put(f'rl.ctl.{bench}.random.cleanabs', f['clean']['rule_t6_b4b11_random_gated'])
        put(f'rl.ctl.{bench}.random.cleand', f['clean']['rule_t6_b4b11_random_gated'] - b['clean']['neo'], '+.2f')
    for bench, f in R.get('views', {}).items():                       # the views control: the main path on a flip and a center crop of the same unsure images
        if f is None:
            continue
        b = base[bench]; acc = f['episodic']['rule_t6_views_gated']
        put(f'rl.ctl.{bench}.views', acc); put(f'rl.ctl.{bench}.views.gain', acc - b['episodic']['pair_gated'], '+.2f'); put(f'rl.ctl.{bench}.views.routed', 100 * f['routed']['rule_t6_views_gated'], '.0f')
        put(f'rl.ctl.{bench}.views.cleand', f['clean']['rule_t6_views_gated'] - b['clean']['neo'], '+.2f')
        casc = R['held4'][bench]['episodic']['rule_t6_b4b11_gated'] - b['episodic']['pair_gated']            # the share of the branch gain the views keep
        put(f'rl.ctl.{bench}.views.share', 100 * (acc - b['episodic']['pair_gated']) / casc, '.0f')
    if R['streams'] is not None:
        t = R['streams']['table']
        for a, key in (('neo', 'neo'), ('pair_gated', 'pairg'), ('exit_b5b9_gated', 'b5b9g'), ('rule_t6_b4b11_gated', 'b4b11')):
            put(f'rl.st.cont.{key}', t[a]['cont']); put(f'rl.st.cr.{key}', t[a]['cleanret_clean'])
        put('rl.st.cont.gain', round(t['rule_t6_b4b11_gated']['cont'], 2) - round(t['pair_gated']['cont'], 2), '+.2f'); put('rl.st.cont.gainneo', t['rule_t6_b4b11_gated']['cont'] - t['neo']['cont'], '+.2f')
        put('rl.st.cr.d', t['rule_t6_b4b11_gated']['cleanret_clean'] - t['neo']['cleanret_clean'], '+.2f')
    if R.get('streams_eta') is not None:                            # ETA and SAR (alone, with NEO applied to the recorded head input, and ETA with the main path inside) on the same streams
        te = R['streams_eta']['table']; t = R['streams']['table']
        for a, key in (('neo', 'neo2'), ('eta', 'eta'), ('eta+neo', 'etaneo'), ('sar', 'sar'), ('sar+neo', 'sarneo'), ('twosite+eta_gated', 'etamain'), ('twosite+eta_gated+neo', 'etamainneo')):
            if a in te:
                put(f'rl.st.cont.{key}', te[a]['cont']); put(f'rl.st.cr.{key}', te[a]['cleanret_clean'])
                put(f'rl.st.cont.{key}.vscascade', te[a]['cont'] - t['rule_t6_b4b11_gated']['cont'], '+.2f'); put(f'rl.st.cont.{key}.vsneo', te[a]['cont'] - te['neo']['cont'], '+.2f')
                put(f'rl.st.cont.cascade.vs.{key}', t['rule_t6_b4b11_gated']['cont'] - te[a]['cont'], '+.2f')
    # wall-clock per batch relative to NEO: our rows, ETA and SAR from one run, PEA and FOA from a second run with its own NEO reference
    for a, key in (('neo', 'neo'), ('tok_k16_fp32_stats32', 'tok'), ('pair_k16_fp32_stats32', 'pair'), ('exit_b5b9_k16_fp32_stats32', 'b5b9'), ('rule_t6_b4b11_k16_fp32_stats32', 'b4b11'),
                   ('rule_t6_b4_k16_fp32_stats32', 'b4'), ('rule_t6_b11_k16_fp32_stats32', 'b11'), ('rule_t6_b4b11_random_k16_fp32_stats32', 'random'), ('eta', 'eta'), ('sar', 'sar')):
        t = (R['timing'] or {}).get(a)
        if t is None:
            put(f'rl.cost.{key}.ms', '--', 's'); put(f'rl.cost.{key}.x', '--', 's')
        else:
            put(f'rl.cost.{key}.ms', t['ms_per_batch'], '.0f'); put(f'rl.cost.{key}.x', t['x_neo'])
    for a, key in (('pea', 'pea'), ('pea_aug', 'peaviews'), ('foa', 'foa')):
        t = (R.get('timing_others') or {}).get(a)
        if t is None:
            put(f'rl.cost.{key}.ms', '--', 's'); put(f'rl.cost.{key}.x', '--', 's')
        else:
            put(f'rl.cost.{key}.ms', t['ms_per_batch'], '.0f'); put(f'rl.cost.{key}.x', t['x_neo'])
    # analytic compute per batch in forward passes (paper/compute_cost.py), with the unsure shares of the held-out runs
    from compute_cost import VITB_GMACS as GM, forward_passes
    u = R['held4']['inc']['routed']['rule_t6_b4b11_gated']; ua = R['held4']['inc_abl']['routed']
    u59 = float(np.mean([f for c, f in E['held4']['inc']['routed']['exit_b5b9_gated'].items() if c != 'clean']))
    for key, fp in forward_passes(u, ua['rule_t6_b4_gated'], ua['rule_t6_b11_gated'], R['random']['inc']['routed']['rule_t6_b4b11_random_gated'], u59).items():
        put(f'rl.macs.{key}', fp * GM, '.1f')
    # per corruption on heldout4 for the calibrated stack (mean of severities 3 and 5): ImageNet-C (rl.h.fam.*) and ImageNet-C-bar (rl.h.cfam.*)
    CBARC = ('blue_noise_sample', 'brownish_noise', 'caustic_refraction', 'checkerboard_cutout', 'cocentric_sine_waves', 'inverse_sparkles', 'perlin_noise', 'plasma_noise', 'single_frequency_greyscale', 'sparkles')
    for bench, corrs, pre in (('inc', CORR, 'rl.h.fam'), ('cbar', CBARC, 'rl.h.cfam')):
        f = R['held4'][bench]
        if f is None:
            continue
        pc, bpc = f['per_cell']['rule_t6_b4b11_gated'], base[bench]['per_cell']
        g3 = lambda d, co: np.mean([d[f'{co}-{s}'] for s in (3, 5)])
        wins = 0
        for co in corrs:
            put(f'{pre}.{co}.neo', g3(bpc['neo'], co), '.1f'); put(f'{pre}.{co}.pair', g3(bpc['tok+place_b6'], co), '.1f'); put(f'{pre}.{co}.stack', g3(pc, co), '.1f')
            put(f'{pre}.{co}.gain', g3(pc, co) - g3(bpc['tok+place_b6'], co), '+.1f'); put(f'{pre}.{co}.routed', 100 * np.mean([f['routed_per_cell']['rule_t6_b4b11_gated'][f'{co}-{s}'] for s in (3, 5)]), '.0f')
            wins += g3(pc, co) >= g3(bpc['tok+place_b6'], co)
        put(f'{pre}.wins', wins, 'd'); put(f'{pre}.n', len(corrs), 'd')
    put('rl.ss.inc.theta65', R['picks']['inc']['trunk_entropy_quantile']['0.65']); put('rl.ss.cbar.theta65', R['picks']['cbar_t6']['trunk_entropy_quantile']['0.65'])
    # the picks on the design images
    for key, pk in R['picks'].items():
        put(f'rl.pick.{key}.trunk', pk['trunk_pick_by_entropy'][4:], 's'); put(f'rl.pick.{key}.lo', pk['second_opinion_pick_shallower'][4:], 's'); put(f'rl.pick.{key}.hi', pk['second_opinion_pick_deeper'][4:], 's')
        put(f'rl.pick.{key}.gain', pk['second_opinion_pick_pair_gain'], '+.2f'); put(f'rl.pick.{key}.best.gain', pk['second_opinion_best_gain'], '+.2f')
        put(f'rl.pick.{key}.best', pk['second_opinion_best'].replace('px+b', '').replace('|', ' and '), 's')
        put(f'rl.pick.{key}.share', 100 * pk['second_opinion_pick_pair_gain'] / pk['second_opinion_best_gain'], '.0f')
    # how many unlabeled images the rule needs
    small_ns = {}
    # the sample-size file measures every gain against the block-6 main path of the dev predictions; on ImageNet-C-bar the full-sample
    # main block is 7, so its gains are moved to that reference (the accuracy of block 7 minus block 6 on the same predictions)
    A = R['picks']['cbar_t7']['trunk_accuracy']; shift = {'inc': 0.0, 'cbar': A['px+b7'] - A['px+b6']}
    put('rl.ss.cbar.ref.shift', shift['cbar'], '+.2f')
    for bench, ss in R['sample_size'].items():
        put(f'rl.ss.{bench}.full.gain', ss['full_rule']['gain_over_pair'] - shift[bench], '+.2f')
        for n, d in ss['pooled_first_n_per_cell'].items():
            put(f'rl.ss.{bench}.n{n}.cfg', f"{d['trunk'][4:]}/{d['shallower'][4:]},{d['deeper'][4:]}", 's'); put(f'rl.ss.{bench}.n{n}.gain', d['gain_over_pair'] - shift[bench], '+.2f')
        for K, d in ss['per_cell_first_k_batches'].items():
            put(f'rl.ss.{bench}.k{K}.gain', d['gain_over_pair'] - shift[bench], '+.2f'); put(f'rl.ss.{bench}.k{K}.n', d['images_per_cell'], 'd')
        first = [n for n, d in ss['pooled_first_n_per_cell'].items() if d['same_as_full']]
        put(f'rl.ss.{bench}.n.same', first[0] if first else 'none', 's')
        # the small-sample calibration run held out (the arms rule_t5_b3b7_gated and rule_t7_b5b9_gated of the held4 small runs, 'small' in rule()):
        # the n per cell whose pooled pick is that configuration, and its gain on the saved predictions
        pooled = ss['pooled_first_n_per_cell']; ns = sorted(pooled, key=int); cfg = lambda d: '{}/{},{}'.format(*(x[4:] for x in (d['trunk'], d['shallower'], d['deeper'])))
        small = {'inc': '5/3,7', 'cbar': '7/5,9'}[bench]; same = [n for n in ns if cfg(pooled[n]) == small]
        put(f'rl.ss.{bench}.small.cfg', small, 's'); put(f'rl.ss.{bench}.small.gain', pooled[same[-1]]['gain_over_pair'] - shift[bench], '+.2f')
        put(f'rl.ss.{bench}.small.ns', (', '.join(same[:-1]) + ' and ' + same[-1]) if len(same) > 1 else same[0], 's')
        small_ns[bench] = set(same)
    put('rl.ss.small.n', max(small_ns['inc'] & small_ns['cbar'], key=int), 's')   # the largest n per cell that gives both held-out small picks
    # one calibration per shift from 192 images, held out (addendum 5), against the full-sample calibration on the same cells
    full = {'inc': (R['held4']['inc']['episodic']['rule_t6_b4b11_gated'], R['held4']['inc']['episodic']['tok+place_b6'] if 'tok+place_b6' in R['held4']['inc']['episodic'] else E['held4']['inc']['episodic']['pair_gated']),
            'cbar': (R['held4']['cbar']['episodic']['rule_t7_b5b11_gated'], R['held4']['cbar']['episodic']['tok+place_b7'])}
    for bench in ('inc', 'cbar'):
        for K in ('k1', 'k3', 'k10'):
            f = R.get('fewshot', {}).get(bench, {}).get(K)
            if f is None:
                for sub in ('stack', 'base', 'neo', 'gain', 'gainneo', 'vsfull', 'vsfullbase', 'm5'):
                    put(f'rl.fs.{bench}.{K}.{sub}', '--', 's')
                continue
            put(f'rl.fs.{bench}.{K}.stack', f['stack']); put(f'rl.fs.{bench}.{K}.base', f['base']); put(f'rl.fs.{bench}.{K}.neo', f['neo'])
            put(f'rl.fs.{bench}.{K}.gain', f['stack'] - f['base'], '+.2f'); put(f'rl.fs.{bench}.{K}.gainneo', f['stack'] - f['neo'], '+.2f')
            put(f'rl.fs.{bench}.{K}.vsfull', f['stack'] - full[bench][0], '+.2f'); put(f'rl.fs.{bench}.{K}.vsfullbase', f['stack'] - full[bench][1], '+.2f')
            put(f'rl.fs.{bench}.{K}.vsfullbase.base', f['base'] - full[bench][1], '+.2f')     # the per-shift main path against the full-sample main path
            put(f'rl.fs.{bench}.{K}.m5', 'holds' if abs(f['stack'] - full[bench][0]) <= 0.5 and f['stack'] > full[bench][1] else 'fails', 's')
    p192 = R.get('pooled192_cbar')
    if p192 is not None:
        put('rl.ss.cbar.p192.held', p192['episodic']['rule_t7_b5b8_gated']); put('rl.ss.cbar.p192.held.gain', p192['episodic']['rule_t7_b5b8_gated'] - full['cbar'][1], '+.2f')
    else:
        put('rl.ss.cbar.p192.held.gain', '--', 's')
    # checks from the saved logits (scripts/analysis/review_checks.py): intervals of the held-out gains, the correlation without the
    # first three blocks, the threshold, the pick among every pair, the entropy a pick saves against TokAlign alone. Prefix rc.
    rc = R.get('review')
    if rc is not None:
        put('rc.bootstrap', rc['bootstrap_draws'], 'd')
        for key, r in rc['pairs'].items():
            put(f'rc.{key}.gain', r['gain'], '+.2f'); put(f'rc.{key}.ci.lo', r['ci_images'][0], '+.2f'); put(f'rc.{key}.ci.hi', r['ci_images'][1], '+.2f')
            put(f'rc.{key}.cic.lo', r['ci_cells'][0], '+.2f'); put(f'rc.{key}.cic.hi', r['ci_cells'][1], '+.2f'); put(f'rc.{key}.n', r['n_cells'], 'd')
            put(f'rc.{key}.rho', r['rho_all'], '+.2f'); put(f'rc.{key}.rho4', r['rho_from_b4'], '+.2f'); put(f'rc.{key}.saved', r['entropy_saved_vs_px'], '.2f')
            put(f'rc.{key}.gap', r['gap'], '+.2f'); put(f'rc.{key}.pick', r['pick'][4:] if r['pick'] != 'px' else 'px', 's'); put(f'rc.{key}.above', r['cells_above'], 'd')
            put(f'rc.{key}.ci.zero', 'yes' if r['ci_images'][0] <= 0 <= r['ci_images'][1] else 'no', 's')
        corr = [k for k in rc['pairs'] if k.endswith('_inc') or k.endswith('_cbar')]; nat = [k for k in rc['pairs'] if k not in corr]
        put('rc.corr.rho4.min', min(rc['pairs'][k]['rho_from_b4'] for k in corr), '+.2f'); put('rc.corr.rho4.max', max(rc['pairs'][k]['rho_from_b4'] for k in corr), '+.2f')
        put('rc.nat.rho4.min', min(rc['pairs'][k]['rho_from_b4'] for k in nat), '+.2f'); put('rc.nat.rho4.max', max(rc['pairs'][k]['rho_from_b4'] for k in nat), '+.2f')
        put('rc.corr.ci.zero.n', sum(rc['pairs'][k]['ci_images'][0] <= 0 for k in corr), 'd'); put('rc.corr.n', len(corr), 'd'); put('rc.nat.n', len(nat), 'd')
        put('rc.corr.cic.zero.n', sum(rc['pairs'][k]['ci_cells'][0] <= 0 for k in corr), 'd')
        pname = {'vitb': 'ViT-B/16', 'vits': 'ViT-S/16', 'deitb': 'DeiT-B/16', 'deits': 'DeiT-S/16'}; dname = {'inc': 'ImageNet-C', 'cbar': r'ImageNet-$\overline{\text{C}}$'}
        put('rc.corr.cic.zero.names', ' and '.join(f"{pname[k.split('_')[0]]} on {dname[k.split('_')[1]]}" for k in corr if rc['pairs'][k]['ci_cells'][0] <= 0), 's')
        half = lambda k: (rc['pairs'][k]['ci_images'][1] - rc['pairs'][k]['ci_images'][0]) / 2
        put('rc.corr.half.max', max(half(k) for k in corr), '.2f'); put('rc.nat.half.max', max(half(k) for k in nat), '.2f'); put('rc.nat.half.min', min(half(k) for k in nat), '.2f')
        put('rc.corr.saved.min', min(rc['pairs'][k]['entropy_saved_vs_px'] for k in corr), '.2f'); put('rc.corr.saved.max', max(rc['pairs'][k]['entropy_saved_vs_px'] for k in corr), '.2f')
        miss = [k for k in nat if rc['pairs'][k]['gap'] < -0.2]; hit = [k for k in nat if k not in miss]   # a miss: more than 0.2 points from the best block (Fig S3)
        put('rc.nat.miss.saved', ', '.join(f"{rc['pairs'][k]['entropy_saved_vs_px']:.2f}" for k in miss), 's'); put('rc.nat.hit.saved.min', min(rc['pairs'][k]['entropy_saved_vs_px'] for k in hit), '.2f')
        put('rc.nat.hit.saved.max', max(rc['pairs'][k]['entropy_saved_vs_px'] for k in hit), '.2f'); put('rc.nat.miss.n', len(miss), 'd')
        for b, r in rc['table1'].items():
            put(f'rc.t1.{b}.gain', r['gain'], '+.2f'); put(f'rc.t1.{b}.ci.lo', r['ci_images'][0], '+.2f'); put(f'rc.t1.{b}.ci.hi', r['ci_images'][1], '+.2f')
            put(f'rc.t1.{b}.cic.lo', r['ci_cells'][0], '+.2f'); put(f'rc.t1.{b}.cic.hi', r['ci_cells'][1], '+.2f'); put(f'rc.t1.{b}.above', r['cells_above'], 'd')
            put(f'rc.t1.{b}.changed', r['changed'], '.1f'); put(f'rc.t1.{b}.fixed', r['fixed'], '.1f'); put(f'rc.t1.{b}.broken', r['broken'], '.1f')
            put(f'rc.t1.{b}.half', (r['ci_images'][1] - r['ci_images'][0]) / 2, '.2f'); put(f'rc.t1.{b}.halfc', (r['ci_cells'][1] - r['ci_cells'][0]) / 2, '.2f')
        for q, r in rc['theta'].items():
            tag = q.replace('.', '').replace('nat', '')
            put(f'rc.theta.q{tag}.theta', r['theta'], '.2f'); put(f'rc.theta.q{tag}.gain', r['gain'], '+.2f'); put(f'rc.theta.q{tag}.unsure', r['unsure'], '.0f')
        pd = rc['pairs_dev']
        put('rc.pairs.n', pd['n'], 'd'); put('rc.pairs.pick.gain', pd['pick_gain'], '+.2f'); put('rc.pairs.pick.rank', pd['pick_rank'], 'd'); put('rc.pairs.best', pd['best'].replace('|', ' and '), 's')
        put('rc.pairs.best.gain', pd['best_gain'], '+.2f'); put('rc.pairs.mean', pd['mean'], '+.2f'); put('rc.pairs.median', pd['median'], '+.2f'); put('rc.pairs.worst', pd['worst'].replace('|', ' and '), 's')
        put('rc.pairs.worst.gain', pd['worst_gain'], '+.2f'); put('rc.pairs.noadj.mean', pd['mean_no_adjacent'], '+.2f'); put('rc.pairs.noadj.n', pd['n_no_adjacent'], 'd')
        put('rc.pairs.adj57', pd['adjacent_5_7'], '+.2f'); put('rc.pairs.ends112', pd['ends_1_12'], '+.2f')


def models_datasets():
    """More models and natural shifts: per pair the calibration picks, the signal check, the held-out gains against NEO and the base
    path, the ablations and the ETA comparison. Prefix ca.<model>_<dataset>."""
    CA = v('ca')
    for key, p in CA.items():
        if key.startswith('_') or not isinstance(p, dict) or 'calib' not in p:
            continue
        c = p['calib']; k = f'ca.{key}'
        put(k + '.trunk', c['trunk'], 'd'); put(k + '.lo', c['shallower'], 'd'); put(k + '.hi', c['deeper'], 'd'); put(k + '.theta', c['theta'])
        sig = p['signal']; best = sig['trunk_best_by_accuracy']; acc = sig['trunk_accuracy']
        put(k + '.best', best[4:] if best.startswith('px+b') else 'none', 's'); put(k + '.rho', sig['trunk_spearman_confidence_accuracy'], '+.2f')
        put(k + '.pick.acc', acc[sig['trunk_pick_by_entropy']]); put(k + '.best.acc', acc[best]); put(k + '.pick.gap', acc[sig['trunk_pick_by_entropy']] - acc[best], '+.2f')
        sw = p['sweep']['episodic']; put(k + '.d.neo', sw['neo']); put(k + '.d.px', sw['tok']); put(k + '.best.gainneo', acc[best] - sw['neo'], '+.2f')
        put(k + '.so.gain', p['second_opinions']['second_opinion_pick_pair_gain'], '+.2f'); put(k + '.so.best.gain', p['second_opinions']['second_opinion_best_gain'], '+.2f')
        if 'held' in p:
            h = p['held']['episodic']; T = c['trunk']; base = h[f'tok+place_b{T}']; stack = h[c['arm']]
            put(k + '.h.neo', h['neo']); put(k + '.h.px', h['tok']); put(k + '.h.base', base); put(k + '.h.base.gainneo', base - h['neo'], '+.2f')
            put(k + '.h.stack', stack); put(k + '.h.stack.gain', stack - base, '+.2f'); put(k + '.h.stack.gainneo', stack - h['neo'], '+.2f'); put(k + '.h.stack.gainpx', stack - h['tok'], '+.2f')
            put(k + '.h.stack.clean', p['held']['clean'][c['arm']]); put(k + '.h.stack.cleand', p['held']['clean'][c['arm']] - p['held']['clean']['neo'], '+.2f')
            put(k + '.h.stack.routed', 100 * p['held']['routed'][c['arm']], '.0f')
            put(k + '.h.lo.gain', h[f"rule_t{T}_b{c['shallower']}_gated"] - base, '+.2f'); put(k + '.h.hi.gain', h[f"rule_t{T}_b{c['deeper']}_gated"] - base, '+.2f')
            put(k + '.h.m2', 'holds' if stack - base >= 0.5 and p['held']['clean'][c['arm']] - p['held']['clean']['neo'] >= -0.2 else 'fails', 's')
        if 'eta' in p:
            e = p['eta']['episodic']; T = c['trunk']
            put(k + '.eta.neo', e['eta+neo']); put(k + '.eta.ours', e[f'tok+eta+place_b{T}+neo']); put(k + '.eta.ours.gain', e[f'tok+eta+place_b{T}+neo'] - e['eta+neo'], '+.2f')
            es = CA.get('etastack', {}).get(key)             # ETA with the whole calibrated stack inside, same held-out cells
            if es is not None:
                put(k + '.eta.stack', es['episodic'][f"tok+eta+rule_t{T}_b{c['shallower']}b{c['deeper']}+neo"])
            else:
                put(k + '.eta.stack', '--', 's')
            if 'held' in p:
                put(k + '.h.stack.vseta', p['held']['episodic'][c['arm']] - e['eta+neo'], '+.2f')


def across():
    """One row per model-dataset pair for the cross-model table, all on the same held-out images: the calibration picks, the
    signal's rank correlation, and the gains over NEO of the base path, the calibrated stack, ETA (+NEO) and ETA with the base
    path's alignment inside. Prefix ax.<pair>."""
    R, E, CA, TS = v('rule'), v('exit'), v('ca'), v('two_site')
    ES = CA.get('etastack', {})
    def row(key, trunk, lo, hi, rho, neo, base, stack, eta=None, etaours=None, clean=None, eta_clean=None):
        put(f'ax.{key}.pick', f'{trunk}/{lo},{hi}', 's'); put(f'ax.{key}.rho', rho, '+.2f')
        put(f'ax.{key}.base', base - neo, '+.2f'); put(f'ax.{key}.stack', stack - neo, '+.2f'); put(f'ax.{key}.stack.vsbase', round(stack - neo, 2) - round(base - neo, 2), '+.2f')   # the difference of the two printed gains
        put(f'ax.{key}.eta', eta - neo if eta is not None else float('nan'), '+.2f') if eta is not None else put(f'ax.{key}.eta', '--', 's')
        put(f'ax.{key}.etaours', etaours - neo, '+.2f') if etaours is not None else put(f'ax.{key}.etaours', '--', 's')
        if clean is not None:
            put(f'ax.{key}.clean', clean, '+.2f')
        es = ES.get(key)                                   # ETA with the calibrated stack inside, same cells
        if es is not None:
            arm = f'tok+eta+rule_t{trunk}_b{lo}b{hi}+neo'
            put(f'ax.{key}.etastack', es['episodic'][arm] - neo, '+.2f'); put(f'ax.{key}.etastack.vsetaours', round(es['episodic'][arm], 2) - round(etaours, 2), '+.2f') if etaours is not None else None
            put(f'ax.{key}.etastack.abs', es['episodic'][arm]); put(f'ax.{key}.etastack.routed', 100 * es['routed'].get(arm, float('nan')), '.0f')
            if eta_clean is not None:                      # the clean cell against ETA with NEO on the same split (claim M4)
                put(f'ax.{key}.etastack.cleand', es['clean'][arm] - eta_clean, '+.2f'); put(f'ax.{key}.etastack.cleand.abs', abs(es['clean'][arm] - eta_clean), '.2f')
                put(f'ax.{key}.m4', 'holds' if es['episodic'][arm] - etaours >= 0.5 and es['clean'][arm] - eta_clean >= -0.2 else 'fails', 's')
        else:
            put(f'ax.{key}.etastack', '--', 's')
    def blank(key):
        for sub in ('pick', 'rho', 'base', 'stack', 'eta', 'etaours', 'etastack'):
            put(f'ax.{key}.{sub}', '--', 's')
    # ViT-B/16 and ViT-S/16 on ImageNet-C: heldout3 (the slice with the ETA runs); the calibration from the ImageNet-C design images
    b = TS['held3_vitb']['episodic']; lfb = E['site_choice']['inc']['label_free']
    row('vitb_inc', 6, 4, 11, lfb['trunk_spearman_confidence_accuracy'], b['neo'], b['tok+place_b6'], R['slices']['heldout3']['episodic']['rule_t6_b4b11_gated'], b['eta+neo'], b['tok+eta+place_b6+neo'],
        eta_clean=TS['held3_vitb']['clean']['eta+neo'])
    s = TS['held3_vits']['episodic']; lfs = R['vits']['pick_t6']
    row('vits_inc', 6, 4, 11, lfs['trunk_spearman_confidence_accuracy'], s['neo'], s['tok+place_b6'], R['vits']['held3']['episodic']['rule_t6_b4b11_gated'], s['eta+neo'], s['tok+eta+place_b6+neo'],
        eta_clean=TS['held3_vits']['clean']['eta+neo'])
    # ViT-B/16 on ImageNet-C-bar: heldout4, its own calibration (trunk 7) and the base path of that trunk
    c = R['held4']['cbar']['episodic']; cb = E['held4']['cbar']['episodic']; lfc = E['site_choice']['cbar']['label_free']
    ec = CA.get('eta_vitb_cbar', {}).get('episodic')
    row('vitb_cbar', 7, 5, 11, lfc['trunk_spearman_confidence_accuracy'], cb['neo'], c['tok+place_b7'], c['rule_t7_b5b11_gated'],
        ec['eta+neo'] if ec else None, ec['tok+eta+place_b7+neo'] if ec else None, eta_clean=CA['eta_vitb_cbar']['clean']['eta+neo'] if ec else None)
    for key in ('vits_cbar', 'deitb_cbar'):                 # the grid's cells that have not been run yet render as '--'
        if key not in CA or 'held' not in CA[key]:
            blank(key)
    # block 1 among the 13 placements on the eight corruption pairs (Section C.2): least accurate everywhere, how often least confident
    corr_lf = [lfb, lfs, lfc] + [CA[k]['signal'] for k in ('vits_cbar', 'deitb_inc', 'deitb_cbar', 'deits_inc', 'deits_cbar')]
    arms = ['px'] + [f'px+b{k}' for k in range(1, 13)]
    acc_rank = [sorted(arms, key=lf['trunk_accuracy'].get).index('px+b1') + 1 for lf in corr_lf]
    ent_rank = [sorted(arms, key=lf['trunk_mean_entropy'].get, reverse=True).index('px+b1') + 1 for lf in corr_lf]
    put('sg.corr.b1.leastacc.n', sum(r == 1 for r in acc_rank), 'd'); put('sg.corr.b1.leastconf.n', sum(r == 1 for r in ent_rank), 'd')
    put('sg.corr.b1.conf.rank.max', max(ent_rank), 'd'); put('sg.corr.n', len(corr_lf), 'd')
    # the pairs of the models-datasets plan
    for key, p in CA.items():
        if key.startswith('_') or not isinstance(p, dict) or 'held' not in p:
            continue
        cfg, h, sig = p['calib'], p['held']['episodic'], p['signal']; T = cfg['trunk']
        eta = p['eta']['episodic'] if 'eta' in p else None
        row(key, T, cfg['shallower'], cfg['deeper'], sig['trunk_spearman_confidence_accuracy'], h['neo'], h[f'tok+place_b{T}'], h[cfg['arm']],
            eta['eta+neo'] if eta else None, eta[f'tok+eta+place_b{T}+neo'] if eta else None, p['held']['clean'][cfg['arm']] - p['held']['clean']['neo'],
            eta_clean=p['eta']['clean']['eta+neo'] if eta else None)


def render(numbers):
    lines = ['% Generated by paper/make_numbers.py from results/paper_numbers.json; do not edit.',
             '\\providecommand{\\pn}[1]{\\ifcsname pn@#1\\endcsname\\csname pn@#1\\endcsname\\else\\errmessage{paper number #1 is not defined}\\fi}']
    minus = lambda t: ('\\ensuremath{-}' + t[1:]) if re.match(r'^-\d', t) else t   # a real minus sign in text and math mode
    lines += [f'\\expandafter\\def\\csname pn@{k}\\endcsname{{{minus(t)}}}' for k, t in numbers.items()]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    build()
    OUT.write_text(render(N))
    print(f'{len(N)} numbers written to {OUT}')
