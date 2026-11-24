"""Figures for the course report and the long paper, drawn from results/paper_numbers.json only (no recomputation):
the depth curve and the drift-by-depth map of the where-to-align study, the frontier, the gate, the continual stream and
the gain-against-drift scatter of the pixel-site stage.

Each figure is written at its final inserted width (the report includes it with that exact width), so every text
element is 8 pt, i.e. the 10 pt body size minus 2 pt. Vector PDF with embedded TrueType fonts + PNG preview.
Colours: neutral greys for hosts without TokAlign, one accent blue for every arm that contains our stages; the
ETA-hosted arms are distinguished by hatching (bars) or square markers (scatter), never by a new colour.
Usage: python3 paper/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / 'results' / 'paper_numbers.json').read_text())
OUT = Path(__file__).resolve().parent / 'figures'; OUT.mkdir(exist_ok=True)

# Times New Roman (TrueType) to match the ICML body font; registered explicitly because matplotlib's cache may miss
# user-installed fonts, and a TrueType face embeds cleanly as Type 42 (an OpenType-CFF fallback does not)
from matplotlib import font_manager
for f in (Path.home() / '.local/share/fonts/msttcore').glob('times*.ttf'):
    font_manager.fontManager.addfont(str(f))
FS = 8                       # the only font size used in every figure
COL, FULL = 3.25, 6.75       # ICML column and text widths (in)
C = {'accent': '#2a6fbf', 'accent_light': '#a9c8ea', 'host': '#7a7a7a', 'host_dark': '#4a4a4a', 'host_light': '#c4c4c4',
     'ink': '#111111', 'grid': '#e2e2e2', 'zero': '#555555'}
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif'], 'mathtext.fontset': 'stix',
    'font.size': FS, 'axes.titlesize': FS, 'axes.labelsize': FS, 'legend.fontsize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': C['ink'], 'axes.linewidth': 0.6,
    'xtick.color': C['ink'], 'ytick.color': C['ink'], 'text.color': C['ink'], 'axes.labelcolor': C['ink'],
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False, 'hatch.linewidth': 0.6})


def save(fig, name):
    # no bbox_inches='tight': the PDF keeps the figsize so that \includegraphics at that width does not rescale text
    fig.savefig(OUT / f'{name}.pdf'); fig.savefig(OUT / f'{name}.png', dpi=220); plt.close(fig)


def seg_name(key, short=False):
    cell = key.split('@')[0]
    if cell == 'clean':
        return 'clean'
    name, sev = cell.rsplit('-', 1)
    name = {'gaussian_noise': 'gauss.', 'shot_noise': 'shot', 'impulse_noise': 'impulse', 'defocus_blur': 'defocus',
            'glass_blur': 'glass', 'motion_blur': 'motion', 'zoom_blur': 'zoom', 'elastic_transform': 'elastic',
            'jpeg_compression': 'JPEG'}.get(name, name)
    if short:
        name = {'defocus': 'defoc.', 'contrast': 'contr.', 'brightness': 'bright.'}.get(name, name)
    return f'{name}\n{sev}'


# ---------- Figure 1 (introduction): the depth sweep on design cells, ViT-B/16: gain of the alignment at one site
T = D['two_site']; SITES = ('pixel',) + tuple(f'b{k}' for k in range(1, 13)); xs = np.arange(len(SITES))
GROUPS = {'noise': ('gaussian_noise', 'shot_noise', 'impulse_noise'), 'blur': ('defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'),
          'weather': ('snow', 'frost', 'fog', 'brightness'), 'digital': ('contrast', 'elastic_transform', 'pixelate', 'jpeg_compression')}
sw = T['sweep_vitb']
fam_gain = lambda s, co: np.mean([sw['frozen_per_cell'][s][f'{co}-{sev}'] - sw['neo_per_cell'][f'{co}-{sev}'] for sev in (3, 5)])
LF = {b: D['exit']['site_choice'][b]['label_free'] for b in ('inc', 'cbar')}
fig, (ax, axe) = plt.subplots(2, 1, figsize=(COL, 2.30), layout='constrained', sharex=True, gridspec_kw={'height_ratios': [1, 1.05]})  # drawn at the column width: 8 pt text stays 8 pt
ax.axhline(0, color=C['zero'], lw=0.6)
ax.plot(xs, [sw['frozen_gain'][s] for s in SITES], marker='o', ms=3.5, lw=1.0, color=C['accent'], label='frozen ViT with NEO')
ax.plot(xs, [sw['eta_gain'][s] for s in SITES], marker='s', ms=3.2, lw=1.0, color=C['host_dark'], ls='--', label='ETA with NEO')
ax.set_ylabel('mean gain (pt)'); ax.legend(loc='lower center', ncol=2, handlelength=1.6, columnspacing=1.2, borderaxespad=0.1)  # below the dip at block 1, where no curve runs
ax.grid(True, axis='y', color=C['grid'], lw=0.5); ax.set_axisbelow(True); ax.set_title('(a) one map at one placement: gain over no map, 30 cells', loc='left')
lo = min(min(sw['frozen_gain'].values()), min(sw['eta_gain'].values())); ax.set_ylim(min(-10, lo - 0.5), 6)
# (b) the calibration signal: mean prediction entropy of the pixel-plus-block arms (no labels), lowest at the most accurate block
arms = ['px'] + [f'px+b{k}' for k in range(1, 13)]
for b, col, ls, mk, lab in (('inc', C['accent'], '-', 'o', 'ImageNet-C'), ('cbar', C['host_dark'], '--', 's', r'ImageNet-$\overline{\mathrm{C}}$')):
    ent = [LF[b]['trunk_mean_entropy'][a] for a in arms]; acc = [LF[b]['trunk_accuracy'][a] for a in arms]
    axe.plot(xs, ent, marker=mk, ms=3.0, lw=1.0, color=col, ls=ls, label=lab)
    kmin, kbest = int(np.argmin(ent)), int(np.argmax(acc))
    axe.scatter([kmin], [ent[kmin]], marker='o', s=70, facecolors='none', edgecolors=col, linewidths=1.0, zorder=4)
    axe.scatter([kbest], [ent[kbest]], marker='*', s=55, color=col, zorder=5)
axe.scatter([], [], marker='o', s=40, facecolors='none', edgecolors=C['ink'], label='lowest entropy (the pick)')
axe.scatter([], [], marker='*', s=40, color=C['ink'], label='most accurate block')
axe.set_ylabel('mean entropy (nat)')
ylo, yhi = axe.get_ylim(); axe.set_ylim(ylo, yhi + 0.9 * (yhi - ylo))  # room above the curves for the legend, clear of the block-1 points
axe.legend(loc='upper center', ncol=2, handlelength=1.4, columnspacing=0.8, borderaxespad=0.1, handletextpad=0.3)
axe.grid(True, axis='y', color=C['grid'], lw=0.5); axe.set_axisbelow(True); axe.set_title('(b) TokAlign + LayerAlign after block $k$: mean entropy', loc='left')
axe.set_xticks(xs); axe.set_xticklabels(['pixels'] + [s[1:] for s in SITES[1:]]); axe.set_xlabel('placement: pixels, or patch tokens after block $k$')
save(fig, 'depth_curve')

# ---------- Figure S (supplement): the depth sweep by corruption family (frozen ViT with NEO)
styles = {'noise': (C['accent'], '-', 'o'), 'blur': (C['host_dark'], '-', 's'), 'weather': (C['accent_light'], '-', '^'), 'digital': (C['host'], '--', 'D')}
fig, axf = plt.subplots(figsize=(COL, 1.5), layout='constrained')
axf.axhline(0, color=C['zero'], lw=0.6)
for g, cos in GROUPS.items():
    col, ls, mk = styles[g]
    axf.plot(xs, [np.mean([fam_gain(s, co) for co in cos]) for s in SITES], marker=mk, ms=3.0, lw=1.0, color=col, ls=ls, label=g)
axf.set_ylabel('mean gain over NEO (pt)'); axf.legend(loc='lower right', ncol=2, handlelength=1.6, columnspacing=1.0, borderaxespad=0.1)
axf.grid(True, axis='y', color=C['grid'], lw=0.5); axf.set_axisbelow(True)
axf.set_xticks(xs); axf.set_xticklabels(['pixels'] + [s[1:] for s in SITES[1:]]); axf.set_xlabel('placement: pixels, or patch tokens after block $k$')
save(fig, 'depth_family')

# ---------- Figure 2 (results): where the shift appears: z of the spectral drift at every site, and the best site per corruption
dr = T['drift_vitb']; K = dr['k_batches']
CORR_ORDER = [co for g in GROUPS.values() for co in g]
Z = np.array([[np.mean([dr['z_spec_first_k'][f'{co}-{sev}'][s] for sev in (3, 5)]) for s in SITES] for co in CORR_ORDER])
best = [max(SITES, key=lambda s: fam_gain(s, co)) for co in CORR_ORDER]
fig, ax = plt.subplots(figsize=(COL, 2.1), layout='constrained')
im = ax.imshow(np.log10(np.clip(Z, 1, None)), cmap='Blues', aspect='auto', vmin=0, vmax=max(2.0, float(np.log10(Z.max()))))
ax.scatter([SITES.index(b) for b in best], np.arange(len(CORR_ORDER)), marker='o', s=14, facecolors='white', edgecolors=C['ink'], linewidths=0.6, zorder=3, label='best single placement')
ax.set_xticks(xs); ax.set_xticklabels(['px'] + [s[1:] for s in SITES[1:]]); ax.set_xlabel('placement: pixels, or patch tokens after block $k$')
ax.set_yticks(np.arange(len(CORR_ORDER))); ax.set_yticklabels([seg_name(co + '-5').split('\n')[0] for co in CORR_ORDER]); ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05); cb.set_label(f'drift z (first {K} batches), log$_{{10}}$'); cb.outline.set_visible(False)
ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0, handletextpad=0.3)
save(fig, 'drift_depth')

# ---------- Figure (supplement): why these two blocks (design cells, saved logits): the calibrated picks 4 and 11 (solid), the labelled search 5 and 9 (dashed)
SO = {b: D['exit']['site_choice'][b]['second_opinions'] for b in ('inc', 'cbar')}
cand = ['px'] + [f'px+b{k}' for k in range(1, 13) if k != 6]
xpos = {'px': 0, **{f'px+b{k}': k for k in range(1, 13)}}
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(FULL, 2.05), layout='constrained', gridspec_kw={'width_ratios': [1, 1, 1.05]})
for ax_, key, ylab, ttl in ((a1, 'agree_routed', 'agreement with block 6 (%)', '(a) how often a placement agrees with block 6'),
                            (a2, 'gain_routed_mixture', 'gain over the main path (pt)', '(b) main path plus one extra branch')):
    for b, col, ls, mk, lab in (('inc', C['accent'], '-', 'o', 'ImageNet-C'), ('cbar', C['host_dark'], '--', 's', r'ImageNet-$\overline{\mathrm{C}}$')):
        xs_ = [xpos[a] for a in cand[:6]] + [6] + [xpos[a] for a in cand[6:]]                      # a gap at block 6 (the trunk)
        ys_ = [SO[b]['per_site'][a][key] for a in cand[:6]] + [np.nan] + [SO[b]['per_site'][a][key] for a in cand[6:]]
        ax_.plot(xs_, ys_, marker=mk, ms=3.0, lw=1.0, color=col, ls=ls, label=lab)
    ax_.axvspan(5.6, 6.4, color=C['accent_light'], alpha=0.5, lw=0)
    ax_.text(6, ax_.get_ylim()[0], 'main', ha='center', va='bottom', color=C['accent'])
    for k in (4, 11):
        ax_.axvline(k, color=C['accent'], lw=0.6, ls=':')
    for k in (5, 9):
        ax_.axvline(k, color=C['host'], lw=0.6, ls='--')
    ax_.set_xticks(range(13)); ax_.set_xticklabels(['px'] + [str(k) for k in range(1, 13)])
    ax_.set_xlabel('extra branch after block (px: TokAlign alone)'); ax_.set_ylabel(ylab); ax_.set_title(ttl, loc='left')
    ax_.grid(True, axis='y', color=C['grid'], lw=0.5); ax_.set_axisbelow(True)
a2.legend(loc='upper left', handlelength=1.6, borderaxespad=0.1)
cand_c = ['px'] + [f'px+b{k}' for k in range(1, 13) if k != 6]
pr = SO['inc']['pairs']
rows_c, cols_c = cand_c[:-1], cand_c[1:]                                   # the shallower branch (px to 11) against the deeper (1 to 12)
M = np.full((len(rows_c), len(cols_c)), np.nan)
for i, a_ in enumerate(rows_c):
    for j, b_ in enumerate(cols_c):
        if i <= j:
            M[i, j] = pr.get(f'{a_}|{b_}', pr.get(f'{b_}|{a_}'))
im = a3.imshow(np.ma.masked_invalid(M), cmap='Blues', aspect='auto', vmin=min(0.0, np.nanmin(M)), vmax=np.nanmax(M), interpolation='nearest')
fig.colorbar(im, ax=a3, fraction=0.05, pad=0.02)
for (ka, kb), ec, ls, lw in (((5, 9), C['host'], '--', 1.0), ((4, 11), C['accent'], '-', 1.4)):
    i_, j_ = rows_c.index(f'px+b{ka}'), cols_c.index(f'px+b{kb}')
    a3.add_patch(Rectangle((j_ - 0.5, i_ - 0.5), 1, 1, fill=False, ec=ec, lw=lw, ls=ls))
lab = lambda d: d.replace('px+b', '')
a3.set_xticks(range(len(cols_c))); a3.set_xticklabels([lab(d) for d in cols_c]); a3.set_yticks(range(len(rows_c))); a3.set_yticklabels([lab(d) for d in rows_c])
a3.set_xlabel('deeper branch'); a3.set_ylabel('shallower branch')
a3.set_title('(c) two extra branches, ImageNet-C (pt)', loc='left')
save(fig, 'why_sites')

# ---- Figure 2 (method): (a) the calibration from real design data, (b) one batch of the calibrated stack; full text width
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
TOK, TOK_FILL = '#1f8a70', '#d2ece4'                          # TokAlign, the map at the pixels
LAY, LAY_FILL = C['accent'], '#dce8f6'                        # LayerAlign, the same map after a block
W, H = FULL, 1.68
LFI = D['exit']['site_choice']['inc']['label_free']
fig = plt.figure(figsize=(W, H))
def fx(x):                                                    # figure x (in) -> fraction
    return x / W
# ---------- (a) calibration
fig.text(fx(0.02), 1.60 / H, '(a) calibration: once, from a few hundred unlabeled images', ha='left', va='center', color=C['host_dark'])
fig.patches.append(FancyBboxPatch((fx(0.02), 0.62 / H), 0.74 / W, 0.56 / H, boxstyle='round,pad=0,rounding_size=0.006', transform=fig.transFigure, fc='#ececec', ec=C['host'], lw=0.8))
fig.text(fx(0.39), 0.90 / H, 'run the 13\nplacements\npx + block $k$', ha='center', va='center', linespacing=1.05)
fig.patches.append(FancyArrowPatch((fx(0.76), 0.90 / H), (fx(0.86), 0.90 / H), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=6, lw=0.7, color=C['host_dark'], shrinkA=0, shrinkB=0))
ks = list(range(1, 13))
a1 = fig.add_axes([fx(1.06), 0.40 / H, 0.98 / W, 0.96 / H])
ent = [LFI['trunk_mean_entropy'][f'px+b{k}'] for k in ks]; kt = ks[int(np.argmin(ent))]
a1.plot([0] + ks, [LFI['trunk_mean_entropy']['px']] + ent, marker='o', ms=2.6, lw=0.9, color=LAY)   # the 13 placements: px, then px + block k
a1.scatter([kt], [min(ent)], marker='o', s=60, facecolors='none', edgecolors=LAY, linewidths=1.0, zorder=4)
a1.annotate(f'main = {kt}', (kt, min(ent)), xytext=(0.30, 0.90), textcoords='axes fraction', ha='center', va='top', color=LAY,
            arrowprops=dict(arrowstyle='-', color=LAY, lw=0.6, shrinkA=0, shrinkB=5))
a1.set_title('mean entropy', loc='left', pad=2); a1.set_xticks([0, 6, 12]); a1.set_xticklabels(['px', '6', '12']); a1.set_yticks([]); a1.set_xlabel('placement', labelpad=1)
a1.grid(True, axis='y', color=C['grid'], lw=0.5)
fig.patches.append(FancyArrowPatch((fx(2.08), 0.90 / H), (fx(2.18), 0.90 / H), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=6, lw=0.7, color=C['host_dark'], shrinkA=0, shrinkB=0))
a2 = fig.add_axes([fx(2.36), 0.40 / H, 0.98 / W, 0.96 / H])
me = LFI['second_opinion_mixture_entropy']
others = [k for k in ks if k != kt]
M = np.full((len(others), len(others)), np.nan)
for i, ka in enumerate(others):
    for j, kb in enumerate(others):
        if ka < kb:
            M[i, j] = me[f'px+b{ka}|px+b{kb}']
pa, pb = sorted(int(x[4:]) for x in LFI['second_opinion_pick_pair'].split('|'))
a2.imshow(np.ma.masked_invalid(M), cmap='Blues_r', aspect='auto', interpolation='nearest')
a2.add_patch(Rectangle((others.index(pb) - 0.5, others.index(pa) - 0.5), 1, 1, fill=False, ec=LAY, lw=1.2, zorder=4))
a2.annotate(f'branches = {pa}, {pb}', (others.index(pb), others.index(pa)), xytext=(0.04, 0.10), textcoords='axes fraction', ha='left', va='center', color=LAY,
            arrowprops=dict(arrowstyle='-', color=LAY, lw=0.6, shrinkA=0, shrinkB=4))
a2.set_title('entropy of the 3-way average', loc='left', pad=2)
a2.set_xticks([others.index(k) for k in (1, 12)]); a2.set_xticklabels(['1', '12']); a2.set_yticks([others.index(k) for k in (1, 12)]); a2.set_yticklabels(['1', '12'])
a2.set_xlabel('deeper block', labelpad=1); a2.set_ylabel('shallower block', labelpad=1)
for a_ in (a1, a2):
    for sp in ('top', 'right'):
        a_.spines[sp].set_visible(False)
# ---------- (b) deployment: the previous column-width diagram, shifted right and widened by 10 %
XB, SX = 3.42, 1.0
YT, YM, YB, BH = 1.04, 0.64, 0.25, 0.23
def tx(x):
    return (XB + SX * x) / W
def fbox(x0, x1, y, text, fc='#ececec', ec=None):
    fig.patches.append(FancyBboxPatch((tx(x0), y / H), SX * (x1 - x0) / W, BH / H, boxstyle='round,pad=0,rounding_size=0.006', transform=fig.transFigure,
                                      fc=fc, ec=ec or C['host'], lw=0.8))
    fig.text((tx(x0) + tx(x1)) / 2, (y + BH / 2) / H, text, ha='center', va='center')
def fbar(x0, y, col):
    fig.patches.append(Rectangle((tx(x0), y / H), SX * 0.05 / W, BH / H, transform=fig.transFigure, fc=col, ec='none'))
def farrow(x0, y0, x1, y1, ls='-', col=None):
    fig.patches.append(FancyArrowPatch((tx(x0), y0 / H), (tx(x1), y1 / H), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=6,
                                       lw=0.7, color=col or C['host_dark'], ls=ls, shrinkA=0, shrinkB=0))
fig.text(tx(0.01), 1.60 / H, '(b) deployment: every batch', ha='left', va='center', color=C['host_dark'])
fig.text(tx(0.01), (YT + BH / 2) / H, 'main\npath', ha='left', va='center')
fig.text(tx(0.01), (YM + BH / 2) / H, 'branch,\nblock 11', ha='left', va='center', linespacing=1.0)
fig.text(tx(0.01), (YB + BH / 2) / H, 'branch,\nblock 4', ha='left', va='center', linespacing=1.0)
NEO0, NEO1, HX0, HX1, MX0, MX1 = 2.20, 2.48, 2.54, 2.88, 2.54, 3.12
fbox(0.40, 0.60, YT, 'gate', fc=LAY_FILL, ec=LAY); farrow(0.60, YT + BH / 2, 0.66, YT + BH / 2)
fbar(0.66, YT, TOK); fbox(0.73, 0.92, YT, 'px'); farrow(0.92, YT + BH / 2, 0.98, YT + BH / 2)
fbox(0.98, 1.27, YT, '1–4'); farrow(1.27, YT + BH / 2, 1.33, YT + BH / 2)
fbox(1.33, 1.55, YT, '5–6'); fbar(1.56, YT, LAY); fbox(1.63, 2.14, YT, '7–12'); farrow(2.14, YT + BH / 2, NEO0, YT + BH / 2)
fbox(NEO0, NEO1, YT, 'NEO', fc='white'); farrow(NEO1, YT + BH / 2, HX0, YT + BH / 2)
fbox(HX0, HX1, YT, r'$H>\theta$?', fc=LAY_FILL, ec=LAY)
farrow(HX1, YT + BH / 2, 2.95, YT + BH / 2, col=LAY); fig.text(tx(2.97), (YT + BH / 2) / H, 'output', ha='left', va='center')
fig.text(tx(HX1 + 0.05), (YT + BH / 2 + 0.05) / H, 'no', ha='left', va='bottom', color=LAY)
farrow(1.50, YT, 1.63, YM + BH / 2, ls='--', col=LAY)
fbox(1.63, 1.90, YM, '7–11'); fbar(1.91, YM, LAY); fbox(1.98, 2.14, YM, '12'); farrow(2.14, YM + BH / 2, NEO0, YM + BH / 2); fbox(NEO0, NEO1, YM, 'NEO', fc='white')
farrow(1.24, YT, 1.305, YB + BH, ls='--', col=LAY)
fbar(1.28, YB, LAY); fbox(1.35, 2.14, YB, '5–12'); farrow(2.14, YB + BH / 2, NEO0, YB + BH / 2); fbox(NEO0, NEO1, YB, 'NEO', fc='white')
fig.patches.append(FancyBboxPatch((tx(MX0), YB / H), SX * (MX1 - MX0) / W, (YM + BH - YB) / H, boxstyle='round,pad=0,rounding_size=0.006', transform=fig.transFigure,
                                  fc=LAY_FILL, ec=LAY, lw=0.8))
fig.text((tx(MX0) + tx(MX1)) / 2, (YB + YM + BH) / 2 / H, 'mix\n' + r'$w_a{\propto}e^{-H_a}$', ha='center', va='center', linespacing=1.1)
farrow(NEO1, YM + BH / 2, MX0, YM + BH / 2); farrow(NEO1, YB + BH / 2, MX0, YB + BH / 2)
xy = (HX0 + HX1) / 2
farrow(xy, YT, xy, YM + BH, ls='--', col=LAY); fig.text(tx(xy + 0.04), (YT + YM + BH) / 2 / H, 'yes', ha='left', va='center', color=LAY)
farrow(3.04, YM + BH, 3.04, YT + BH / 2 - 0.08, col=LAY)
fig.patches.append(FancyBboxPatch((tx(0.62), 0.20 / H), SX * (3.17 - 0.62) / W, (YM + BH + 0.08 - 0.20) / H, boxstyle='round,pad=0,rounding_size=0.008', transform=fig.transFigure,
                                  fc='none', ec=LAY, lw=0.7, ls='--'))
fig.text(tx(0.66), (YB + BH / 2) / H, r'only if $H>\theta$', ha='left', va='center', color=LAY)
for x, col, t in ((0.62, TOK, 'TokAlign (pixels)'), (1.60, LAY, 'LayerAlign (after a block)')):
    fig.patches.append(Rectangle((tx(x), 0.03 / H), SX * 0.05 / W, 0.09 / H, transform=fig.transFigure, fc=col, ec='none'))
    fig.text(tx(x + 0.08), 0.075 / H, t, ha='left', va='center', color=C['host_dark'])
save(fig, 'overview')


# ---------- Figure (supplement, main-text subset): the calibration signal across models and shifts. Per pair: accuracy of the
# pixel-plus-block versions (labels, only to check) and their mean prediction entropy (the signal), the pick circled, the best starred.
def signal_panels():
    """One panel per model-shift pair, rows = models, columns = shifts (C, C-bar, Sketch, R); None where the pair was not run."""
    LFV = D['exit']['site_choice']; RL = D['rule']; CA = D.get('ca', {})
    fixed = {'vitb_inc': LFV['inc']['label_free'], 'vits_inc': RL['vits']['pick_t6'], 'vitb_cbar': LFV['cbar']['label_free']}
    names = {'vitb': 'ViT-B', 'vits': 'ViT-S', 'deitb': 'DeiT-B', 'deits': 'DeiT-S', 'inc': 'C', 'cbar': '$\\overline{\\mathrm{C}}$', 'sketch': 'S', 'r': 'R'}
    grid = []
    for m in ('vitb', 'vits', 'deitb', 'deits'):
        for ds in ('inc', 'cbar', 'sketch', 'r'):
            key = f'{m}_{ds}'
            lf = fixed.get(key) or (CA[key]['signal'] if key in CA else None)
            grid.append((f'{names[m]}, {names[ds]}', lf if lf is not None and 'trunk_mean_entropy' in lf else None))
    return grid
PANELS = signal_panels()
arms = ['px'] + [f'px+b{k}' for k in range(1, 13)]
ncol = 4; nrow = 4
fig, axes = plt.subplots(nrow, ncol, figsize=(FULL, 1.35 * nrow), layout='constrained')
for ax, (title, lf) in zip(axes.flat, PANELS):
    if lf is None:
        ax.set_visible(False); continue
    acc = [lf['trunk_accuracy'][a] for a in arms]; ent = [lf['trunk_mean_entropy'][a] for a in arms]
    kpick, kbest = arms.index(lf['trunk_pick_by_entropy']), arms.index(lf['trunk_best_by_accuracy'])
    ax.plot(xs, acc, marker='o', ms=2.2, lw=0.9, color=C['ink']); ax.scatter([kbest], [acc[kbest]], marker='*', s=55, color=C['ink'], zorder=5)
    ax2 = ax.twinx(); ax2.plot(xs, ent, marker='s', ms=2.0, lw=0.9, color=C['accent'], ls='--'); ax2.scatter([kpick], [ent[kpick]], marker='o', s=60, facecolors='none', edgecolors=C['accent'], linewidths=1.0, zorder=5)
    ax2.tick_params(axis='y', colors=C['accent'], length=2); ax2.spines['right'].set_visible(True); ax2.spines['right'].set_color(C['accent'])
    ax.set_title(f"{title}, $\\rho$ {lf['trunk_spearman_confidence_accuracy']:+.2f}", loc='left', pad=2)
    ax.locator_params(axis='y', nbins=4); ax2.locator_params(axis='y', nbins=3)
    ax.set_xticks([0, 3, 6, 9, 12]); ax.set_xticklabels(['px', '3', '6', '9', '12']); ax.tick_params(length=2)
    ax.grid(True, axis='y', color=C['grid'], lw=0.5); ax.set_axisbelow(True)
for ax in axes[:, 0] if nrow > 1 else [axes[0]]:
    ax.set_ylabel('top-1 (%)')
for ax in axes[-1] if nrow > 1 else axes:
    ax.set_xlabel('TokAlign + LayerAlign after block')
axes.flat[0].plot([], [], marker='o', ms=2.2, lw=0.9, color=C['ink'], label='accuracy (labels, to check)'); axes.flat[0].plot([], [], marker='s', ms=2.0, lw=0.9, color=C['accent'], ls='--', label='mean entropy (no labels)')
axes.flat[0].scatter([], [], marker='*', s=40, color=C['ink'], label='most accurate'); axes.flat[0].scatter([], [], marker='o', s=40, facecolors='none', edgecolors=C['accent'], label='the pick')
fig.legend(loc='outside lower center', ncol=4, handlelength=1.6, columnspacing=1.2, frameon=False)
save(fig, 'signal_grid')

# ---------- Figure (supplement): how many unlabeled images the calibration needs (ViT-B/16, from the saved design predictions)
SS = D['rule']['sample_size']
fig, (b1, b2) = plt.subplots(1, 2, figsize=(FULL, 1.9), layout='constrained')
for b, col, mk, lab in (('inc', C['accent'], 'o', 'ImageNet-C'), ('cbar', C['host_dark'], 's', r'ImageNet-$\overline{\mathrm{C}}$')):
    pooled = SS[b]['pooled_first_n_per_cell']; ns = sorted(int(n) for n in pooled)
    b1.plot(ns, [pooled[str(n)]['gain_over_pair'] for n in ns], marker=mk, ms=3, lw=1.0, color=col, label=lab)
    b1.axhline(SS[b]['full_rule']['gain_over_pair'], color=col, lw=0.6, ls=':')
    prev = None
    for n in ns:                                                 # the blocks picked, written where the pick changes
        d = pooled[str(n)]; pick = '{}/{},{}'.format(*(x[4:] for x in (d['trunk'], d['shallower'], d['deeper'])))
        if pick != prev:
            below = b == 'cbar' and n == 640                        # the one label that would sit on the curve if written above its point
            b1.annotate(pick, (n, d['gain_over_pair']), xytext=(0, -9 if below else 5), textcoords='offset points', ha='center', va='top' if below else 'bottom', fontsize=FS, color=col)
        prev = pick
    per = SS[b]['per_cell_first_k_batches']; Ks = sorted(int(k) for k in per)
    b2.plot([per[str(k)]['images_per_cell'] for k in Ks], [per[str(k)]['gain_over_pair'] for k in Ks], marker=mk, ms=3, lw=1.0, color=col, label=lab)
    b2.axhline(SS[b]['full_rule']['gain_over_pair'], color=col, lw=0.6, ls=':')
b1.set_xscale('log'); b1.set_xticks([64, 128, 192, 320, 640, 1280, 2000]); b1.set_xticklabels(['64', '128', '192', '320', '640', '1280', '2000']); b1.minorticks_off()
b1.set_xlabel('calibration images per cell, one choice per benchmark'); b1.set_ylabel('gain over the main path (pt)'); b1.set_title('(a) one calibration for a whole benchmark', loc='left')
b2.set_xscale('log'); b2.set_xticks([64, 128, 192, 320, 640]); b2.set_xticklabels(['64', '128', '192', '320', '640']); b2.minorticks_off()
b2.set_xlabel('calibration images per shift (the first batches of 64)'); b2.set_title('(b) one calibration per shift', loc='left')
for b in (b1, b2):
    b.grid(True, axis='y', color=C['grid'], lw=0.5); b.set_axisbelow(True)
ylo, yhi = b1.get_ylim(); b1.set_ylim(ylo, yhi + 0.3 * (yhi - ylo))   # room above the curves for the legend
b1.legend(loc='upper left', ncol=2, handlelength=1.6, columnspacing=1.0, borderaxespad=0.1)
save(fig, 'calibration_size')
# ---------- Figure (supplement): accuracy against compute (forward passes, paper/compute_cost.py) and against time per batch
# on the six timing cells (rule/timing and rule/timing_others): NEO, TokAlign, the main path (ours, base), the cascade (ours,
# full), the random-image control, ETA, SAR, PEA, PEA with views, FOA.
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_cost import VITB_GMACS, forward_passes
TM, TO = D['rule']['timing'], D['rule']['timing_others']; RL4 = D['rule']['held4']; EX4 = D['exit']['held4']['inc']
FP = forward_passes(RL4['inc']['routed']['rule_t6_b4b11_gated'], RL4['inc_abl']['routed']['rule_t6_b4_gated'], RL4['inc_abl']['routed']['rule_t6_b11_gated'],
                    D['rule']['random']['inc']['routed']['rule_t6_b4b11_random_gated'], np.mean([f for c, f in EX4['routed']['exit_b5b9_gated'].items() if c != 'clean']))
pts = [  # (label, compute key, timing entry, colour, marker, filled, label placement for (a) and (b): ('offset', dx, dy) in points or ('data', x, y) with a leader line)
    ('NEO', 'neo', TM['neo'], C['host_dark'], 'o', True, ('offset', 0, -8), ('offset', 0, -8)),
    ('TokAlign', 'tok', TM['tok_k16_fp32_stats32'], C['accent'], 'o', True, ('data', 1.3, 62.6), ('data', 1.3, 62.6)),
    ('main path', 'pair', TM['pair_k16_fp32_stats32'], C['accent'], 'o', True, ('data', 1.3, 64.6), ('data', 1.3, 64.6)),
    ('cascade', 'b4b11', TM['rule_t6_b4b11_k16_fp32_stats32'], C['accent'], 'o', True, ('data', 1.9, 68.4), ('data', 2.0, 68.4)),
    ('random images', 'random', TM['rule_t6_b4b11_random_k16_fp32_stats32'], C['accent'], 'o', False, ('data', 2.0, 66.4), ('data', 2.1, 66.4)),
    ('ETA with NEO', 'eta', TM['eta'], C['host_dark'], 's', True, ('offset', 0, -8), ('offset', 0, -8)),
    ('SAR with NEO', 'sar', TM['sar'], C['host_dark'], 's', True, ('offset', 7, -3), ('offset', 7, -3)),
    ('PEA', 'pea', TO['pea'], C['host_dark'], 'o', True, ('offset', 0, 6), ('offset', 0, 6)),
    ('PEA with views', 'peaviews', TO['pea_aug'], C['host_dark'], 'o', True, ('offset', 0, -8), ('offset', 0, -8)),
    ('FOA', 'foa', TO['foa'], C['host_dark'], 'o', True, ('offset', 0, 6), ('offset', 0, 6)),
]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.4), layout='constrained', sharey=True)
for ax, panel in ((a1, 0), (a2, 1)):
    xs_, ys_ = [], []
    for lab, key, t, col, mk, filled, place_a, place_b in pts:
        x = FP[key] * VITB_GMACS if panel == 0 else t['x_neo']; y = t['acc_6cells']; place = place_a if panel == 0 else place_b
        ax.scatter([x], [y], marker=mk, s=28, facecolors=col if filled else 'white', edgecolors=col, linewidths=1.0, zorder=3)
        if place[0] == 'offset':
            dx, dy = place[1:]
            ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords='offset points', ha='center' if dx == 0 else ('left' if dx > 0 else 'right'),
                        va='bottom' if dy > 0 else 'top', color=col)
        else:
            ax.annotate(lab, (x, y), xytext=(place[1] * (VITB_GMACS if panel == 0 else 1), place[2]), textcoords='data', ha='left', va='center', color=col,
                        arrowprops=dict(arrowstyle='-', color=C['host_light'], lw=0.6, shrinkA=0, shrinkB=3))
        xs_.append(x); ys_.append(y)
    ax.set_xscale('log'); ax.minorticks_off()
    if panel == 0:
        ax.set_xlim(0.85 * VITB_GMACS, 45 * VITB_GMACS); ax.set_xticks([20, 30, 50, 100, 200, 500]); ax.set_xticklabels(['20', '30', '50', '100', '200', '500'])
    else:
        ax.set_xlim(0.85, 45); ax.set_xticks([1, 2, 3, 5, 10, 30]); ax.set_xticklabels(['1', '2', '3', '5', '10', '30'])
    ax.grid(True, color=C['grid'], lw=0.5); ax.set_axisbelow(True)
    ax.set_ylim(min(ys_) - 1.5, max(ys_) + 2.5)
a1.set_xlabel('GMACs per image (log scale)'); a1.set_ylabel('top-1 (%) on the six cells')
a2.set_xlabel('time per batch relative to NEO (log scale)')
a1.set_title('(a) accuracy against compute', loc='left'); a2.set_title('(b) accuracy against wall-clock time', loc='left')
save(fig, 'frontier')
print('figures written:', sorted(p.name for p in OUT.glob('*.pdf')))
