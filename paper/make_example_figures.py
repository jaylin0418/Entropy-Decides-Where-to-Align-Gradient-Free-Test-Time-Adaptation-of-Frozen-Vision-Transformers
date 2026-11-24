"""Method-overview and example figures of the course report (supplement), recomputed model-free on the CPU from real
design images, the saved source patch statistics and the committed scores. No model weights are loaded and no forward
pass is run: the pixel-site map (TokAlign) and the spectral gate are linear algebra on the pixel patches, so the exact
map the episodic run applied is reproduced here with the repository's own stage code (tokalign/stages.py, float64).

Selection rules (fixed, no hand-picking):
  cells   one corruption per behaviour of the depth sweep: gaussian_noise-5 (the pixel site helps, block 6 hurts),
          zoom_blur-5 (block 6 helps, the pixel site hurts), contrast-5 (both help); the framework figure uses the
          Gaussian-noise image
  image   the first image of batch 16 of the design split (index 960 in split order), so the running statistics hold
          16 batches = 1,024 images, as in the episodic run up to and including that batch
  patches the patch of the clean image with the lowest pixel standard deviation (flat) and the highest (textured)
The network rows of the framework figure are schematic; its inset is the entropy of the trunk's (pair's) predictions on the
2,000 design images of the cell, from the saved logits of the design run, with the routed share of the second-opinion run.
Displayed pixels are clipped to [0, 1] for display only. The drift and gain panels read results/paper_numbers.json
(two_site/drift_vitb and two_site/held3_vitb), nothing is recomputed there.

Usage: python paper/make_example_figures.py --imagenet-val <val> --imagenet-c <imagenet-c> --cache-dir cache --stats stats/vitb.pt --logits runs/logits
Outputs paper/figures/framework.{pdf,png} and paper/figures/example_numbers.json (the gate scores
and sample counts printed inside the figures).
"""
from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokalign import data, models, stages  # noqa: E402

OUT = ROOT / 'paper' / 'figures'
for f in (Path.home() / '.local/share/fonts/msttcore').glob('times*.ttf'):
    font_manager.fontManager.addfont(str(f))
FS = 8
FULL = 6.75
C = {'accent': '#2a6fbf', 'accent_fill': '#dce8f6', 'host': '#7a7a7a', 'host_fill': '#ececec', 'host_dark': '#4a4a4a',
     'ink': '#111111', 'grid': '#e2e2e2', 'flat': '#e8a33d', 'tex': '#7a3fb0', 'tok': '#1f8a70'}  # tok: TokAlign, as in Figure 2
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif'], 'mathtext.fontset': 'stix',
    'font.size': FS, 'axes.titlesize': FS, 'axes.labelsize': FS, 'legend.fontsize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': C['ink'], 'axes.linewidth': 0.6,
    'xtick.color': C['ink'], 'ytick.color': C['ink'], 'text.color': C['ink'], 'axes.labelcolor': C['ink'],
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False})
ROWS = [('gaussian_noise-5', 'Gaussian noise 5'), ('zoom_blur-5', 'zoom blur 5'), ('contrast-5', 'contrast 5')]
THETA = 1.5                          # the routing threshold of the second-opinion stack (nat)
BATCH, IMAGE_BATCH = 64, 16          # the shown image is the first of batch 16 (split order)


def show(x):  # normalized (3, H, W) tensor -> displayable (H, W, 3) in [0, 1], clipped for display only
    return np.clip(x.permute(1, 2, 0).numpy() * 0.5 + 0.5, 0, 1)


def replay(cell, imgs, src):
    """The pixel-site stage and the gate on the design cell, batches 1..IMAGE_BATCH in the order of the episodic run."""
    cfg = stages.StageConfig()
    tok, gate = stages.TokAlign(src, cfg, 'cpu'), stages.SpectralGate(src, cfg, 'cpu')
    for b in range(IMAGE_BATCH):
        x = models.to_input(np.asarray(imgs[b * BATCH:(b + 1) * BATCH]), 'cpu')
        p = stages.unfold(x)
        tok.update(p)
    on = gate(p)                                           # the gate reads the batch that holds the image
    y = tok.apply(x, p)
    n, n0 = tok.run.n, cfg.n0_tok_images * 196
    S = (n * tok.run.cov().double() + n0 * tok.S_s) / (n + n0)
    S = (1 - cfg.lam) * S + cfg.lam * tok.S_s
    return {'x': x[0], 'y': y[0], 'patches': p[:196], 'patches_out': stages.unfold(y)[:196], 'cov_t': tok.run.cov().numpy(),
            'cov_s': tok.S_s.numpy(), 'cov_hat': S.numpy(), 'n': int(n), 'gate': gate.log[-1], 'on': bool(on), 'cell': cell}


def ax_at(fig, x, y, w, h, H, **kw):  # inches from the bottom-left corner
    return fig.add_axes([x / FULL, y / H, w / FULL, h / H], **kw)


def img_ax(ax, im, title=None, color=C['host'], lw=0.4):
    ax.imshow(im, interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(lw); s.set_color(color)
    if title:
        ax.set_title(title, pad=2)


def outline(ax, p, col, lw=1.0):
    i, j = divmod(p, 14); ax.add_patch(Rectangle((j * 16 - 0.5, i * 16 - 0.5), 16, 16, fill=False, ec=col, lw=lw))


def box(fig, x, y, w, h, H, text, ours=False, dashed=False):
    fig.patches.append(FancyBboxPatch((x / FULL, y / H), w / FULL, h / H, boxstyle='round,pad=0,rounding_size=0.006',
                                      transform=fig.transFigure, fc=C['accent_fill'] if ours else C['host_fill'],
                                      ec=C['accent'] if ours else C['host'], lw=0.8, ls='--' if dashed else '-'))
    fig.text((x + w / 2) / FULL, (y + h / 2) / H, text, ha='center', va='center', color=C['ink'], linespacing=1.1)


def arrow(fig, x0, y0, x1, y1, H, col=None, ls='-'):
    fig.patches.append(FancyArrowPatch((x0 / FULL, y0 / H), (x1 / FULL, y1 / H), transform=fig.transFigure, arrowstyle='-|>',
                                       mutation_scale=7, lw=0.8, color=col or C['host'], ls=ls))


def strip(ax_, v, title=None):  # one 768-d patch vector as a grey-level strip
    ax_.imshow(np.clip(v, -1, 1)[None, :], cmap='gray', vmin=-1, vmax=1, aspect='auto', interpolation='nearest')
    ax_.set_xticks([]); ax_.set_yticks([])
    for sp in ax_.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color(C['host'])
    if title:
        ax_.set_title(title, pad=2)


def patch_pick(clean):  # flat and textured patch of the clean image, by pixel standard deviation
    sd = stages.unfold(clean[None])[:196].std(1).numpy()
    return int(sd.argmin()), int(sd.argmax())


def half_up(v):  # one decimal, half up, with its sign: the rounding of the tables (paper/make_numbers.py fmt 'up')
    q = Decimal(repr(float(v))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
    return f'{q:+}'


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf'); fig.savefig(OUT / f'{name}.png', dpi=220); plt.close(fig)


def framework(r, clean, pick, trunk_entropy, theta, routed_run):
    """Figure S1: one dev batch through the calibrated cascade. The top row (pixel site and gate) is recomputed
    from the real batch; the network rows are schematic; the inset is the real entropy of the trunk's predictions on the
    2,000 images of the cell (saved logits of the design run) with the threshold and the routed share."""
    H = 4.30
    fig = plt.figure(figsize=(FULL, H))
    y0 = 2.85
    flat, tex = pick
    ax = ax_at(fig, 0.12, y0, 1.0, 1.0, H); img_ax(ax, show(r['x']), 'input (Gaussian noise 5)')
    for t in range(1, 14):
        ax.axhline(t * 16 - 0.5, color='white', lw=0.25, alpha=0.7); ax.axvline(t * 16 - 0.5, color='white', lw=0.25, alpha=0.7)
    outline(ax, flat, C['flat']); outline(ax, tex, C['tex'])
    fig.text(0.62 / FULL, (y0 - 0.04) / H, r'196 patches of $16{\times}16{\times}3$', ha='center', va='top')
    px, pxo = r['patches'].numpy(), r['patches_out'].numpy()
    for q, (p, col) in enumerate(((flat, C['flat']), (tex, C['tex']))):
        yy = y0 + 0.58 - q * 0.56
        a_ = ax_at(fig, 1.27, yy, 0.38, 0.38, H); img_ax(a_, np.clip(px[p].reshape(3, 16, 16).transpose(1, 2, 0) * 0.5 + 0.5, 0, 1), color=col, lw=1.0)
        strip(ax_at(fig, 1.73, yy + 0.12, 0.58, 0.14, H), px[p], '$x$ (768-d)' if q == 0 else None)
        strip(ax_at(fig, 4.70, yy + 0.12, 0.58, 0.14, H), pxo[p], "$x'$ (768-d)" if q == 0 else None)
    arrow(fig, 1.14, y0 + 0.50, 1.25, y0 + 0.50, H)
    vmax = float(np.abs(r['cov_s'][:64, :64]).max())
    for j, (A_, t) in enumerate(((r['cov_t'], r'online $\Sigma_t$'), (r['cov_s'], r'source $\Sigma_s$'), (r['cov_hat'], r'shrunk $\hat\Sigma$'))):
        a_ = ax_at(fig, 2.52 + j * 0.62, y0 + 0.50, 0.52, 0.52, H)
        a_.imshow(A_[:64, :64], cmap='Greys', vmin=0, vmax=vmax, interpolation='nearest')
        a_.set_xticks([]); a_.set_yticks([]); a_.set_title(t, pad=2)
        for sp in a_.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color(C['host'])
    fig.text(3.38 / FULL, (y0 + 0.40) / H, r'$\hat\Sigma=(1-\lambda)\,\frac{n\Sigma_t+n_0\Sigma_s}{n+n_0}+\lambda\Sigma_s$', ha='center', va='top')
    fig.text(3.38 / FULL, (y0 + 0.11) / H, f'$n$ = {r["n"]:,} patches (batches 1 to {IMAGE_BATCH})', ha='center', va='top')
    fig.text(3.38 / FULL, (y0 + 1.17) / H, 'running statistics, shrunk to the source', ha='center', va='bottom', color=C['tok'])
    arrow(fig, 2.34, y0 + 0.50, 2.50, y0 + 0.50, H, C['tok'])
    arrow(fig, 4.40, y0 + 0.76, 4.66, y0 + 0.76, H, C['tok'])
    fig.text(4.70 / FULL, (y0 + 1.17) / H, r"$x'=\Sigma_s^{1/2}\hat\Sigma^{-1/2}(x-m_t)+\mu_s$", ha='left', va='bottom', color=C['tok'])
    a_ = ax_at(fig, 5.52, y0, 1.0, 1.0, H); img_ax(a_, show(r['y']), 'after TokAlign')
    outline(a_, flat, C['flat']); outline(a_, tex, C['tex'])
    arrow(fig, 5.36, y0 + 0.50, 5.50, y0 + 0.50, H, C['tok'])
    # the gate: a model-free side branch on the same batch of patches
    spec, shift, on = r['gate']
    yg = 2.28
    fig.patches.append(FancyBboxPatch((1.27 / FULL, yg / H), 3.55 / FULL, 0.36 / H, boxstyle='round,pad=0,rounding_size=0.006',
                                      transform=fig.transFigure, fc='white', ec=C['accent'], lw=0.8, ls='--'))
    fig.text(3.045 / FULL, (yg + 0.18) / H, 'spectral gate: DCT power of the same patches, no model call\n'
             f'$g_{{\\mathrm{{spec}}}}$ = {spec:.2f} > 0.4, $g_{{\\mathrm{{mean}}}}$ = {shift:.2f} < 0.3: gate {"on" if on else "off"} for this batch',
             ha='center', va='center', color=C['accent'], linespacing=1.15)
    fig.text(5.00 / FULL, (yg + 0.18) / H, 'gray: frozen network or NEO\ngreen: TokAlign, blue: LayerAlign,\ngate and extra branches (dashed)', ha='left', va='center', color=C['ink'], linespacing=1.15)
    # the trunk: the pair (schematic: no token values are computed here)
    yt, bh = 1.55, 0.38
    fig.patches.append(FancyArrowPatch(path=MPath([(6.35 / FULL, y0 / H), (6.35 / FULL, 2.12 / H), (0.38 / FULL, 2.12 / H), (0.38 / FULL, (yt + bh) / H)]),
                                       transform=fig.transFigure, arrowstyle='-|>', mutation_scale=7, lw=0.8, color=C['accent']))
    box(fig, 0.05, yt, 0.66, bh, H, 'patch\nembedding')
    box(fig, 0.81, yt, 0.60, bh, H, 'blocks\n1 to 4')
    box(fig, 1.51, yt, 0.46, bh, H, 'blocks\n5, 6')
    box(fig, 2.07, yt, 0.82, bh, H, 'LayerAlign\nafter block 6', ours=True)
    box(fig, 2.99, yt, 0.62, bh, H, 'blocks\n7 to 12')
    box(fig, 3.71, yt, 0.62, bh, H, 'NEO, head')
    box(fig, 4.43, yt, 0.66, bh, H, 'entropy $H$\n$H>\\theta$ ?', ours=True)
    for x0, x1 in ((0.71, 0.81), (1.41, 1.51), (1.97, 2.07), (2.89, 2.99), (3.61, 3.71), (4.33, 4.43)):
        arrow(fig, x0, yt + bh / 2, x1, yt + bh / 2, H)
    arrow(fig, 5.09, yt + bh / 2, 5.30, yt + bh / 2, H)
    fig.text(5.33 / FULL, (yt + bh / 2) / H, 'no: output\nthe main path', ha='left', va='center', linespacing=1.1)
    fig.patches.append(FancyArrowPatch((2.48 / FULL, yg / H), (2.48 / FULL, (yt + bh) / H), transform=fig.transFigure,
                                       arrowstyle='-|>', mutation_scale=6, lw=0.7, color=C['accent'], ls='--'))
    fig.text(2.53 / FULL, (yt + bh + 0.12) / H, '$g$', ha='left', va='center', color=C['accent'])
    # the extra branches, run only for the images with H > theta
    y9, y5, bb = 0.88, 0.22, 0.36
    box(fig, 1.55, y9, 0.56, bb, H, 'blocks\n7 to 11')
    box(fig, 2.21, y9, 0.82, bb, H, 'LayerAlign\nafter block 11', ours=True)
    box(fig, 3.13, y9, 0.62, bb, H, 'block 12')
    box(fig, 3.85, y9, 0.48, bb, H, 'NEO,\nhead')
    box(fig, 0.81, y5, 0.82, bb, H, 'LayerAlign\nafter block 4', ours=True)
    box(fig, 1.73, y5, 0.62, bb, H, 'blocks\n5 to 12')
    box(fig, 2.45, y5, 0.48, bb, H, 'NEO,\nhead')
    for x0, x1, yy in ((2.11, 2.21, y9), (3.03, 3.13, y9), (3.75, 3.85, y9), (1.63, 1.73, y5), (2.35, 2.45, y5)):
        arrow(fig, x0, yy + bb / 2, x1, yy + bb / 2, H)
    for xs, ys, ye, lab in ((1.74, yt, y9 + bb, 'unmapped tokens\nafter block 6'), (1.11, yt, y5 + bb, 'saved tokens\nafter block 4')):
        fig.patches.append(FancyArrowPatch((xs / FULL, ys / H), (xs / FULL, ye / H), transform=fig.transFigure,
                                           arrowstyle='-|>', mutation_scale=6, lw=0.7, color=C['accent'], ls='--'))
    fig.text(1.06 / FULL, 1.05 / H, 'saved tokens\nafter block 4', ha='right', va='center', color=C['accent'], linespacing=1.05)
    fig.text(1.79 / FULL, 1.40 / H, 'unmapped tokens after block 6', ha='left', va='center', color=C['accent'])
    # the confidence-weighted average
    xm, wm = 4.43, 0.88
    box(fig, xm, y5, wm, y9 + bb - y5, H, 'confidence-\nweighted average\n$w_a\\propto e^{-H_a}$\n(main path, block 11,\nblock 4)', ours=True)
    arrow(fig, 4.33, y9 + bb / 2, xm, y9 + bb / 2, H)
    arrow(fig, 2.93, y5 + bb / 2, xm, y5 + bb / 2, H)
    fig.patches.append(FancyArrowPatch((4.76 / FULL, yt / H), (4.76 / FULL, (y9 + bb) / H), transform=fig.transFigure,
                                       arrowstyle='-|>', mutation_scale=6, lw=0.7, color=C['accent'], ls='--'))
    fig.text(4.81 / FULL, ((yt + y9 + bb) / 2) / H, 'yes: run the\nextra branches', ha='left', va='center', color=C['accent'], linespacing=1.05)
    arrow(fig, xm + wm, (y5 + y9 + bb) / 2, 5.48, (y5 + y9 + bb) / 2, H)
    fig.text(5.50 / FULL, ((y5 + y9 + bb) / 2) / H, 'output', ha='left', va='center')
    # inset: the real trunk entropy on the 2,000 images of this cell, the threshold and the routed share
    ai = ax_at(fig, 5.98, 0.30, 0.70, 0.86, H)
    bins = np.linspace(0, max(5.0, float(trunk_entropy.max())), 26)
    ai.hist(trunk_entropy[trunk_entropy <= theta], bins=bins, color=C['host'], edgecolor='none')
    ai.hist(trunk_entropy[trunk_entropy > theta], bins=bins, color=C['accent'], edgecolor='none')
    ai.axvline(theta, color=C['ink'], lw=0.7, ls='--')
    ai.set_yticks([]); ai.set_xticks([0, theta, 4]); ai.set_xticklabels(['0', r'$\theta$', '4'])
    ai.spines['left'].set_visible(False)
    ai.set_title(f'main-path $H$\n{100 * routed_run:.0f}% unsure', pad=2, linespacing=1.05)
    save(fig, 'framework')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imagenet-val', required=True); ap.add_argument('--imagenet-c', required=True)
    ap.add_argument('--cache-dir', default='cache'); ap.add_argument('--stats', required=True)
    ap.add_argument('--logits', default='runs/logits', help='saved logits of the design runs (the trunk of Figure S1)')
    args = ap.parse_args()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    src = {k: v for k, v in torch.load(args.stats, map_location='cpu').items() if k in ('tok_mu', 'tok_cov')}
    index = data.ImageNetIndex(args.imagenet_val)
    idx = data.split_indices('design')[:2000]
    i_img = (IMAGE_BATCH - 1) * BATCH
    cleans_u8 = data.load_images(index, 'clean', idx, args.imagenet_c, args.cache_dir)
    clean = models.to_input(np.asarray(cleans_u8[i_img:i_img + 1]), 'cpu')[0]
    pick = patch_pick(clean)
    D = json.loads((ROOT / 'results' / 'paper_numbers.json').read_text())
    R, nums = [], {'image': 'design split, first image of batch 16 (split index 960)', 'patches_flat_textured': list(pick), 'cells': {}}
    for cell, _ in ROWS:
        imgs = data.load_images(index, cell, idx, args.imagenet_c, args.cache_dir)
        r = replay(cell, imgs, src); R.append(r)
        nums['cells'][cell] = {'n_patches': r['n'], 'g_spec': r['gate'][0], 'g_mean': r['gate'][1], 'gate_on': r['on'],
                               'drift_z_pixel': D['two_site']['drift_vitb']['z_spec_first_k'][cell]['pixel'],
                               'drift_z_b6': D['two_site']['drift_vitb']['z_spec_first_k'][cell]['b6']}
        print(cell, nums['cells'][cell], flush=True)
    cell0 = ROWS[0][0]
    z = np.load(Path(args.logits) / f'forward_only_design_vitb_two_site_tok+place_b6_{cell0}.npz')
    lg = torch.tensor(z['logits'].astype(np.float32)).double(); lp = torch.log_softmax(lg, 1)
    trunk_h = (-(lp.exp() * lp).sum(1)).numpy()
    routed_run = json.loads((ROOT / 'results' / 'scores' / 'forward_only_design_vitb_exit.json').read_text())['routed_fraction']['exit_b5b9_gated'][cell0]
    nums['trunk_entropy'] = {'cell': cell0, 'theta': THETA, 'share_above_theta_from_logits': float((trunk_h > THETA).mean()), 'routed_share_of_the_run': routed_run}
    framework(R[0], clean, pick, trunk_h, THETA, routed_run)
    (OUT / 'example_numbers.json').write_text(json.dumps(nums, indent=1) + '\n')
    print('wrote framework.pdf, example_numbers.json')


if __name__ == '__main__':
    main()
