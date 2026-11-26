"""Compute the source statistics once (no labels): mean and covariance of the 16x16x3 patches and of the head input,
from 3,000 clean ImageNet validation images (split 'source'). With --blocks also, for every block's output, the mean and
covariance of the patch tokens (``patch_mu`` / ``patch_cov``; the block-site alignment of the main path reads block 6)
and of all 197 tokens (``blk_mu`` / ``blk_cov``; only the multi-block alignment arms of the smoke test use them).
Always also the within-image and between-image parts of the patch covariance (``tok_cov_within`` / ``tok_cov_between``
at the pixel site and, with --blocks, ``patch_cov_within`` / ``patch_cov_between`` per block): the covariance of the
image means and the mean covariance of the deviations from the image mean, which add up to the pooled covariance.
The hierarchical alignment aligns the two parts separately.

    python scripts/compute_source_stats.py --model vitb --blocks --imagenet-val /data/imagenet/val --out stats/vitb.pt
    python scripts/compute_source_stats.py --model vitb --n-src 50 --imagenet-val /data/imagenet/val --out stats/vitb_n50.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokalign import data, models  # noqa: E402
from tokalign.runner import BATCH, seed_everything  # noqa: E402
from tokalign.stages import Running, unfold  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=list(models.CONFIG), default='vitb')
    ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--n-src', type=int, default=3000)
    ap.add_argument('--blocks', action='store_true', help='also per-block token statistics')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    seed_everything()
    dev = torch.device(args.device)
    model = models.build(args.model).to(dev)
    hook = models.HeadInput(model)
    index = data.ImageNetIndex(args.imagenet_val)
    idx = data.split_indices('source')[:args.n_src]
    imgs = data.load_cell(index, 'clean', idx, clean_resize=models.clean_resize(args.model))
    tok, head = Running(768, dev), Running(model.embed_dim, dev)
    tok_b, tok_w = Running(768, dev), Running(768, dev)              # image means / within-image deviations of the patches
    d = model.embed_dim
    blk = [Running(d, dev) for _ in model.blocks] if args.blocks else []
    patch = [Running(d, dev) for _ in model.blocks] if args.blocks else []
    patch_b = [Running(d, dev) for _ in model.blocks] if args.blocks else []
    patch_w = [Running(d, dev) for _ in model.blocks] if args.blocks else []
    skip = getattr(model, 'num_prefix_tokens', 1)                     # CLS (timm 0.4.12 has no attribute; one prefix token)
    def split(u):                                                    # u (B, T, d) -> image means (B, d), deviations (B*T, d)
        m = u.mean(1)
        return m, (u - m[:, None, :]).reshape(-1, u.shape[-1])
    def record(r, rp, rb, rw):
        def hook(m, i, o):                                           # returns None: the block output is left unchanged
            r.update(o.reshape(-1, d)); rp.update(o[:, skip:].reshape(-1, d))
            mb, ew = split(o[:, skip:]); rb.update(mb); rw.update(ew)
        return hook
    for l, (r, rp, rb, rw) in enumerate(zip(blk, patch, patch_b, patch_w)):
        model.blocks[l].register_forward_hook(record(r, rp, rb, rw))
    with torch.no_grad():
        for s in range(0, len(imgs), BATCH):
            x = models.to_input(imgs[s:s + BATCH], dev)
            u = unfold(x)
            tok.update(u)
            mb, ew = split(u.reshape(x.shape[0], -1, 768)); tok_b.update(mb); tok_w.update(ew)
            model(x)
            head.update(hook.h)
    out = {'tok_mu': tok.mean().cpu(), 'tok_cov': tok.cov().cpu(), 'head_mu': head.mean().cpu(), 'head_cov': head.cov().cpu(),
           'tok_cov_between': tok_b.cov().cpu(), 'tok_cov_within': tok_w.cov().cpu(),
           'model': args.model, 'n_images': len(imgs)}
    if blk:
        out['blk_mu'], out['blk_cov'] = [r.mean().cpu() for r in blk], [r.cov().cpu() for r in blk]
        out['patch_mu'], out['patch_cov'] = [r.mean().cpu() for r in patch], [r.cov().cpu() for r in patch]
        out['patch_cov_between'], out['patch_cov_within'] = [r.cov().cpu() for r in patch_b], [r.cov().cpu() for r in patch_w]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f'saved {args.out}: {len(imgs)} images')


if __name__ == '__main__':
    main()
