"""Analytic compute per batch of every timed method, in units of one ViT-B/16 forward pass (17.56 GMACs per image, 196 patch
tokens of d = 768), shared by paper/make_numbers.py (the compute column of Table S6) and paper/make_figures.py (Figure S7).
Convention: a map costs one d x d multiply per token (0.66 % of a forward pass), the extra branches run only on the unsure share
u of the images (the branch after a block a1 below the main block b re-runs blocks a1+1..12, the branch after a block a2 above it re-runs blocks b+1..12,
each with its own map), a backward pass counts as two forward passes (ETA: one forward and one backward, SAR: two of each),
PEA makes two passes with a map after every block (times three views with augmentation), and FOA runs its population of 27
prompts plus the unprompted model, 28 forward passes per batch."""
VITB_GMACS = 17.56
MAP = 196 * 768 ** 2 / (VITB_GMACS * 1e9)   # one map, in forward passes


def branch(a1, a2, b=6, blocks=12):
    """Forward passes of the two extra branches of one unsure image (blocks a1 < b < a2, as every timed cascade has)."""
    return (blocks - a1) / blocks + MAP + (blocks - b) / blocks + MAP


def forward_passes(u_cascade, u_b4, u_b11, u_random, u_b5b9):
    """Compute per batch in forward passes for every row of Table 1 and Table S6. The u's are the unsure shares of the runs."""
    main = 1 + 2 * MAP
    return {'neo': 1.0, 'tok': 1 + MAP, 'pair': main,
            'b4b11': main + u_cascade * branch(4, 11), 'b4': main + u_b4 * ((12 - 4) / 12 + MAP), 'b11': main + u_b11 * ((12 - 6) / 12 + MAP),
            'random': main + u_random * branch(4, 11), 'b5b9': main + u_b5b9 * branch(5, 9),
            'eta': 3.0, 'sar': 6.0, 'pea': 2 * (1 + 12 * MAP), 'peaviews': 3 * 2 * (1 + 12 * MAP), 'foa': 28.0}
