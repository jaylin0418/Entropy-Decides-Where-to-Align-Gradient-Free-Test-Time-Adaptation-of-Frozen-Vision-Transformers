# Hierarchical token alignment: the estimator, the proposition behind it, and the novelty boundary

Internal note (2026-11-08) for the method pre-registered in `docs/heldout_test_plan_hier.md`. The paper text is
written from this note once the design bar and the held-out runs are in; the numbers come from the repository only.

## 1. Setting

At a site of a frozen ViT (pixel patches, or the T = 196 patch tokens after block k) the tokens of image i are
x_ip ∈ R^d, p = 1..T. Write m_i = (1/T) Σ_p x_ip for the image mean and e_ip = x_ip − m_i for the within-image
deviation. For any batch the token covariance splits exactly into two parts (the one-way ANOVA identity):

    Σ = Cov_ip(x_ip) = Cov_i(m_i) + E_i Cov_p(e_ip) =: Σ_B + Σ_W .

The pooled CORAL map of TokAlign / LayerAlign, y = μ_s + (x − μ_t) M with M = Σ_t^{-1/2} Σ_s^{1/2}, treats the sum.
The hierarchical map aligns each part with its own map:

    y_ip = μ_s + (m_i − μ_t) M_B + e_ip M_W,   M_B = Σ_{B,t}^{-1/2} Σ_{B,s}^{1/2},   M_W = Σ_{W,t}^{-1/2} Σ_{W,s}^{1/2} .

Both maps are closed-form, label-free and computed in the forward pass from running target statistics with a source
prior (n₀ = 64 images for Σ_B, n₀ = 4 images × T tokens for Σ_W) and the constant shrinkage λ = 0.1 of TokAlign.
Σ_B is estimated from B = 64 image means per batch, Σ_W from 64 × 196 deviations, so the between part is the one that
needs the prior; this is the reason for the two different n₀.

## 2. Proposition (population statistics)

Let the source parts be (Σ_B, Σ_W) and the target parts (Σ_B + Γ, Σ_W + Δ) with Γ, Δ ⪰ 0.

(a) *Exactness.* The hierarchical map restores both parts: M_Bᵀ (Σ_B + Γ) M_B = Σ_B and M_Wᵀ (Σ_W + Δ) M_W = Σ_W. The
pooled map restores only the sum: Mᵀ (Σ_B + Σ_W + Γ + Δ) M = Σ_B + Σ_W, and Mᵀ (Σ_B + Γ) M ≠ Σ_B in general.

(b) *Where the pooled map puts the error.* If Σ_B, Σ_W, Γ, Δ share an eigenbasis with eigenvalues b_j, w_j, γ_j, δ_j
(a_j = b_j + w_j), the pooled map leaves, along direction j,

    between variance  b_j · a_j / (a_j + γ_j + δ_j),     within variance  w_j · a_j / (a_j + γ_j + δ_j) .

So when only the within part is inflated (γ = 0, δ > 0, the pixel-noise case), the pooled map compresses the
between-image variance by the factor a_j/(a_j + δ_j) in exactly the directions the noise inflated, although the
between part did not move. When only the between part moves (γ > 0, δ = 0, a global colour or contrast change), the
pooled map flattens the within-image structure of every image by a_j/(a_j + γ_j). The hierarchical map changes
neither. The two maps coincide iff γ_j / b_j = δ_j / w_j for every j, i.e. iff the shift scales both parts alike.

*Proof.* (a) is the definition of the symmetric square root. (b): in the shared eigenbasis all matrices are diagonal
and the pooled map is the diagonal matrix with entries √(a_j / (a_j + γ_j + δ_j)); conjugating diag(b_j) and
diag(w_j) by it gives the two expressions. Equality with the hierarchical map requires
a_j/(a_j + γ_j + δ_j) = b_j/(b_j + γ_j) = w_j/(w_j + δ_j), which is the stated condition. ∎
`tests/test_hier.py::test_proposition_*` checks (a) on random non-commuting matrices and (b) numerically.

## 3. What the proposition predicts, and how the prediction is tested without labels

The class information the head reads at a mid-depth site is between-image information (images of different classes
have different token means; the within-image deviations are the spatial layout of one image). Compressing Σ_B along
the noise directions therefore removes class separation, which is the reading we give to the one negative row of the
pooled block-6 map on the design split (noise: −1.5 to −1.9 pt against NEO, while all other families gain). The
decomposition run (`scripts/analysis/decompose_by_depth.py`, no labels) measures log tr Σ̂_W / tr Σ_W,s and
log tr Σ̂_B / tr Σ_B,s at every site for every cell. The reading is right if, at the sites where the pooled map hurts
(block 6, noise), the within log-ratio is large and positive while the between log-ratio is near zero; it is wrong
if both parts move alike, in which case the proposition says the two maps nearly coincide and the hierarchical map
cannot help. Accuracy on the design split then says whether the restored between part is what the head needed
(design bar (iii): `hier_b6` ≥ NEO on the six noise cells).

## 4. Novelty boundary (what is known, what is ours)

Known:
- The within/between (instance/batch) decomposition of feature statistics is classical (one-way ANOVA, random
  effects) and is the basis of instance vs batch normalisation: IBN-Net (Pan et al. 2018), Batch-Instance
  Normalisation (Nam & Kim 2018), Switchable Whitening (Pan et al. 2019; instance whitening vs batch whitening of
  CNN features at training time).
- Test-time normalisation methods mix instance and batch *per-channel mean and variance* inside BN layers of CNNs:
  MixNorm (Hu et al. 2021), α-BN (You et al. 2021), TTN (Lim et al. 2023, learned per-layer interpolation weight),
  UnMix-TNS (Tomar et al. 2024), DIGA (Wang et al. 2023, segmentation). None forms a covariance, none has a
  recolouring to the source, and none applies to ViTs, which have no BN.
- Covariance alignment to the source is CORAL (Sun et al. 2016) and, per token at the input, TokAlign; alignment with
  class structure (within-class / between-class covariances) needs labels or pseudo-labels: CFA (Kojima et al. 2022),
  CAFe (Adachi et al. 2023), class-conditional DA.

Ours:
- Splitting the token covariance at a ViT site into the between-image and within-image parts and aligning each with
  its own closed-form whitening-recolouring map, label-free, forward-only, with a source prior sized to the number of
  samples each part has. The decomposition is by image, not by class, so it needs no labels.
- The proposition in §2: the pooled map is exact for the sum and provably distorts the part that did not move, by a
  computable factor; the hierarchical map is exact for both parts. This explains a failure mode of pooled alignment
  (noise at mid depth) that the drift magnitude cannot explain (the drift at block 6 is largest for noise).
- A label-free diagnostic that measures the two parts separately at every depth, so the prediction can be checked
  independently of accuracy.

Boundary of the claim: the proposition is about population statistics of a two-level model with independent parts;
with finite batches the estimator adds a prior and shrinkage, and whether the restored between part is what the head
uses is an empirical question answered by the design bar and the held-out claims T3/T4. If the decomposition shows
both parts moving alike for most corruptions, the method is expected to tie the pooled map there, and the paper says
so.

## Outcome (2026-11-09)

The design bar of `heldout_test_plan_hier.md` failed (+0.15 over the pooled two-site map, noise still negative at block 6);
see the outcome section there. The proposition stands and is tested numerically (`tests/test_hier.py`); the empirical
premise that noise inflates the within-image part at block 6 does not hold (the within part is unchanged there and the
between part collapses), which is why the two-level map cannot repair block 6 on noise. The report presents the split, the
proof and the decomposition as the explanation of what the block-6 map corrects (a within-image correction) and reports
the two-level accuracies as a negative result.
