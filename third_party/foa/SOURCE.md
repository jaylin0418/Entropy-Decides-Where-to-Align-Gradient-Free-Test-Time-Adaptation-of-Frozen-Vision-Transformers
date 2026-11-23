Official FOA code (Niu et al., "Test-Time Model Adaptation with Only Forward Passes", ICML 2024), used by
`scripts/episodic/run_foa.py` (the FOA rows of the five-host table and the FOA settings in the supplement).

- Upstream: https://github.com/mr-eggplant/FOA, commit `004b02dab6a7c02afce1caa606afac525dd74f7e`.
- Licence: NTUITIVE Dual License Agreement, non-commercial use only. The licence restricts distribution, so this
  repository does not include the code; fetch it as below. Research evaluation only.
- Needed: `models/vpt.py`, `tta_library/foa.py` and the modules they import (`utils/__init__.py`, `utils/cli_utils.py`,
  `calibration_library/metrics.py`, `quant_library/quant_layers/matmul.py`), unchanged. The runs used ViT-B/16 only,
  for which the upstream code works as is.
- Requires `cma` (pip) and timm >= 0.9 for FOA's model code (the runs used timm 1.0.15 in a separate environment).

Fetch from the repository root:

```
git clone https://github.com/mr-eggplant/FOA /tmp/foa_upstream && git -C /tmp/foa_upstream checkout 004b02dab6a7c02afce1caa606afac525dd74f7e
(cd /tmp/foa_upstream && cp --parents models/vpt.py tta_library/foa.py utils/__init__.py utils/cli_utils.py calibration_library/metrics.py quant_library/quant_layers/matmul.py "$OLDPWD/third_party/foa/")
```
