Official PEA code for ViTs (progressive embedding alignment), used by `scripts/episodic/run_pea.py` (the PEA rows of
the five-host table and the PEA settings in the supplement).

- Upstream: https://github.com/TheMaXiao/PEA_TTA, commit `9c358ba682052862df0739fbbadb71f1b586614d`.
- Licence: **none found** (no LICENSE file, no licence in the README). Without a licence the code may not be
  redistributed, so this repository does not include it; fetch it as below. Research evaluation only.
- Needed, unchanged: `pea_vit.py` (it imports only torch, torchvision and tqdm). `run_pea.py` uses the `imagenet_c`
  defaults of the upstream `main_vit.py` (momentum 0.02, alpha 0.2, gamma 4.0, entropy threshold 1.0). `pea_vit.py`
  turns TF32 on when it is imported; `run_pea.py` turns it off again right after loading it.

Fetch from the repository root:

```
git clone https://github.com/TheMaXiao/PEA_TTA /tmp/pea_upstream && git -C /tmp/pea_upstream checkout 9c358ba682052862df0739fbbadb71f1b586614d
(cd /tmp/pea_upstream && cp --parents pea_vit.py "$OLDPWD/third_party/pea/")
```
