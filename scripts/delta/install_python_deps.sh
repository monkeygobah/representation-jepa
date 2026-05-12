#!/usr/bin/env bash
# Install repo dependencies missing from NCSA's DeltaAI PyTorch module.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

python -m pip install --user \
  imageio \
  more-itertools \
  scikit-image \
  tzlocal \
  seaborn \
  nevergrad \
  opencv-python-headless \
  pyyaml \
  timm==1.0.26 \
  webdataset

python - <<'PY'
mods = [
    "numpy", "pandas", "scipy", "imageio", "more_itertools", "skimage",
    "PIL", "sklearn", "pytz", "tzlocal", "seaborn", "matplotlib",
    "tqdm", "nevergrad", "cv2", "yaml", "timm", "webdataset", "torch", "torchvision",
]
for m in mods:
    __import__(m)
    print(f"OK {m}")
PY

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
