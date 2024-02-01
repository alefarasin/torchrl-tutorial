# torchrl-tutorial

Implementation of the examples in the official TorchRL guide: https://pytorch.org/rl/

# Installation
- scoop bucket add versions
- scoop install versions/python311
- poetry env use python311

- poetry source add pytorch https://download.pytorch.org/whl/cu121/ -p explicit
- poetry add --source pytorch torch torchvision torchaudio
