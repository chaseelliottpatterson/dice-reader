#nvidia-smi works alreadypy
# https://developer.nvidia.com/cuda-downloads
# https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import torch
import torchvision

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

print("CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
