# smoke_test.py
import sys, types

# pycolmap richiede una DLL nativa bloccata da application control policy
# su questa macchina; non serve per inferenza depth/pose base (solo per
# export in formato COLMAP, che non usiamo). Stub per bypassare l'import.
sys.modules["pycolmap"] = types.ModuleType("pycolmap")

import torch
from depth_anything_3.api import DepthAnything3

device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")
model = model.to(device=device)

import glob
images = sorted(glob.glob("test_images/*.jpg"))
prediction = model.inference(images)

print("depth:", prediction.depth.shape)
print("extrinsics:", prediction.extrinsics.shape)
print("intrinsics:", prediction.intrinsics.shape)