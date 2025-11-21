import torch

ckpt = torch.load("two_tower.pt", map_location="cpu")
print(type(ckpt))
try:
    print(ckpt.keys())
except AttributeError:
    print("Not a dict, probably just a state_dict")
