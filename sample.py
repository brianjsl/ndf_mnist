import torch

a = torch.tensor([1,2,3])
torch.save(a, './checkpoints/a.pt')