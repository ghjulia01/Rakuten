# python tools/test_gpu.py
# Test si torch_directml est bien installé et fonctionne

import torch, torch_directml
dml = torch_directml.device()
x = torch.randn(2048, 2048, device=dml)
y = x @ x
print(y.device)  