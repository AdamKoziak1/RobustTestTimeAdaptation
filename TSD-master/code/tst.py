import torch, time, torch.cuda.profiler as nv
from utils.util import drop_low_singular_values

bs=64
xb = torch.randn(bs, 3, 224, 224)        
t0 = time.time()
_ = drop_low_singular_values(xb, 223, full_decomposition=True)
print("full:", (time.time()-t0)/bs)

xb = xb.cuda(non_blocking=True)           # GPU
torch.cuda.synchronize()
t0 = time.time()
_ = drop_low_singular_values(xb, 223)
torch.cuda.synchronize()
print("lowrank SVD:", (time.time()-t0)/bs)
