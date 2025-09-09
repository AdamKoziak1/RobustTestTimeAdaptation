import torch, time, torch.cuda.profiler as nv
from utils.util import drop_low_singular_values

bs=64
xb = torch.randn(bs, 3, 224, 224)    



# t0 = time.time()
# _ = drop_low_singular_values(xb, 223, full_decomposition=True)
# print("full:", (time.time()-t0)/bs)

# xb = xb.cuda(non_blocking=True)           # GPU
# torch.cuda.synchronize()
# t0 = time.time()
# _ = drop_low_singular_values(xb, 223)
# torch.cuda.synchronize()
# print("lowrank SVD:", (time.time()-t0)/bs)



k=50
B, C, H, W = xb.shape      
x_flat = xb.reshape(B * C, H, W)

t0 = time.time()
U, S, Vh = torch.linalg.svd(x_flat, full_matrices=True)
print("full:", (time.time()-t0)/bs)





t0 = time.time()
q        = H - k
U,S,Vh   = torch.svd_lowrank(x_flat, q=q, niter=2)
print("lowrank SVD:", (time.time()-t0)/bs)




t0 = time.time()
S = torch.linalg.svdvals(x_flat)
print("SVDvals:", (time.time()-t0)/bs)


