import torch
from sampling import furthest_point_sample

x = torch.randn(2, 100, 3).cuda()
print(x.shape)
y = furthest_point_sample(x, 10)
print(y.shape)
