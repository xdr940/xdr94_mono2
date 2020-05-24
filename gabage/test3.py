import torch
#[3,3,3
# 3,3,3]
#[2
# 2
# 2]
#
#
#
def uint8or(t1,t2):

    return ((t1 + t2) > 0).float()

a = torch.ones([2,3])*3
b = torch.ones([3])*2

print(uint8or(a,b))