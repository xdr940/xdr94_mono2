
import torch
import torch.nn.functional as F
kernel33 = torch.tensor([0,1,0,
                           1, 1, 1,
                           0, 1, 0]).type(torch.float).reshape([1,1, 3, 3]).cuda()
weight75 = torch.tensor([0, 0, 0,0,0,
                           1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1,
                           1,1,1,1,1,
                           1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1,
                           0,0,0,0,0]).type(torch.float).reshape([1,1,7,5])




def dilation(batch,kernel=kernel33):

    w_sum = kernel.sum()
    batch =batch.type_as(kernel)
    batch = batch.unsqueeze(dim=1)
    #batch = torch.tensor(batch).type_as(kernel)
    ret = F.conv2d(input=batch, weight=kernel, padding=1)
    ret[ret < w_sum] = 0
    ret /= w_sum
    ret = F.conv2d(input=ret, weight=kernel, padding=1)
    ret[ret < w_sum] = 0
    ret /= w_sum
    ret = F.conv2d(input=ret, weight=kernel, padding=1)
    ret[ret < w_sum] = 0
    ret /= w_sum
    return ret.squeeze(dim=1)


def erosion(batch,kernel=kernel33):


    w_sum = kernel.sum()
    batch =batch.type_as(kernel)
    batch = batch.unsqueeze(dim=1)
    ret = F.conv2d(input=batch, weight=kernel, padding=1)
    ret[ret > 0] = 1
    ret = F.conv2d(input=ret, weight=kernel, padding=1)
    ret[ret > 0] = 1
    ret = F.conv2d(input=ret, weight=kernel, padding=1)
    ret[ret > 0] = 1
    return ret.squeeze(dim=1)

def rectify(batch):
    return dilation(erosion(batch))