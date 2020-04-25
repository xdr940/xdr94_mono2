import torch
import torch.nn.functional as F
import  numpy as np
import torch.nn as nn
import datetime
def main():
    print()
    B,H,W= 2,3,4
    err = np.array([[[.4, .3, .25, .7],
                     [3., .45, 3., 3.],
                     [2.5, 1., 3., 3.]],

                    [[.4, .3, .25, .1],
                     [3., .45, 3., 3.],
                     [3., 2., 3., 3.]],

                    [[.1, .2, .25, .1],
                     [3., .05, 3., 3.],
                     [2., 3., 3., 3.]],

                    [[.1, .2, .25, .1],
                     [1., .05, 3., 3.],
                     [1., 4., 3., 3.]]])
    err_t = torch.tensor(err)
    err = np.expand_dims(err, axis=0)
    b_err = np.concatenate([err, 2 * err], axis=0)
    b_err = torch.tensor(b_err).float()#1,4,3,4
    b_err = b_err.transpose(2,1).transpose(2,3)
    on = torch.ones([2,3,4,1]).float()
    a = b_err*on
    var = b_err.var(dim=1,unbiased = False)
    print(var.shape)
    print(var)
    var = var.flatten(start_dim=1)
    print(var.shape)
    print(var)

    max,idx = var.max(dim=1)
    mean = var.mean(dim=1).unsqueeze(1)
    print(max.shape)
    print(max)
    print(mean.shape)
    print(mean)
    print(var/max.unsqueeze(1))
    print(var> mean)

    #var = var.reshape(B,H,W)
    #print(var.shape)
    #print(var)
   # var = b_err.var(dim=1)  # b4hw --> bhw




def main2():
    pass
    img = torch.tensor([[.1, .2, .25, .1],
                     [1., .05, 3., 3.],
                     [3., 3., 3., 3.]])

    norm_= F.normalize(img,p=1)
    print(norm_)



def main3():
    m = nn.BatchNorm2d(2, affine=True)  # affine参数设为True表示weight和bias将被使用
    input = torch.tensor([[[[1.4174, -1.9512, -0.4910, -0.5675],
              [1.2095, 1.0312, 0.8652, -0.1177],
              [-0.5964, 0.5000, -1.4704, 2.3610]],

             [[-0.8312, -0.8122, -0.3876, 0.1245],
              [0.5627, -0.1876, -1.6413, -1.8722],
              [-0.0636, 0.7284, 2.1816, 0.4933]]]])

    mean = torch.Tensor.mean(input[0][0])
    var = torch.Tensor.var(input[0][0], False)
    print(m)
    print('m.eps=',m.eps)
    print('--')
    print(input)
    print(mean)
    print(var)
    print('--')
    for i in range(4):
        mean = input[0][0].mean()
        var = input[0][0].var(False)

        batchnormone = ((input[0][0][0][i] - mean) / (torch.pow(var, 0.5) + m.eps)) \
                       * m.weight[0] + m.bias[0]
        print(batchnormone)

    print(m(input))

def main4():
    m = nn.InstanceNorm2d(2, affine=True)  # affine参数设为True表示weight和bias将被使
    input = torch.tensor([[[[-0.4011, 0.7260, 1.2056, -2.0086],
                            [0.1647, 0.7125, -0.1252, 0.2391],
                            [-0.4096, 0.1757, -1.4462, -0.5267]],

                           [[1.3256, -1.3513, -1.7509, -0.2615],
                            [1.0622, -2.4472, -0.4389, -0.6118],
                            [-0.6774, 1.2427, 0.1448, -0.3426]]],

                          [[[0.4014, -1.5730, 0.2782, -0.1271],
                            [0.3693, 0.7909, -0.9022, -0.8561],
                            [0.0980, 0.0439, 0.6934, -0.9185]],

                           [[-0.6728, 0.1291, -0.5936, -2.0691],
                            [-0.7393, 1.5585, -0.5516, 0.2580],
                            [1.6092, 1.0505, -1.9855, -0.7834]]]])

    #mean = (input[0][0] + input[1][0]).mean()
    #var = torch.cat([input[0][0].unsqueeze(0), input[1][0].unsqueeze(0)],dim=0).var(False)
    mean = input[0][0].mean()
    var = input[0][0].var()
    print(m.weight)
    batchnormone = ((input[0][0][0][0] - mean) / (torch.pow(var, 0.5) + m.eps)) \
                   * m.weight[0] + m.bias[0]
    print(batchnormone)

    output = m(input)
    print(output)
def main0():
    a = torch.tensor([1,2,1.,2])
    print(    a.var(False))

main()

#print(var_norm.shape)
#print(var_norm)
