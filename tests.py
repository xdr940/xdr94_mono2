import numpy as np
import matplotlib.pyplot as plt
import torch
error_maps = np.load('./data_temp/tt.npy')


depth_0 = np.load('./data_temp/depth_0.npy')
depth_w_m1 = np.load('./data_temp/depth_w_m1.npy')
depth_w_1 = np.load('./data_temp/depth_w_1.npy')



sample0 = error_maps[0]
sample1 = error_maps[1]
sample2 = error_maps[2]
sample3 = error_maps[3]
sample5 = error_maps[5]

sample = sample0

plt.subplot(3,3,1)
plt.imshow(sample[0],cmap='rainbow')
plt.subplot(3,3,2)
plt.imshow(sample[1],cmap='rainbow')
plt.subplot(3,3,3)

plt.imshow(sample[2],cmap='rainbow')
plt.subplot(3,3,4)

plt.imshow(sample[3],cmap='rainbow')



sample = torch.tensor(sample)
min_err_map,idx = torch.min(sample,dim=0)
#min_err_map = 1- torch.softmax(sample,dim=0).type(torch.float)
#idx = torch.argmax(min_err_map,dim=0)
#min_err_map,_ = torch.max(min_err_map,dim=0)




temp = min_err_map.detach().cpu().numpy()
idx = idx.detach().cpu().numpy()

plt.subplot(3,3,5)
plt.imshow(temp,cmap='rainbow')
plt.subplot(3,3,6)
plt.imshow(idx)
plt.subplot(3,3,7)

temp2 = np.zeros_like(temp)
idxs = np.logical_and(temp>0.3,idx<=1)

temp2[(temp>0.3)*(temp<0.38)]=0.9
plt.imshow(temp2,cmap='rainbow')
plt.show()



print('ok')