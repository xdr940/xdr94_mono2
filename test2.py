
import numpy as np
import matplotlib.pyplot as plt
import torch

error_maps_g =np.load('./data_temp/combind_g.npy')


sample0 = error_maps_g[0]
sample1 = error_maps_g[1]
sample2 = error_maps_g[2]
sample3 = error_maps_g[3]
sample5 = error_maps_g[5]

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
temp,idx = torch.min(sample,dim=0)
temp_test = torch.softmax(sample,dim=0).type(torch.float)
#idx = torch.argmax(temp,dim=0)

#temp,_ = torch.max(temp,dim=0)
temp = temp.detach().cpu().numpy()
idx = idx.detach().cpu().numpy()

plt.subplot(3,3,5)
plt.imshow(temp)
plt.subplot(3,3,6)
plt.imshow(idx)
plt.subplot(3,3,7)

temp2 = np.zeros_like(temp)
idxs = np.logical_and(temp>0.3,idx<=1)

temp2[(temp>0.3)*(temp<0.38)]=0.9
plt.imshow(temp2,cmap='rainbow')
plt.show()



print('ok')