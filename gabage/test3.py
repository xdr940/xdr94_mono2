import numpy as np
import matplotlib.pyplot as plt
pose = np.load('poses.npy')
start = np.array([0,0,0,1]).reshape([4,1])

len,_,_ = pose.shape

p =[]
last_p = start
for i in range(len):
    last_p = pose[i,] @ last_p

    p.append(np.expand_dims(last_p,axis=0))
p = np.concatenate(p,axis=0)
p= np.squeeze(p,axis=2)

plt.plot(p[:,0],p[:,1])
plt.show()

print('ok')