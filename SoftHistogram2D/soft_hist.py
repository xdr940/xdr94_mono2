import torch
import torch.nn as nn
import matplotlib.pyplot as plt



class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)

    def forward(self, x):
        self.centers = self.centers.unsqueeze(dim=1).cuda()#(bins,1)
        x_ = torch.unsqueeze(x, 0)#(1,len)
        x1 = self.centers - x_ #(1,len) -(bins,1) == (bins,len), largest data
        x2 = torch.sigmoid(self.sigma * (x1 + self.delta/2)) - torch.sigmoid(self.sigma * (x1 - self.delta/2))
        x3 = x2.sum(dim=1)
        return x3
class SoftHistogramBCHW(nn.Module):
    def __init__(self, device,bins, min, max, sigma,b=None,c=None,h=None,w=None,Type='H'):
        super(SoftHistogramBCHW, self).__init__()
        assert Type=='W' or Type=='H'
        self.device = device
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)#
        self.centers = self.centers.to(self.device)
        self.Type =Type

        #
        self.centers = self.centers.unsqueeze(dim=0)#[bins,1]

        if self.Type =='H':#垂直不变，对水平每一行进行统计
            self.centers = torch.ones([h,1]).to(self.device)@self.centers#[h,1]@[1,bins] = [h,bins]
            self.centers = self.centers.unsqueeze(dim=-1)#[h,bins,1]#倒数第一个,有时候BCHW有时候HW
            # centers
            self.centers = self.centers @ torch.ones([1, w]).to(self.device)  # [h,bins,1]@[1,w] =[h,bins,w]

        elif self.Type =='W':
            self.centers = torch.ones([w,1]).to(self.device)@self.centers#[w,1]@[1.bins] = [w,bins]
            self.centers = self.centers.unsqueeze(dim=-1)#[h,bins,1]#倒数第一个,有时候BCHW有时候HW
            self.centers = self.centers @ torch.ones([1, h]).to(self.device)  # [w,bins,1]@[1,h] =[w,bins,h]



    def forward(self, x):
        #x
        if self.Type=='H':#[b,c,h,w]
            x_=x
        else:#self.Type=='W'
            x_ = torch.transpose(x,-1,-2)#[b,c,w,h]

        x_ = x_.unsqueeze(dim=-2)  # [b,c,h,w]->[b,c,h,1,w]  or [h,w]->[h,1,w]

        x_=torch.ones([self.bins,1]).cuda()@x_

        #for 'H' [bins,1]@[h,1,w] = [h,bins,w] or [bins,1]@[b,c,h,1,w] ->[b,c,h,bins,w]
        #for 'W' [bins,1]@[w,1,h] = [w,bins,h] or [bins,1]@[b,c,w,1,h] ->[b,c,w,bins,h]

        ret = self.centers - x_
        ret = torch.sigmoid(self.sigma * (ret + self.delta / 2)) - torch.sigmoid(self.sigma * (ret - self.delta / 2))  # bins,h,w
        ret = ret.sum(dim=-1)
        return ret
class SoftHistogram2D_W(nn.Module):
    def __init__(self,device, bins, min, max, sigma,b=None,c=None,h=None,w=None):
        super(SoftHistogram2D_W, self).__init__()
        assert h!=None and w !=None
        self.device = device
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)#
        self.centers = self.centers.to(self.device)

        #
        self.centers = self.centers.unsqueeze(dim=0)#[bins,1]


        self.centers = torch.ones([w,1]).to(self.device)@self.centers#[w,1]@[1.bins] = [w,bins]
        self.centers = self.centers.unsqueeze(dim=-1)#[h,bins,1]#倒数第一个,有时候BCHW有时候HW
        self.centers = self.centers @ torch.ones([1, h]).to(self.device)  # [h,bins,1]@[1,w] =[h,bins,w]



    def forward(self, x):
        x_ = torch.transpose(x,-1,-2)#[b,c,w,h]

        x_ = x_.unsqueeze(dim=-2)  # [b,c,h,w]->[b,c,h,1,w]  or [h,w]->[h,1,w]

        x_=torch.ones([self.bins,1]).to(self.device)@x_
        #for 'H' [bins,1]@[h,1,w] = [h,bins,w] or [bins,1]@[b,c,h,1,w] ->[b,c,h,bins,w]
        #for 'W' [bins,1]@[w,1,h] = [w,bins,h] or [bins,1]@[b,c,w,1,h] ->[b,c,w,bins,h]

        ret = self.centers - x_
        ret = torch.sigmoid(self.sigma * (ret + self.delta / 2)) - torch.sigmoid(self.sigma * (ret - self.delta / 2))  # bins,h,w
        ret = ret.sum(dim=-1)
        return ret



class SoftHistogram2D_H(nn.Module):
    def __init__(self,device, bins, min, max, sigma,b=None,c=None,h=None,w=None,):
        super(SoftHistogram2D_H, self).__init__()
        self.device = device
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = min + self.delta * (torch.arange(bins).float() + 0.5)#

        #
        self.centers = self.centers.unsqueeze(dim=0).type(torch.float).to(self.device)#[bins,1]

        self.centers = torch.ones([h,1],dtype=torch.float).to(self.device)@self.centers#[h,1]@[1,bins] = [h,bins]
        self.centers = self.centers.unsqueeze(dim=-1)#[h,bins,1]#倒数第一个,有时候BCHW有时候HW
        # centers
        self.centers = (self.centers @ torch.ones([1, w],dtype=torch.float).to(self.device))  # [h,bins,1]@[1,w] =[h,bins,w]
        self.centers = nn.Parameter(self.centers,requires_grad=False)
        #translation I matrix
        self.ones = torch.ones([self.bins,1],dtype=torch.float).to(self.device)
        self.ones = nn.Parameter(self.ones,requires_grad=False)
        #print('ok')





    def forward(self, x):
        x_ = x.unsqueeze(dim=-2)  # [b,c,h,w]->[b,c,h,1,w]  or [h,w]->[h,1,w]
        x_=self.ones@x_#big memory 10^8 iterm
        #for 'H' [bins,1]@[h,1,w] = [h,bins,w] or [bins,1]@[b,c,h,1,w] ->[b,c,h,bins,w]
        #for 'W' [bins,1]@[w,1,h] = [w,bins,h] or [bins,1]@[b,c,w,1,h] ->[b,c,w,bins,h]

        x_ = self.centers - x_
        x_ = torch.sigmoid(self.sigma * (x_ + self.delta / 2)) - torch.sigmoid(self.sigma * (x_ - self.delta / 2))  # bins,h,w
        x_ = x_.sum(dim=-1)
        return  x_



def test_softhistgram():
    x = torch.randn([200,])*100
    x=x.cuda()
    print(x.shape)
    #histc_hard
    y = x.histc(bins=100, min=0, max=100)
    y_np = y.detach().cpu().numpy()
    plt.plot(y_np, 'r-',label='hard_histc')

    #histc_soft
    hist_soft = SoftHistogram(bins=100,min=0,max=100,sigma=3)
    y_s = hist_soft(x)
    y_s_np = y_s.detach().cpu().numpy()
    plt.plot(y_s_np,'b-',label="soft histc")


    #draw
    plt.legend()
    plt.title('hitogram1d comparision')
    plt.show()

def test_SoftHistogram2D_H():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img = plt.imread('0000014.png')  # 600,800
    img = img[:,:,0]

    img = torch.tensor(img, dtype=torch.float).to(device)
    img = img.unsqueeze(0)
    img = img.unsqueeze(0) * 255
    b,c,h,w = img.shape


    #hist_soft
    hist_soft = SoftHistogram2D_H(device=device,bins=255,min=0,max=255,sigma=3,b=b,c=c,h=h,w=w)
    out = hist_soft(img)
    outnp = out[0][0].detach().cpu().numpy()

    #hist_hard
    out2 = []
    for i in range(h):
        out2.append(img[0][0][i].histc(bins=255,min=0,max=255).unsqueeze(dim=0))
    out2 = torch.cat(out2,dim=0)
    out2np = out2.detach().cpu().numpy()

    #draw
    plt.subplot(1,2,1)
    plt.imshow(outnp)
    plt.title('soft hist')
    plt.subplot(1,2,2)
    plt.imshow(out2np)
    plt.title('hard hist')
    plt.show()

def test2D_H_time_space():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img = plt.imread('pic128192.png')  # 600,800

    img = torch.tensor(img,dtype=torch.float).to(device)
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)*255

    img2=img+1

    img3 = img+2

    batch = torch.cat([img,img2,img3],dim=0).cuda()



    b,c,h,w = batch.shape
    batch.retain_grad()

    batch2 = torch.ones([4,1,128,192],dtype=torch.float).to(device)
    b,c,h,w = batch2.shape
    batch2.retain_grad()

    func = SoftHistogram2D_H(device=device,bins=100,min=0,max=255,sigma=3,b=b,c=c,h=h,w=w)
    batch.requires_grad_()
    batch.retain_grad()
    i =0
    record = []
    '''
    while i<10:
        st = time()
        out = func(batch2)
        record.append((time()-st)*1000000)
        i+=1
    print(record)
    '''
    out = func(img)
    outnp = out[0][0].detach().cpu().numpy()
    plt.imshow(outnp)
    plt.show()
    print('ov')

if __name__=="__main__":


    test_softhistgram()
