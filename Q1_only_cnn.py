import numpy as np
import random
import cv2
from matplotlib import pyplot as plt
from scipy import stats
import glob
import pickle as p
from sklearn.svm import SVC
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torchvision.models as models
from torch.autograd import Variable

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

glove = {}
'''
f = open('glove.6b.100d.txt','r',encoding="utf8")
for x in f.readlines():
    sp = x.split()
    word = str(sp[0])
    vec = np.array(sp[1:]).astype('float64')
    glove[word] = vec.copy()
f.close()

f = open('glovedict','wb')
p.dump(glove,f)
f.close()
'''

f = open('glovedict','rb')
glove = p.load(f)
f.close()
glove["#end#"] = np.zeros(100)

allvecs = []
wordlist = list(glove.keys())
wordlist.sort()
for i in wordlist:
    allvecs.append(glove[i])
allvecs = np.array(allvecs).astype('float64')

print("loading data")

f = open('mapping/1/captions.txt','r')
lines = f.readlines()
traindata = []
trainlabels = []
ctr=1
print(len(lines))
for line in lines:
    imgname = line.split('#')[0]
    caption = line.split('\t')[1]
    img = cv2.imread('Images/'+imgname)
    if img is not None:
        traindata.append(cv2.resize(img,(26,26),interpolation = cv2.INTER_AREA))
        trainlabels.append("startseq "+caption.replace('.',' ').replace(',',' ').replace('\n',' ')+" endseq")
    print (ctr)
    ctr+=1
    if ctr >100:
        break

Z = len(traindata)
traindata = np.array(traindata).reshape(-1,3,26,26).astype('float64')/255.0
traindata,devdata = traindata[:int(9*Z/10)],traindata[int(9*Z/10):]
trainlabels,devlabels = trainlabels[:int(9*Z/10)],trainlabels[int(9*Z/10):]

device = 'cuda'

class captioner(nn.Module):
    def __init__(self):
        super(captioner, self).__init__()
        self.K=4
        K=self.K
        self.C_1 = nn.Conv2d(3, 4, kernel_size=3, stride=1)
        self.B_1 = nn.BatchNorm2d(4)
        self.P_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.C_2 = nn.Conv2d(4,16,kernel_size=3, stride=1)
        self.B_2 = nn.BatchNorm2d(16)
        self.P_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.agg = nn.Conv2d(16,K,kernel_size=1, stride=1)

        self.H = 25
        
        self.rnn = nn.RNNCell(16, self.H)

        self.out = nn.Linear(self.H,100)

    def forward(self, x):

        K=self.K
        
        x = self.P_1(torch.relu(self.B_1(self.C_1(x))))
        x = self.P_2(torch.relu(F.dropout(self.B_2(self.C_2(x)),p=.5)))
        x = x.permute(0,2,3,1)
        x = x.view(x.shape[1]*x.shape[2],1,x.shape[3])
        
        op=[]
        hidden = torch.rand((1,self.H)).to(device)
        for i in range(25):
            hidden = self.rnn(x[i],hidden)
            out = self.out(hidden)
            op.append(torch.tanh(out))
        x = torch.stack(op,dim=0).squeeze().unsqueeze(0)
        return x



net = captioner().cuda()

losstype = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

batch_size=1
bestep=0
for epoch in range(5):
    td,tl = traindata,trainlabels
    trainloss=0
    for i in range(0,len(traindata),batch_size):
        inputs = torch.from_numpy(td[i:i+batch_size,:]).float().to(device)
        labels=[]
        k=0
        wrd = tl[i].split()
        while k < len(wrd) and k<25:
            
            if wrd[k].lower() in glove:
                labels.append(glove[wrd[k].lower()])
            else:
                print(wrd[k])
            k+=1
        labels = np.array(labels)
        paddedlabels = np.zeros((1,25,labels.shape[1]))
        paddedlabels[0,:labels.shape[0],:] = labels
        finlabel = torch.from_numpy(paddedlabels).float().to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = losstype(outputs, finlabel)
        loss.backward()
        optimizer.step()
        trainloss+=loss
        if i%10==0:
            print (i)

    print("epoch : ",epoch)
    print("train loss : ",trainloss)

print("testing")
op = net(torch.from_numpy(traindata[0:1]).float().to(device)).detach().cpu().numpy()
s=''
for i in op[0]:
    index = np.argmin(np.linalg.norm(allvecs - i,axis=1))
    if wordlist[index]=='endseq':
        break
    s+=wordlist[index]+' '
print(s)
print(trainlabels[0])
cv2.imshow('1',traindata[0].reshape(26,26,3))
cv2.waitKey(0)
cv2.destroyAllWindows()
