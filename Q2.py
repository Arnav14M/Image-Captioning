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
import nltk

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
        traindata.append(cv2.resize(img,(24,24),interpolation = cv2.INTER_AREA))
        #trainlabels.append(caption.replace('.',' ').replace(',',' ').replace('\n',' ')+" #end#")
        trainlabels.append(caption.replace('.',' ').replace(',',' ').replace('\n',' '))
    print (ctr)
    ctr+=1
    if ctr >50:
        break

Z = len(traindata)
traindata = np.array(traindata).transpose(0,3,1,2).astype('float64')/255.0
traindata,devdata = traindata[:int(9*Z/10)],traindata[int(9*Z/10):]
trainlabels,devlabels = trainlabels[:int(9*Z/10)],trainlabels[int(9*Z/10):]

device = 'cuda'

class captioner(nn.Module):
    def __init__(self):
        super(captioner, self).__init__()
        self.K=12
        K=self.K
        self.C_1 = nn.Conv2d(3, 4, kernel_size=3, stride=1)
        self.B_1 = nn.BatchNorm2d(4)
        self.P_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.C_2 = nn.Conv2d(4,16,kernel_size=3, stride=1)
        self.B_2 = nn.BatchNorm2d(16)
        self.P_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.agg = nn.Conv2d(16,K,kernel_size=1, stride=1)

        self.fc_inter = nn.Linear(16*K,100)
        self.H = 50
        
        self.rnn = nn.LSTMCell(100, self.H)

        self.out = nn.Linear(self.H,100)

    def forward(self, x, oplen):

        K=self.K
        
        x = self.P_1(torch.relu(self.B_1(self.C_1(x))))
        x = self.P_2(torch.relu(F.dropout(self.B_2(self.C_2(x)),p=.5)))
        
        vlad = torch.zeros([x.shape[0], K, x.shape[1]], dtype=x.dtype, layout=x.layout, device=x.device)
        x_flat = x.view(x.shape[0],x.shape[1],-1)
        a = F.softmax(self.agg(x).view(x.shape[0],K,-1), dim=1)
        centres = self.agg.weight.squeeze()/(2*.1) # c = w/2B
        cmod = centres.expand(x_flat.size(-1), -1, -1).permute(1,2,0)
        for k in range(K):
            residual = x_flat-cmod[k]
            residual = residual*a[:,k:k+1,:]
            vlad[:,k,:] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)
        
        x = vlad.view(1,-1)
        x = self.fc_inter(x)
        
        op=[]
        hidden = torch.rand((1,self.H)).to(device)
        c = torch.rand((1,self.H)).to(device)
        out=x
        for i in range(oplen):
            #print(hidden.shape,out.shape)
            hidden,c = self.rnn(out,(hidden,c))
            out = self.out(hidden)
            op.append(torch.tanh(out))
        x = torch.stack(op,dim=0).squeeze().unsqueeze(0)
        return x



net = captioner().cuda()

losstype = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

batch_size=1
bestep=0

TRAIN = 0

if TRAIN:
    for epoch in range(10):
        td,tl = traindata,trainlabels
        trainloss=0
        for i in range(0,len(traindata),batch_size):
            inputs = torch.from_numpy(td[i:i+batch_size,:]).float().to(device)
            labels=[]
            k=0
            wrd = tl[i].split()
            while k < len(wrd) and k<20:
                
                if wrd[k].lower() in glove:
                    labels.append(glove[wrd[k].lower()])
                else:
                    print(wrd[k])
                k+=1
            labels = np.array(labels)
            #paddedlabels = np.zeros((1,100,labels.shape[1]))
            #paddedlabels[0,:labels.shape[0],:] = labels
            #finlabel = torch.from_numpy(paddedlabels).float().to(device)
            finlabel = torch.from_numpy(labels).float().to(device)
            optimizer.zero_grad()
            outputs = net(inputs,len(labels))
            loss = losstype(outputs[0], finlabel)
            loss.backward()
            optimizer.step()
            trainloss+=loss
            if i%10==0:
                print (i)

        print("epoch : ",epoch)
        print("train loss : ",trainloss)
    torch.save(net, "./Model/Q2_1.pth")
else:
    net = torch.load("./Model/Q2_1.pth")

print("testing")

n=12
op = net(torch.from_numpy(traindata[n:n+1]).float().to(device),30).detach().cpu().numpy()
s=''
for i in op[0]:
    index = np.argmin(np.linalg.norm(allvecs - i,axis=1))
    if wordlist[index]=='#end#':
        break
    s+=wordlist[index]+' '
print(s)
print(trainlabels[n])
score = nltk.translate.bleu_score.sentence_bleu([trainlabels[0].split()], s.split())
#cv2.imshow('1',traindata[0].reshape(24,24,3))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
