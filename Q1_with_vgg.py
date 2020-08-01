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
glove["endseq"] = np.zeros(100)

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
        traindata.append(cv2.resize(img,(224,224),interpolation = cv2.INTER_AREA))
        trainlabels.append(caption.replace('.',' ').replace(',',' ').replace('\n',' ')+" endseq")
    print (ctr)
    ctr+=1
    if ctr >300:
        break

device = 'cuda'

Z = len(traindata)
traindata = np.array(traindata).transpose(0,3,1,2).astype('float64')/255.0
traindata,devdata = traindata[:int(9*Z/10)],traindata[int(9*Z/10):]
trainlabels,devlabels = trainlabels[:int(9*Z/10)],trainlabels[int(9*Z/10):]

vgg16 = models.vgg16(pretrained=True)
modules=list(vgg16.children())[:-1]
pre_trained_net=nn.Sequential(*modules).to(device)



class captioner(nn.Module):
    def __init__(self):
        super(captioner, self).__init__()
        K=self.K=4

        self.H = 500
        
        self.rnn = nn.RNNCell(512, self.H)

        self.out = nn.Linear(self.H,100)

    def forward(self, x):

        K=self.K
        x = pre_trained_net(x)
        x = x.permute(0,2,3,1)
        x = x.view(x.shape[1]*x.shape[2],1,x.shape[3])
        
        op=[]
        hidden = torch.rand((1,self.H)).to(device)
        for i in range(49):
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

TRAIN=0
if TRAIN:
    for epoch in range(10):
        td,tl = traindata,trainlabels
        trainloss=0
        for i in range(0,len(traindata),batch_size):
            inputs = torch.from_numpy(td[i:i+batch_size,:]).float().to(device)
            labels=[]
            k=0
            wrd = tl[i].split()
            while k < len(wrd) and k<49:
                
                if wrd[k].lower() in glove:
                    labels.append(glove[wrd[k].lower()])
                else:
                    print(wrd[k])
                k+=1
            labels = np.array(labels)
            paddedlabels = np.zeros((1,49,labels.shape[1]))
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
    torch.save(net, "./Model/Q1_vgg.pth")
else:
    net = torch.load("./Model/Q1_vgg.pth")

    

print("testing")
op = net(torch.from_numpy(traindata[6:7]).float().to(device)).detach().cpu().numpy()
s=''
for i in op[0]:
    index = np.argmin(np.linalg.norm(allvecs - i,axis=1))
    if wordlist[index]=='endseq':
        break
    s+=wordlist[index]+' '
print(s)
print(trainlabels[6])
score = nltk.translate.bleu_score.sentence_bleu([trainlabels[6].split()], s.split())
print(score)

op = net(torch.from_numpy(devdata[6:7]).float().to(device)).detach().cpu().numpy()
s=''
for i in op[0]:
    index = np.argmin(np.linalg.norm(allvecs - i,axis=1))
    if wordlist[index]=='endseq':
        break
    s+=wordlist[index]+' '
print(s)
print(devlabels[6])
score = nltk.translate.bleu_score.sentence_bleu([devlabels[6].split()], s.split())
print(score)
#cv2.imshow('1',devdata[0].transpose(1,2,0))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
