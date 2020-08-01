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


f = open('mapping/1/captions.txt','r')

corpus=''
for i in f.readlines():
    corpus+=i.split('\t')[1].lower()

unique = list(set(corpus.replace('.',' ').replace('\n',' ').replace(',',' ').split()))
L = len(unique)
d = {}
for i in range(L):
    d[unique[i]] = i



corpus = corpus.split('.\n')
mat = np.zeros((L,L))

win=3

for sentence in corpus:
    curr = sentence.replace('.',' ').replace('\n',' ').replace(',',' ').split()
    for word_index in range(len(curr)):
        for adj in range(max(0,word_index-win),min(len(curr)-1,word_index+win)):
            mat[d[curr[word_index]]][d[curr[adj]]]+=1
mat[range(L),range(L)] = 0

probmat = mat/np.sum(mat,axis=1).reshape((-1,1)).astype('float64')


'''
c = np.argmax(mat,axis=0)
for i in range(20):
    print (unique[i],unique[c[i]])
'''



W = Variable(torch.randn(L, L).type(torch.float64), requires_grad=True)
B1 = Variable(torch.randn(1, L).type(torch.float64), requires_grad=True)
B2 = Variable(torch.randn(L, 1).type(torch.float64), requires_grad=True)



op = torch.from_numpy(np.clip(np.log(probmat),0,9999))

optimizer = torch.optim.Adam([W,B1,B2], lr=0.01)

print(W)
t=time.time()
for i in range(20):
    pred = torch.mm(W,torch.t(W))+B1+B2
    loss = (pred-op).pow(2).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print (i)
print(time.time()-t)    
print(W)

f = open('GLOVE_REP','wb')
p.dump((W,B1,B2),f)
f.close()
