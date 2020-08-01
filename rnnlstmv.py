import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np
import cv2

# Prepare the GloVe mapping:
GLOVE_DIR ='glove.6B'
embeddings_index = {}
file = open('eng_glove/glove.6B.50d.txt', encoding = 'utf8')
for line in file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
file.close()


# Prepare the Data set:
corpus = list(embeddings_index.keys())
corpus.sort()
mapping = {}
for idx, word in enumerate(corpus):
    mapping[word] = idx
TXT_DIR = 'mapping/1'
IMG_DIR = 'Images'
CNT = 50
file = open('mapping/1/captions.txt', encoding = 'utf8')
traindata = []
trainlabels = []
for i, line in enumerate(file):

    # Loading the images and stacking them as np arrays:
    image_name = line.split()[0][:-2]
    img = cv2.resize(cv2.imread(os.path.join(IMG_DIR, image_name)),(24,24),interpolation = cv2.INTER_AREA)
    img = img/255.0
    traindata.append(img)
    
    # Loading the corresponding captions and storing the captions as np arrays:

    train_sentence = ' '.join(line.split()[1:-1])+' berlin'
    label_vector = []
    for j, word in enumerate(train_sentence.split()):
        key = word.replace('-', ' ').replace(',', ' ').replace('\n', ' ')
        if key.lower() in mapping.keys():
            label_vector.append(mapping[key.lower()])
    trainlabels.append(label_vector)

    # Termination:
    if i+1 == CNT:
        break

print(len(traindata))
print(len(trainlabels))
file.close()


# The Recurrent Neural Network with LSTM:
device = 'cuda'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.K=12
        K=self.K
        self.C_1 = nn.Conv2d(3, 4, kernel_size=3, stride=1)
        self.B_1 = nn.BatchNorm2d(4)
        self.P_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.agg = nn.Conv2d(4,K,kernel_size=1, stride=1)

        self.glovesize = 50
        self.fc_inter = nn.Linear(4*K, self.glovesize)
        self.H = 50
        
        self.rnn = nn.LSTMCell(self.glovesize, self.H)
        #self.rnn = nn.RNNCell(self.glovesize, self.H)

        self.out = nn.Linear(self.H, len(corpus))

    def forward(self, x, oplen):

        K=self.K
        
        x = self.P_1(torch.relu(self.B_1(self.C_1(x))))
        
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
        c = torch.rand((1,self.H)).to(device)
        hidden = torch.rand((1,self.H)).to(device)
        out=x
        for i in range(oplen):
            
            hidden,c = self.rnn(out,(hidden,c))
            y=torch.exp(self.out(hidden))
            out = y/torch.sum(y)
            
            op.append(out)
            argm = torch.argmax(out[0]).item()
            
            out = torch.from_numpy(embeddings_index[corpus[argm]].reshape((1, self.glovesize))).to(device)
        
        x = torch.stack(op,dim=0).squeeze().unsqueeze(0)
        return x

# Training:
net = Net().cuda()

losstype = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

batch_size=1
bestep=0

TRAIN = 1

if TRAIN:
    for epoch in range(3):
        td,tl = np.array(traindata).transpose(0, 3, 1, 2),trainlabels
        trainloss=0
        for i in range(0,len(traindata),batch_size):
            inputs = torch.from_numpy(td[i:i+batch_size]).float().to(device)
            labels = tl[i:i+batch_size]
            labels = np.array(labels)
            finlabel = torch.from_numpy(labels[0]).long().to(device)
            optimizer.zero_grad()

            outputs = net(inputs, labels.shape[1])

            loss = losstype(outputs[0], finlabel)
            loss.backward()
            optimizer.step()
            trainloss+=loss
            if i%10==0:
                print (i)

        print("epoch : ",epoch)
        print("train loss : ",trainloss)
    torch.save(net, "./Model/Q1_1.pth")
else:
    net = torch.load("./Model/Q1_1.pth")

# Imperfect Testing:
print("testing")

n = 14
op = net(torch.from_numpy(np.array(traindata).transpose(0, 3, 1, 2)[n:n+1]).float().to(device),30).detach().cpu().numpy()
s=''
for i in op[0]:
    index = np.argmax(i)
    print(corpus[index], end = ' ')
print()
for idx in trainlabels[n]:
    print(corpus[idx], end = ' ')
print()

