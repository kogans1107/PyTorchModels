#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:19:05 2019

@author: karrington
"""

import matplotlib.pyplot as plt
import argparse
import itertools
import torch
import numpy as np
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import time


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):  # this sets up 5 linear layers
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400) # stacked MNIST to 400
        self.fc21 = nn.Linear(400, 20) # two hidden low D
        self.fc22 = nn.Linear(400, 20) # layers, same size
        self.fc3 = nn.Linear(20, 400)  
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x): 
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) # <- Stochasticity!!!
        # How can the previous line allow back propagation?
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

if 'model' not in locals():
    model = VAE().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    
    beta = 0.01
    
    
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD



def acquire_data_hook(self, input_tuple, output_tensor):    # Now THIS is a hook, again, look at the arguments;  a hook must have these. 
   global ACQUIRED_DATA
   ACQUIRED_DATA=output_tensor




acq_hook_handle = model.fc4.register_forward_hook(acquire_data_hook)

def display_images(img):
    # img is a tensor containing a stack of images, shaped the 
    #  way pytorch does it, i.e. number of examples by number of 
    #  channels by number of rows by number of columns. The MNIST images are
    #  grayscale images, i.e. they have one channel only. 
    #
    nr=8 #nr-number of rows we're reshaping the image pixels 
    nc=16 #nc=number of colums we're reshaping the image pixels
    s=28 #side of a square (28by28)
    img_np=img.cpu().detach().numpy()
    new_img=np.reshape(img_np, (nr*nc,s,s))
    disp=np.zeros((nr*s,nc*s))
  
    for i in range(nr):
        for j in range(nc):
                read_data = int(nc*i+j)
                r0=i*s
                c0=j*s
                disp[r0:(r0+s),c0:(c0+s)]=new_img[read_data,:,:]
    plt.imshow(disp)
    plt.pause(0.05) 
  
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        if batch_idx > 467: #last bactch only has 96 examples (#468)
            break
        data = data.to(device)
        optimizer.zero_grad()
        # For a given data object, mu and logvar are fixed, but
        #   recon_batch has stochasticity. 
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

#        if batch_idx % args.log_interval == 0:
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, batch_idx * len(data), len(train_loader.dataset),
#                100. * batch_idx / len(train_loader),
#                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    display_images(ACQUIRED_DATA)
    torch.save(model.state_dict(),'VAEresults/sample_' + str(epoch)+'_' + date_for_filename())

    
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))

def date_for_filename():
    tgt = time.localtime()
    year = str(tgt.tm_year)
    mon = "{:02}".format(tgt.tm_mon)
    day = "{:02}".format(tgt.tm_mday)
    hour = "{:02}".format(tgt.tm_hour)
    minute = "{:02}".format(tgt.tm_min)
    datestr = year + mon + day + '_' + hour + minute
    return datestr




if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'VAEresults/sample_' + str(epoch) + '.png')