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
    
    def generate(self):  
        pass
        #  Need something like z = linspace(-sigma,sigma,...), then self.decode(z)
        #   Want to raster scan the decode function via its inputs, rather than 
        #    sampling randomly. 
        #

    def get_samples(self, mu, logvar):
        Tmu = torch.tensor(mu, dtype=torch.float).to(device)
        Tlogvar = torch.tensor(logvar, dtype=torch.float).to(device)
        z = self.reparameterize(Tmu, Tlogvar)
        sample = model.decode(z)
        return sample


if 'model' not in locals():
    print('new randomly initialized model...\n')
    model = VAE().to(device)

if False: # F9 this to start with a trained model. 
    model.load_state_dict(torch.load('VAEresults100_VAE20190722_1316')) # KTO favorite
    model.load_state_dict(torch.load('VAE20190716_1551')) # WJP favorite

optimizer = optim.Adam(model.parameters(), lr=1e-5)
beta = 0.5

 #this is a away to abreviate some steps
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, beta):
    
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

def display_images(img, nr=8, nc=16, s=28):
    # img is a tensor containing a stack of images, shaped the 
    #  way pytorch does it, i.e. number of examples by number of 
    #  channels by number of rows by number of columns. The MNIST images are
    #  grayscale images, i.e. they have one channel only. 
    #


    #  The default nr, nc, s works for data acquired by forward  
    #    hook from fc4 in class VAE. 
    #

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
    plt.pause(0.05) # makes sure plt flushes its buffer. 
    

def display_bottleneck(axes):
    for batch_idx, (data, which_digit) in enumerate(train_loader):
        break
 

    fc21current,fc22current = model.encode(data.cuda().view(-1,784))

    fc21disp = np.zeros((10,20))
    fc22disp = np.zeros((10,20))


    for i in range(10):
        fc21disp[i,:] = \
        np.mean(fc21current[which_digit==i,:].\
              cpu().detach().numpy(),axis=0)
        fc22disp[i,:] = \
        np.mean(fc22current[which_digit==i,:].\
               cpu().detach().numpy(),axis=0)

   
    axes[0].imshow(fc21disp)
    axes[1].imshow(fc22disp)
    
    plt.pause(0.5)
  

def display_deltas():
#
# This computes and returns the difference vectors between digit
#    encodings. It doesn't yet display anything. 
#

    for batch_idx, (data, which_digit) in enumerate(train_loader):
        break
    
    fc21current,fc22current = model.encode(data.cuda().view(-1,784))
    
    fc21disp = torch.zeros((10,20))
    
    for i in range(10):
        fc21disp[i,:] = \
        torch.mean(fc21current[which_digit==i,:],0)

    delt_disp=np.zeros((10,10,20))  # There a 45 unique 20-vectors 
                                    #  describing the differences
                                    #  between average encodings 
                                    #  for each digit. 
    
    for i in range (10):
        for j in range (i+1,10):
            delt_disp[i,j,:]=(fc21disp[i,:]-fc21disp[j,:]).cpu().detach().numpy()   
    
    return None

def display_means_relationship():
    
#This code displays the means relationship between the means of each handwritten digit. I am attempting 
#    to display the number as a graph and image to see which display provide the most detailed information. 
    plt.clf()
    for batch_idx, (data, which_digit) in enumerate(train_loader):
        break
    fc21current,fc22current = model.encode(data.cuda().view(-1,784))
    
    fc21disp = torch.zeros((10,20))
    
    for i in range(10):
        fc21disp[i,:] = \
        torch.sum(fc21current[which_digit==i,:],0)
    
    disp=torch.zeros(10,1)
    
    for i in range (10):
        disp[i]=torch.mean(fc21disp[i][:])
    
    plt.plot(disp.cpu().detach().numpy(), 'o')

    plt.imshow(disp.cpu().detach().numpy())
    plt.colorbar()
    plt.pause(0.5)
    return 

    

def display_relationship_vector():
#
#  This plots the distance between each digit and every other 
#    pair of digits. Interestingly, these distances are all nearly 
#    the same in model VAE20190716_1551, consistent with the mean
#    encodings being distributed on a 20-sphere. 
#
    for batch_idx, (data, which_digit) in enumerate(train_loader):
        break
    plt.clf()
    fc21current,fc22current = model.encode(data.cuda().view(-1,784))
    
    fc21disp = torch.zeros((10,20))
    
    for i in range(10):
        fc21disp[i,:] = \
        torch.mean(fc21current[which_digit==i,:],0)
        
    mu_dist = torch.pdist(fc21disp)
    
    dist_disp=np.zeros((10,10))
    
    counter=0
    for i in range (10):
        for j in range (i+1,10):
            dist_disp[i,j]=mu_dist[counter]
            counter=counter+1

    plt.plot(dist_disp.transpose())
    plt.pause(0.5)
    return dist_disp
    
    
def display_as_histogram(ax):
    for batch_idx, (data, which_digit) in enumerate(train_loader):
        break
    
    # encode returns two 20-vectors: the means and standard deviations
    #   of normal distributions in the hidden space that generate
    #   each digit when passed to decode. 
    # We want to look at the distribution of these values, rather 
    #   than its mean. Note that here I am lumping all 20 dimensions
    #   into a single histogram for each digit (via hstack); that's
    #   kind of an odd thing to do...just experimenting at this point. 
    fc21current,fc22current = model.encode(data.cuda().view(-1,784))
    
    rc = np.random.uniform(size=3)
    color = (rc[0], rc[1], rc[2], 0.1) #transparency in the histogram
    
    for i in range(3):
        for j in range(4):
            this_digit = i*4 + j
            if this_digit > 9:
                break
            
            data = np.hstack(fc21current[which_digit==this_digit,:].cpu().detach().numpy())
            ax[i,j].hist(data, bins= 10, color=color, density=True)
            ax[i,j].title.set_text(str(this_digit))
            ax[i,j].set_xlim((-3,3))
            ax[i,j].set_ylim((0,0.5))
    

    plt.pause(0.5)

def display_corr():   
    plt.clf()
    
    for batch_idx, (data, which_digit) in enumerate(train_loader):
        break
    
    fc21current,fc22current = model.encode(data.cuda().view(-1,784))

    fc21disp = np.zeros((10,20))
    fc22disp = np.zeros((10,20))

    for i in range(10):
        fc21disp[i,:] = \
        np.mean(fc21current[which_digit==i,:].\
              cpu().detach().numpy(),axis=0)
        fc22disp[i,:] = \
        np.mean(fc22current[which_digit==i,:].\
               cpu().detach().numpy(),axis=0)
    
    rho = np.corrcoef(fc21disp)
    
    for i in range(10):
        for j in range(0,i+1):
            rho[i,j]=0.0
    
    plt.imshow(rho)
    plt.colorbar()
    plt.pause(0.5)
#    plt.show()
#    fig.colorbar()

    return

def display_transition(n0,n1):
    fig,ax = plt.subplots(2,5)
    
    for batch_idx, (data, which_digit) in enumerate(train_loader):
        break
    
    fc21current,fc22current = model.encode(data.cuda().view(-1,784))

    fc21disp = np.zeros((2,20))
#    fc22disp = np.zeros((2,20))

    for i,n in enumerate([n0 ,n1]):
        fc21disp[i,:] = \
        np.mean(fc21current[which_digit==n,:].\
              cpu().detach().numpy(),axis=0)
#        fc22disp[i,:] = \
#        np.mean(fc22current[which_digit==n,:].\
#               cpu().detach().numpy(),axis=0)


    z = torch.zeros((10,20)).to(device)  
    for i in range(10):
        
        alpha = -np.log(11/(i+1)-1);
        z[i,:] = torch.tensor(fc21disp[0,:]*alpha + fc21disp[1,:]*(1-alpha))
        
    xout = model.decode(z).view((10,28,28)).cpu().detach().numpy()
    
    for i in range(2):
        for j in range(5):
            ax[i,j].imshow(xout[i*5+j,:,:])
            
            
    return



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
        loss = loss_function(recon_batch, data, mu, logvar, beta)
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


    torch.save(model.state_dict(),'VAEresults' + str(epoch)+'_VAE' + date_for_filename())


#    display_images(ACQUIRED_DATA)
    

    
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)

            test_loss += loss_function(recon_batch, data, mu, logvar, beta).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)


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
    if "fc2fig" not in locals():
        fc2fig, fc2axes = plt.subplots(2,1)

    if "fc4fig" not in locals():
        fc4fig = plt.figure()
 
    if "hist_fig" not in locals():
        hist_fig, hist_axes = plt.subplots(3,4)

    if "corr_fig" not in locals():
        corr_fig, corr_axes = plt.subplots(1,1)
        
    if "con_fig" not in locals():
        con_fig=plt.figure()
        
    if "num_mean" not in locals():
        num_mean=plt.figure()
        
        
    for epoch in range(1, args.epochs + 1):
        train(epoch)

#        display_as_histogram(fc2axes)
        display_bottleneck(fc2axes)
        plt.figure(fc4fig.number)
        display_images(ACQUIRED_DATA)
        
        display_as_histogram(hist_axes)
        
        plt.figure(fc2fig.number)
        display_bottleneck(fc2axes)
        
        plt.figure(fc4fig.number)
        display_images(ACQUIRED_DATA)
        
        plt.figure(corr_fig.number)
        display_corr()
        
        plt.figure(con_fig.number)
        display_relationship_vector()
        
        plt.figure(num_mean.number)
        display_numerical_relationship()
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'VAEresults/sample_' + str(epoch) + '.png')