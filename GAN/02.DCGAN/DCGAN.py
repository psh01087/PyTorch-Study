import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets

import argparse
import os
import numpy as np
import math

# Make Directory
os.makedirs('result', exist_ok=True)
os.makedirs('mnist', exist_ok=True)
os.makedirs('../../Data/MNIST', exist_ok=True)

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--train_iters', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--latent_dim', type=int, default=100)
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--img_size', type=int, default=28)
parser.add_argument('--sample_step', type=int, default=400)
config = parser.parse_args()
print(config)

img_shape = (config.channels, config.img_size, config.img_size)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()

        self.init_size = config.img_size // 4
        self.deconv1 = deconv(config.latent_dim, 128*8, 4)
        self.deconv2 = deconv(128*8, 128*4, 4)
        self.deconv3 = deconv(128*4, 128*2, 4)
        self.deconv4 = deconv(128*2, 128, 4)
        self.deconv5 = deconv(128, 1, 4, bn=False)

    def forward(self, x):
        out = F.leaky_relu(deconv1(x), 0.2)
        out = F.leaky_relu(deconv2(out), 0.2)
        out = F.leaky_relu(deconv3(out), 0.2)
        out = F.leaky_relu(deconv4(out), 0.2)
        out = F.tanh(deconv5(out))

        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.conv1 = conv(1, 128, 4)
        self.conv2 = conv(128, 128*2, 4)
        self.conv3 = conv(128*2, 128*4, 4)
        self.conv4 = conv(128*4, 128*8, 4)
        self.conv5 = conv(128*8, 1, 4, bn=False)

    def forward(self, x):
        out = F.leaky_relu(conv1(x), 0.2)
        out = F.leaky_relu(conv2(out), 0.2)
        out = F.leaky_relu(conv3(out), 0.2)
        out = F.leaky_relu(conv4(out), 0.2)
        out = F.sigmoid(conv5(out))

        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


# Loss Function
# Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
Loss_function = nn.BCELoss()

# Initailize Generator and Discriminator
G = G()
D = D()
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)

if cuda:
    G.cuda()
    D.cuda()
    Loss_function.cuda()

# Data Loader
dataloader = DataLoader(datasets.MNIST('../../Data/MNIST', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Resize(config.img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                        batch_size=config.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

#============ Train ============#
for epoch in range(config.train_iters):
    for i, (imgs, _) in enumerate(dataloader):
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
        real_imgs = Variable(imgs.type(Tensor))

        #============ Train Generator ============#
        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], config.latent_dim))))

        # Generate a batch of images
        gen_imgs = G(z)
        # Generator Loss
        loss_G = Loss_function(D(gen_imgs), valid)

        loss_G.backward()
        optimizer_G.step()

        #============ Train Discriminator ============#
        optimizer_D.zero_grad()

        # Discrimnator Loss
        real_loss = Loss_function(D(real_imgs), valid)
        fake_loss = Loss_function(D(gen_imgs.detach()), fake)
        loss_D = (real_loss + fake_loss) / 2

        loss_D.backward()
        optimizer_D.step()

        if i % 100 == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [D loss : %f] [G loss : %f]" % (epoch, config.train_iters, i, len(dataloader), loss_D.item(), loss_G.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % config.sample_step == 0:
            save_image(gen_imgs.data[:25], 'result/%d.png' % batches_done, nrow=5, normalize=True)
