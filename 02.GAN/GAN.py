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

def layer(feat_in, feat_out, bn=True):
    """Custom Linear layer for simplicity."""
    layers = []
    layers.append(nn.Linear(feat_in, feat_out))
    if bn:
        layers.append(nn.BatchNorm1d(feat_out, 0.8))
    return nn.Sequential(*layers)


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()

        self.layer1 = layer(config.latent_dim, 128, bn=False)
        self.layer2 = layer(128, 256)
        self.layer3 = layer(256, 512)
        self.layer4 = layer(512, 1024)
        self.FC = nn.Linear(1024, int(np.prod(img_shape)))

    def forward(self, x):
        out = F.leaky_relu(self.layer1(x), 0.2)
        out = F.leaky_relu(self.layer2(out), 0.2)
        out = F.leaky_relu(self.layer3(out), 0.2)
        out = F.leaky_relu(self.layer4(out), 0.2)
        out = F.tanh(self.FC(out))

        out = out.view(out.size(0), *img_shape)

        return out

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

        self.conv1 = layer(int(np.prod(img_shape)), 512)
        self.conv2 = layer(512, 256)
        self.FC = nn.Linear(256, 1)

    def forward(self, x):
        out = x.view(x.size(0), -1)

        out = F.leaky_relu(self.conv1(out), 0.2)
        out = F.leaky_relu(self.conv2(out), 0.2)
        out = F.sigmoid(self.FC(out))

        return out

# Loss Function
# Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
Loss_function = nn.BCELoss()

# Initailize Generator and Discriminator
G = G()
D = D()

if cuda:
    G.cuda()
    D.cuda()
    Loss_function.cuda()

# Data Loader
dataloader = DataLoader(datasets.MNIST('mnist', train=True, download=True,
                        transform=transforms.Compose([
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
