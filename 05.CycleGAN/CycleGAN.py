import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image

import argparse
import os
import numpy as np
import math
import scipy.io

def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""

    transform = transforms.Compose([
                    transforms.Scale(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    svhn = datasets.SVHN(root=config.svhn_path, download=True, transform=transform)
    mnist = datasets.MNIST(root=config.mnist_path, download=True, transform=transform)

    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)
    return svhn_loader, mnist_loader

def str2bool(v):
    return v.lower() in ('true')

def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def merge_images(sources, targets, k=10, batch_size=64):
    _, _, h, w = sources.shape
    row = int(np.sqrt(batch_size))
    merged = np.zeros([3, row*h, row*w*2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
        merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
    return merged.transpose(1, 2, 0)

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

class G12(nn.Module):
    """Generator for transfering from mnist to svhn"""
    def __init__(self, conv_dim=64):
        super(G12, self).__init__()
        # encoding blocks
        self.conv1 = conv(1, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)

        # residual blocks
        self.conv3 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.conv4 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)

        # decoding blocks
        self.deconv1 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)      # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)    # (?, 128, 8, 8)

        out = F.leaky_relu(self.conv3(out), 0.05)    # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)    # ( " )

        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = F.tanh(self.deconv2(out))              # (?, 3, 32, 32)
        return out

class G21(nn.Module):
    """Generator for transfering from svhn to mnist"""
    def __init__(self, conv_dim=64):
        super(G21, self).__init__()
        # encoding blocks
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)

        # residual blocks
        self.conv3 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.conv4 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)

        # decoding blocks
        self.deconv1 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 1, 4, bn=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)      # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)    # (?, 128, 8, 8)

        out = F.leaky_relu(self.conv3(out), 0.05)    # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)    # ( " )

        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = F.tanh(self.deconv2(out))              # (?, 1, 32, 32)
        return out

class D1(nn.Module):
    """Discriminator for mnist."""
    def __init__(self, conv_dim=64, use_labels=False):
        super(D1, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim*4, n_out, 4, 1, 0, False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out

class D2(nn.Module):
    """Discriminator for svhn."""
    def __init__(self, conv_dim=64, use_labels=False):
        super(D2, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim*4, n_out, 4, 1, 0, False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out


if __name__ == '__main__':
    # Make Directory
    os.makedirs('result', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    os.makedirs('mnist', exist_ok=True)
    os.makedirs('svhn', exist_ok=True)

    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnist_path', type=str, default='./mnist')
    parser.add_argument('--svhn_path', type=str, default='./svhn')
    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--model_path', type=str, default='./model')

    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--train_iters', type=int, default=40000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--use_reconst_loss', required=True, type=str2bool)
    config = parser.parse_args()
    print(config)


    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Modeling
    G12 = G12()
    G21 = G21()
    D1 = D1()
    D2 = D2()

    # Optimizer & Loss
    g_params = list(G12.parameters()) + list(G21.parameters())
    d_params = list(D1.parameters()) + list(D2.parameters())
    G_optimizer = optim.Adam(g_params, config.lr, [config.beta1, config.beta2])
    D_optimizer = optim.Adam(d_params, config.lr, [config.beta1, config.beta2])
    loss = nn.CrossEntropyLoss()

    if cuda:
        G12.cuda()
        G21.cuda()
        D1.cuda()
        D2.cuda()

    # Data Loader
    svhn_loader, mnist_loader = get_loader(config)

    # Train
    svhn_iter = iter(svhn_loader)
    mnist_iter = iter(mnist_loader)
    iter_per_epoch = min(len(svhn_iter), len(mnist_iter))

    fixed_svhn = to_var(svhn_iter.next()[0])
    fixed_mnist = to_var(mnist_iter.next()[0])

    for epoch in range(config.train_iters+1):
        # reset data_iter for each iter_per_epoch
        if (epoch+1) % iter_per_epoch == 0:
            svhn_iter = iter(svhn_loader)
            mnist_iter = iter(mnist_loader)

        # Load svhn and mnist dataset
        svhn, s_labels = svhn_iter.next()
        svhn, s_labels = to_var(svhn), to_var(s_labels).long().squeeze()
        mnist, m_labels = mnist_iter.next()
        mnist, m_labels = to_var(mnist), to_var(m_labels)

        svhn_fake_labels = to_var(torch.Tensor([config.num_classes]*mnist.size(0)).long())
        mnist_fake_labels = to_var(torch.Tensor([config.num_classes]*svhn.size(0)).long())

        #============ train D ============#
        # train with real images
        G_optimizer.zero_grad()
        D_optimizer.zero_grad()
        out = D1(mnist)
        #d1_loss = loss(out, m_labels)
        d1_loss = torch.mean((out-1)**2)

        out = D2(svhn)
        #d2_loss = loss(out, s_labels)
        d2_loss = torch.mean((out-1)**2)
        d_mnist_loss = d1_loss
        d_svhn_loss = d2_loss
        d_real_loss = d1_loss + d2_loss
        d_real_loss.backward()
        D_optimizer.step()

        # train with fake images
        G_optimizer.zero_grad()
        D_optimizer.zero_grad()
        fake_mnist = G21(svhn)
        out = D1(fake_mnist)
        #d1_loss = loss(out, mnist_fake_labels)
        d1_loss = torch.mean(out**2)

        fake_svhn = G12(mnist)
        out = D2(fake_svhn)
        #d2_loss = loss(out, svhn_fake_labels)
        d2_loss = torch.mean(out**2)

        d_fake_loss = d1_loss + d2_loss
        d_fake_loss.backward()
        D_optimizer.step()

        #============ train G ============#
        # train mnist-svhn-mnist cycle
        G_optimizer.zero_grad()
        D_optimizer.zero_grad()
        fake_svhn = G12(mnist)
        out = D2(fake_svhn)
        reconst_mnist = G21(fake_svhn)
        #g_loss = loss(out, m_labels)
        g_loss = torch.mean((out-1)**2)

        if config.use_reconst_loss:
            g_loss += torch.mean((mnist - reconst_mnist)**2)

        g_loss.backward()
        G_optimizer.step()

        # train svhn-mnist-svhn cycle
        G_optimizer.zero_grad()
        D_optimizer.zero_grad()
        fake_mnist = G21(svhn)
        out = D1(fake_mnist)
        reconst_svhn = G12(fake_mnist)
        #g_loss = loss(out, s_labels)
        g_loss = torch.mean((out-1)**2)

        if config.use_reconst_loss:
            g_loss += torch.mean((svhn - reconst_svhn)**2)

        g_loss.backward()
        G_optimizer.step()

        # print Log info
        if (epoch+1) % 10 == 0:
            print('Step [%d/%d], d_real_loss: %.4f, d_mnist_loss: %.4f, d_svhn_loss: %.4f, '
                      'd_fake_loss: %.4f, g_loss: %.4f'
                      %(epoch+1, config.train_iters, d_real_loss.item(), d_mnist_loss.item(), d_svhn_loss.item(), d_fake_loss.item(), g_loss.item()))

        # save the sampled images
        if (epoch+1) % 500 == 0:
            fake_svhn = G12(fixed_mnist)
            fake_mnist = G21(fixed_svhn)

            mnist, fake_mnist = to_data(fixed_mnist), to_data(fake_mnist)
            svhn, fake_svhn = to_data(fixed_svhn), to_data(fake_svhn)

            merged = merge_images(mnist, fake_svhn, batch_size=config.batch_size)
            path = os.path.join(config.result_path, 'sample-%d-m-s.png' %(epoch+1))
            scipy.misc.imsave(path, merged)
            print('saved %s' %path)

            merged = merge_images(svhn, fake_mnist, batch_size=config.batch_size)
            path = os.path.join(config.result_path, 'sample-%d-s-m.png' %(epoch+1))
            scipy.misc.imsave(path, merged)
            print('saved %s' %path)

        if (epoch+1) % 5000 == 0:
            # save the model parameters for each epoch
            g12_path = os.path.join(config.model_path, 'g12-%d.pkl' %(epoch+1))
            g21_path = os.path.join(config.model_path, 'g21-%d.pkl' %(epoch+1))
            d1_path = os.path.join(config.model_path, 'd1-%d.pkl' %(epoch+1))
            d2_path = os.path.join(config.model_path, 'd2-%d.pkl' %(epoch+1))

            torch.save(G12.state_dict(), g12_path)
            torch.save(G21.state_dict(), g21_path)
            torch.save(D1.state_dict(), d1_path)
            torch.save(D2.state_dict(), d2_path)
