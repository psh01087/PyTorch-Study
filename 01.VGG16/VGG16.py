import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import transforms
from torchvision import datasets

import argparse
import os
import numpy as np

# Make Directory
os.makedirs('result', exist_ok=True)
os.makedirs('model', exist_ok=True)
os.makedirs('../Data/CIFAL10', exist_ok=True)

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--train_iters', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
config = parser.parse_args()
print(config)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

#transform = transforms.ToTensor()

# Data Loader
trainData = datasets.CIFAR10(root="../Data/CIFAR10",train=True,transform=transform,download=True)
trainLoader = DataLoader(trainData,batch_size=config.batch_size,shuffle=True)

testData = datasets.CIFAR10(root="../Data/CIFAR10", train=False, transform=transform)
testLoader = DataLoader(testData, batch_size=config.batch_size, shuffle=False)


def conv(c_in, c_out, k_size, stride=1, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layer1 = nn.Sequential(
            conv(3, 64, 3),
            conv(64, 64, 3),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            conv(64, 128, 3),
            conv(128, 128, 3),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            conv(128, 256, 3),
            conv(256, 256, 3),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(
            conv(256, 512, 3),
            conv(512, 512, 3),
            conv(512, 512, 3),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = nn.Sequential(
            conv(512, 512, 3),
            conv(512, 512, 3),
            conv(512, 512, 3),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.FC1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU())

        self.FC2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU())

        self.FC3 = nn.Sequential(
            nn.Linear(128, 10),
            nn.BatchNorm1d(10),
            nn.Softmax())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        out = out.view(out.size(0), -1)

        out = self.FC1(out)
        out = self.FC2(out)
        out = self.FC3(out)

        return out


# Loss Function
Loss_function = nn.CrossEntropyLoss()

vgg16 = VGG16()

if cuda:
    vgg16.cuda()
    Loss_function.cuda()


# Optimizer
optimizer = torch.optim.Adam(vgg16.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))


# Train the model
for epoch in range(config.train_iters):
    for i, (images, labels) in enumerate(trainLoader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = vgg16(images)
        loss = Loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss %.4f'
                  % (epoch+1, config.train_iters, i, len(trainLoader), loss.item()))

        if epoch % 10 == 10:
            torch.save(vgg16, 'model/vgg16_%d.pth' % epoch)


# Test the model
vgg16.eval()
correct = 0
total = 0

for images, labels in testLoader:
    images = Variable(images).cuda()
    outputs = vgg16(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Test Accuracy : %d %%' % (100 * correct / total))
