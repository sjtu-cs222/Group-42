import os
import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
 
from reconstruct import ReconstructNet
from reconstruct import imshow as imshow
from reconstruct import getTargets as getTargets
from reconstruct_v2 import ReconstructNet2
from cifar_alex import CifarAlexNet

def resultShow(img1, img2, img3, img4):
    npimg1 = img1.cpu().numpy()
    plt.subplot(221)
    plt.imshow(np.transpose(npimg1, (1, 2, 0)))
    npimg2 = img2.cpu().numpy()
    plt.subplot(222)
    plt.imshow(np.transpose(npimg2, (1, 2, 0)))
    npimg3 = img3.cpu().numpy()
    plt.subplot(223)
    plt.imshow(np.transpose(npimg3, (1, 2, 0)))
    npimg4 = img4.cpu().numpy()
    plt.subplot(224)
    plt.imshow(np.transpose(npimg4, (1, 2, 0)))
    plt.show()

def loadNet(pathToNet):
    st = 0
    res = os.listdir(pathToNet)
    for netFile in res:
        last = int(re.sub("\D","",netFile))
        if last > st:
            st = last
    net = torch.load(pathToNet + "/reconstruct" + str(st) + ".pkl")
    return net

if __name__ == "__main__":
    keepOn = True
    # Prepare the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = True, num_workers = 0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 128, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = torch.device("cuda:0")
    alexnet = torch.load("alex_trained.pkl")
    alexnet.eval()
    alexnet.to(device)
    dataiter = iter(testloader)
    
    net_original = loadNet("./data/exp2").to(device)
    net_prune = loadNet("./data/exp5_prune").to(device)
    net_retrain = loadNet("./data/exp5_retrain").to(device)
    with torch.no_grad():
        for i in range(1, 30):
            images, labels = dataiter.next()
            images = images.to(device)
            targets = getTargets(images)
            res, features = alexnet(images)
            outputs_original = net_original(features)
            outputs_prune = net_prune(features)
            outputs_retrain = net_retrain(features)
            resultShow(targets[8], outputs_original[8], outputs_prune[8], outputs_retrain[8])
            
