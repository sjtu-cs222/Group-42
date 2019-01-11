from __future__ import print_function
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class CifarAlexNet(nn.Module):
    def __init__(self):
        super(CifarAlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 96, 3, padding = 1)
        self.conv4 = nn.Conv2d(96, 96, 3, padding = 1)
        self.conv5 = nn.Conv2d(96, 128, 3, padding= 1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)
        self.dropout = nn.Dropout(p = 0.5)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # y = x # that is reconstruct_v2
        x = self.pool(F.relu(self.conv5(x)))
        y = x # that is reconstruct
        x = self.dropout(x.view(-1, 128 * 4 * 4))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x, y

# Train the alexnet(simplified for cifar)
if __name__ == "__main__":
    keepOn = False
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
    net = CifarAlexNet()
    net.to(device)
    crit = nn.CrossEntropyLoss()
    learningRate = [0.005 for i in range(40)]
    learningRate.extend(0.0005 for i in range(20))
    learningRate.extend(0.0001 for i in range(10))
    # Record the performance
    train_loss = []
    train_accu = []
    test_loss = []
    test_accu = []
    x_axis = []
    start = 0
    if keepOn:
        res = os.listdir("./data/exp")
        start = len(res)
        net = torch.load("./data/exp/alex"+str(start)+".pkl")
    for epoch in range(start,70):
        x_axis.append(epoch + 1)
        optimizer = optim.SGD(net.parameters(), lr = learningRate[epoch], momentum = 0.9)
        correct = 0
        total = 0
        accu_loss = 0
        batchNum = 0

        # Train
        for i, data in enumerate(trainloader, 0):
            batchNum += 1
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, features = net(inputs)
            # Update parameters
            loss = crit(outputs, labels)
            loss.backward()
            optimizer.step()
            # Calculate the performance
            outputs, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            accu_loss += loss.item()
            total += labels.size(0)
            accuracy = correct / total
            print('[train] epoch: %2d, batch: %3d, loss: %.3f, accuracy: %.3f'\
                  % (epoch + 1, i + 1, accu_loss / (i+1), accuracy))
        train_loss.append(accu_loss / batchNum)
        train_accu.append(correct / total)

        # Test
        correct = 0
        total = 0
        accu_loss = 0
        batchNum = 0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                batchNum += 1
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = crit(outputs, labels)
                # Calculate the performance
                outputs, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                accu_loss += loss.item()
                total += labels.size(0)
                accuracy = correct / total
                print('[test] epoch: %2d, batch: %3d, loss: %.3f, accuracy: %.3f'\
                  % (epoch + 1, i + 1, accu_loss / (i+1), accuracy))
        test_loss.append(accu_loss / batchNum)
        test_accu.append(correct / total)

        #draw the figures
        pdf = PdfPages("alex_figure.pdf")
        plt.figure(1)
        plt.subplot(121)
        plt.plot(x_axis, train_accu, x_axis, test_accu)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")

        plt.subplot(122)
        plt.plot(x_axis, train_loss, x_axis, test_loss)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        pdf.savefig()
        plt.close()
        pdf.close()

        # Save the net
        net_name = "./data/exp/alex" + str(epoch+1) + ".pkl"
        torch.save(net, net_name)

    print('over')