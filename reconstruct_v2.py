'''
This file reconstruct images from the features extract from the forth layer of the alexnet
'''

import os
import re
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cifar_alex import CifarAlexNet

def imshow(img):
	img = img.cpu()
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

def getTargets(img):
	torchvision.utils.make_grid(img)
	img = img / 2 + 0.5
	return img


class ReconstructNet2(nn.Module):
	def __init__(self):
		super(ReconstructNet2, self).__init__()
		self.conv1 = nn.Conv2d(96, 96, 3, padding = 1)
		self.conv2 = nn.Conv2d(96, 96, 3, padding = 1)
		self.conv3 = nn.Conv2d(96, 96, 3, padding = 1)
		self.convt1 = nn.ConvTranspose2d(96, 32, 5, padding = 2, output_padding = 1, stride = 2)
		self.convt2 = nn.ConvTranspose2d(32, 3, 5, padding = 2, output_padding = 1, stride = 2)
	def forward(self, inputs):
		x = inputs
		x = F.leaky_relu(self.conv1(x), negative_slope = 0.2)
		x = F.leaky_relu(self.conv2(x), negative_slope = 0.2)
		x = F.leaky_relu(self.conv3(x), negative_slope = 0.2)
		x = F.leaky_relu(self.convt1(x), negative_slope = 0.2)
		x = self.convt2(x)
		return x


if __name__ == "__main__":
	keepOn = False
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
	# Load the pretrained alexnet
	alexnet = torch.load("alex_trained.pkl")
	alexnet.eval()
	# Init ReconstructNet
	net = ReconstructNet2()
	
	st = 0
	if keepOn:
		res = os.listdir("./data/exp3")
		for netFile in res:
			last = int(re.sub("\D","",netFile))
			if last > st:
				st = last
		net = torch.load("./data/exp3/reconstruct" + str(st) + ".pkl")


	net.to(device)
	crit = nn.MSELoss(size_average = False)
	alexnet.to(device)
	# Defince the detail of training
	learningRate = [0.0001 for i in range(50)]
	learningRate.extend([0.00005 for i in range(40)])
	# Record performance
	train_loss = []
	test_loss = []
	x_axis = []
	# Train and Test
	for epoch in range(st, 90):
		x_axis.append(epoch + 1)
		optimizer = optim.Adam(net.parameters(), lr = learningRate[epoch])
		# Train
		accu_loss = 0
		batchNum = 0
		for i, data in enumerate(trainloader, 0):
			batchNum += 1
			optimizer.zero_grad()
			inputs, labels = data
			inputs = inputs.to(device)
			res, feature = alexnet(inputs)
			targets = getTargets(inputs)
			outputs = net(feature)
			loss = crit(outputs, targets)
			accu_loss += loss.item()
			loss.backward()
			optimizer.step()

			print('[train] epoch: %d, batch: %d, loss: %.5f' % (epoch + 1, (i + 1), accu_loss / (i+1)))
		train_loss.append(accu_loss / batchNum)
		batchNum = 0
		# Test
		with torch.no_grad():
			accu_loss = 0
			for i, data in enumerate(testloader, 0):
				batchNum += 1
				inputs, labels = data
				inputs = inputs.to(device)
				res, feature = alexnet(inputs)
				targets = getTargets(inputs)
				outputs = net(feature)
				loss = crit(outputs, targets)
				accu_loss += loss.item()
				# if i == 0:
				# 	imshow(targets[15])
				# 	imshow(outputs[15])

				print('[test] epoch: %d, batch: %d, loss: %.5f' % (epoch + 1, (i + 1), accu_loss / (i+1)))
		test_loss.append(accu_loss / batchNum)
		pdf = PdfPages("reconstruct_v2.pdf")
		plt.figure(1)
		plt.plot(x_axis, train_loss, x_axis, test_loss)
		plt.xlabel("epoch")
		plt.ylabel("loss")
		pdf.savefig()
		plt.close()
		pdf.close()

		# Save the net
		net_name = "./data/exp3/reconstruct" + str(epoch+1) + ".pkl"
		torch.save(net, net_name)
	print("over")