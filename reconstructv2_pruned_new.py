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
from prune_utility import *


def imshow(img):
	img = img.cpu()
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

def getTargets(img):
	torchvision.utils.make_grid(img)
	img = img / 2 + 0.5
	return img

def show_result(img):
	torchvision.utils.make_grid(img)
	img = img / 2 + 0.5
	imshow(img)

def crit(t):
	return (sum(t ** 2) / 2)

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
	# Init ReconstructNet2
	net = ReconstructNet2()
	
	st = 0
	st_retrain = 0
	if keepOn:
		res = os.listdir("./data/exp4_prune")
		for netFile in res:
			last = int(re.sub("\D","",netFile))
			if last > st:
				st = last
		if st > 0:
			net = torch.load("./data/exp4_prune/reconstruct" + str(st) + ".pkl")
			


	net.to(device)
	crit = nn.MSELoss()
	alexnet.to(device)

	# Defince the detail of training
	learningRate = [0.0001 for i in range(50)]
	learningRate.extend([0.00005 for i in range(41)])
	#ADMM algorithm parameters initialization
	Z1 = projection(net.conv1.weight.data, configuration.P1)
	U1 = torch.zeros(Z1.size()).cuda()
	zeros1 = U1
	Z2 = projection(net.conv2.weight.data, configuration.P2)
	U2 = torch.zeros(Z2.size()).cuda()
	zeros2 = U2
	Z3 = projection(net.conv3.weight.data, configuration.P3)
	U3 = torch.zeros(Z3.size()).cuda()
	zeros3 = U3
	Z4 = projection(net.convt1.weight.data, configuration.P4)
	U4 = torch.zeros(Z4.size()).cuda()
	zeros4 = U4
	Z5 = projection(net.convt2.weight.data, configuration.P5)
	U5 = torch.zeros(Z5.size()).cuda()
	zeros5 = U5
	# Record performance
	retrain_loss = []
	test_loss = []
	x_axis = []
	#train and test
	for epoch in range(st, len(learningRate) + 1):
		x_axis.append(epoch + 1)
		optimizer = optim.Adam(net.parameters(), lr = learningRate[epoch])
		# Train
		accu_loss = 0
		batchNum = 0
		for i, data in enumerate(trainloader, 0):
			batchNum += 1
			continue
			optimizer.zero_grad()
			inputs, labels = data
			inputs = inputs.to(device)
			res, feature = alexnet(inputs)
			targets = getTargets(inputs)
			outputs = net(feature) #forward step
			loss = crit(outputs, targets) + 0.00005*(\
				crit(net.conv1.weight.data - Z1 + U1, zeros1) + \
				crit(net.conv2.weight.data - Z2 + U2, zeros2) + \
				crit(net.conv3.weight.data - Z3 + U3, zeros3) + \
				crit(net.convt1.weight.data - Z4 + U4, zeros4) + \
				crit(net.convt2.weight.data - Z5 + U5, zeros5))
			accu_loss += loss.item()
			loss.backward()
			optimizer.step()
			print('[train] epoch: %d, batch: %d, loss: %.5f' % (epoch + 1, (i + 1), accu_loss / (i+1)))
		retrain_loss.append(accu_loss / batchNum)
		#ADMM
		Z1 = net.conv1.weight.data + U1
		Z1 = projection(Z1, percent=configuration.P1)
		U1 = U1 + net.conv1.weight.data - Z1

		Z2 = net.conv2.weight.data + U2
		Z2 = projection(Z2, percent=configuration.P2)
		U2 = U2 + net.conv2.weight.data - Z2

		Z3 = net.conv3.weight.data + U3
		Z3 = projection(Z3, percent=configuration.P3)
		U3 = U3 + net.conv3.weight.data - Z3

		Z4 = net.convt1.weight.data + U4
		Z4 = projection(Z4, percent=configuration.P4)
		U4 = U4 + net.convt1.weight.data - Z4
		
		Z5 = net.convt2.weight.data + U5
		Z5 = projection(Z5, percent = configuration.P5)
		U5 = U5 + net.convt2.weight.data - Z5
		# Test
		batchNum = 0

		with torch.no_grad():
			accu_loss = 0
			for i, data in enumerate(testloader, 0):
				batchNum += 1
				inputs, labels = data
				inputs = inputs.to(device)
				res, feature = alexnet(inputs)
				targets = getTargets(inputs)
				outputs = net(feature)
				loss = crit(outputs, targets) + 0.00005 * (\
					crit(net.conv1.weight.data - Z1 + U1, zeros1) + \
					crit(net.conv2.weight.data - Z2 + U2, zeros2) + \
					crit(net.conv3.weight.data - Z3 + U3, zeros3) + \
					crit(net.convt1.weight.data - Z4 + U4, zeros4) + \
					crit(net.convt2.weight.data - Z5 + U5, zeros5))
				accu_loss += loss.item()
				if i == 0:
					imshow(targets[0])
					imshow(outputs[0])

				print('[test] epoch: %d, batch: %d, loss: %.5f' % (epoch + 1, (i + 1), accu_loss / (i+1)))

		test_loss.append(accu_loss / batchNum)
		pdf = PdfPages("reconstruct_v3_prune.pdf")
		plt.figure(1)
		plt.plot(x_axis, retrain_loss, x_axis, test_loss)
		plt.xlabel("epoch")
		plt.ylabel("loss")
		pdf.savefig()
		plt.close()
		pdf.close()
		# Save the net
		net_name = "./data/exp4_prune/reconstruct" + str(epoch+1) + ".pkl"
		torch.save(net, net_name)
	print("train step complete")

	# little test before official work
	#dataiter = iter(trainloader)
	#images, labels = dataiter.next()
	#images = images.to(device)
	#res, feature = alexnet(images)

	#targets = getTargets(images)



	# retrain step
	# Record performance
	retrain_loss = []
	test_loss = []
	x_axis = []
	for epoch in range(0, 90):
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
			outputs = net(feature) #forward step
			loss = crit(outputs, targets)
			accu_loss += loss.item()
			loss.backward()
			#prune step
			print("net.conv1:")
			net.conv1.weight.data, net.conv1.weight._grad = apply_prune(net.conv1.weight, configuration.P1)
			print("net.conv2:")
			net.conv2.weight.data, net.conv2.weight._grad = apply_prune(net.conv2.weight, configuration.P1)
			print("net.conv3:")
			net.conv3.weight.data, net.conv3.weight._grad = apply_prune(net.conv3.weight, configuration.P1)
			print("net.convt1:")
			net.convt1.weight.data, net.convt1.weight._grad=apply_prune(net.convt1.weight,configuration.P4)
			print("net.convt2:")
			net.convt2.weight.data, net.convt2.weight._grad=apply_prune(net.convt2.weight,configuration.P5)
			optimizer.step()
			print('[retrain] epoch: %d, batch: %d, loss: %.5f' % (epoch + 1, (i + 1), accu_loss / (i+1)))
		retrain_loss.append(accu_loss / batchNum)
		# Test
		batchNum = 0
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
		pdf = PdfPages("reconstruct_v3_prune_retrain.pdf")
		plt.figure(1)
		plt.plot(x_axis, retrain_loss, x_axis, test_loss)
		plt.xlabel("epoch")
		plt.ylabel("loss")
		pdf.savefig()
		plt.close()
		pdf.close()
		# Save the net
		net_name = "./data/exp4_retrain/reconstruct" + str(epoch+1) + ".pkl"
		torch.save(net, net_name)
	print("retrain step complete")