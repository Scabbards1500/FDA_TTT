from torch import nn
import math
import copy
import torch

class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

class ExtractorHead(nn.Module):
	def __init__(self, ext, head):
		super(ExtractorHead, self).__init__()
		self.ext = ext
		self.head = head

	def forward(self, x):
		exts = self.ext(x)
		out = self.head(exts)
		return out

def extractor_from_layer3(net):
	layers = net
	return layers

def extractor_from_layer2(net):
	# # layers = [net.inc, net.down1, net.down2, net.down3, net.down4, net.up1, net.up2, net.up3, net.up4]
	layers = [net.inc, net.down1, net.down2]
	seqlayers = nn.Sequential(*layers)
	return seqlayers

def head_on_layer2(net, width, classes):
	head = copy.deepcopy([net.down3, net.down4])
	head.append(nn.AdaptiveAvgPool2d((1, 1)))
	head.append(ViewFlatten())
	head.append(nn.Linear(1024, 4))
	return nn.Sequential(*head)




