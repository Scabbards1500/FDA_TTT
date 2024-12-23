from __future__ import print_function
import argparse
from tqdm import tqdm
from PIL import Image
from unet import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from utils.prepare_dataset_ttt import *
from utils.misc_ttt import *
from utils.test_helpers_ttt import *
from utils.rotation_ttt import *
from utils.dice_score import dice_loss
import torch.nn.functional as F
import time

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='cifar10')
	parser.add_argument('--level', default=0, type=int)
	parser.add_argument('--corruption', default='original')
	parser.add_argument('--dataroot', default='/data/yusun/datasets/')
	parser.add_argument('--shared', default=None)
	########################################################################
	parser.add_argument('--depth', default=26, type=int)
	parser.add_argument('--width', default=1, type=int)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--group_norm', default=0, type=int)
	parser.add_argument('--fix_bn', action='store_true')
	parser.add_argument('--fix_ssh', action='store_true')
	########################################################################
	parser.add_argument('--lr', default=0.001, type=float)
	parser.add_argument('--niter', default=1, type=int)
	parser.add_argument('--online', action='store_true')
	parser.add_argument('--threshold', default=1, type=float)
	parser.add_argument('--dset_size', default=0, type=int)
	########################################################################
	parser.add_argument('--outf', default='.')
	parser.add_argument('--resume', default=None)
	parser.add_argument('--dir_out', default=r'D:\tempdataset\TTADataset\CHASE\test\output', help='Directory to save results')


	dir_img = Path(r"D:\tempdataset\TTADataset\RITE\test\images")
	dir_out = Path(r"D:\tempdataset\TTADataset\RITE\test\output")
	dir_mask = Path(r"D:\tempdataset\TTADataset\RITE\test\masks5122")



	args = parser.parse_args()
	args.threshold += 0.001		# to correct for numeric errors
	my_makedir(args.outf)
	import torch.backends.cudnn as cudnn
	cudnn.benchmark = True
	net, ext, head, ssh = build_model(args)
	teset, teloader = prepare_data(dir_img,dir_mask)
	device = torch.device('cuda')

	# 自带的

	# print('Resuming from %s...' %(args.resume))
	# ckpt = torch.load(args.resume,map_location=device)

	# ttt_train
	state_dict = torch.load(r"D:\python\UNet-TTA\checkpoints_RITE_ttt\checkpoint_epoch30.pth",map_location=device)
	ssh.load_state_dict(state_dict['ssh'])
	net.load_state_dict(state_dict['net'], strict=False) #成功

	# train
	# state_dict = torch.load(r"D:\python\UNet-TTA\checkpoints_CHASE\checkpoint_epoch10.pth",map_location=device)
	# # ssh.load_state_dict(state_dict['ssh'])
	# net.load_state_dict(state_dict, strict=False) #成功







	criterion_ssh = nn.CrossEntropyLoss().cuda()
	if args.fix_ssh:
		optimizer_ssh = optim.SGD(ext.parameters(), lr=args.lr)
	else:
		optimizer_ssh = optim.SGD(ssh.parameters(), lr=args.lr)

	def adapt_single(inputs):
		if args.fix_bn:
			ssh.eval()
		elif args.fix_ssh:
			ssh.eval()
			ext.train()
		else:
			ssh.train()
		for iteration in range(args.niter):
			# inputs = torch.stack(inputs)
			inputs_ssh, labels_ssh = rotate_batch(inputs, 'rand')
			inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
			optimizer_ssh.zero_grad()
			outputs_ssh = ssh(inputs_ssh)
			loss_ssh = criterion_ssh(outputs_ssh, labels_ssh)
			loss_ssh.backward()
			optimizer_ssh.step()

	def test_single(model, image, label): # 输出正确率
		model.eval()
		inputs = image #1，3，256，256
		with torch.no_grad():
			outputs = model(inputs.cuda()) #ssh:1,4// unet:1,2,256,256
			_, predicted = outputs.max(1)
			confidence = nn.functional.softmax(outputs, dim=1).squeeze()[label].item() #高度怀疑这一步出了问题，因为rotate和seg的评估方式不一
		correctness = 1 if predicted.item() == label else 0
		return correctness, confidence

	def test_single_seg(model, image, label): # 输出正确率
		model.eval().cuda()
		model.cuda()
		inputs = image #1，3，256，256 , label 256，256
		mask = label.unsqueeze(0)
		mask = mask.to(torch.device('cuda'))
		with torch.no_grad():
			outputs = model(inputs.cuda()) #// unet:1,2,256,256
			outputs1 = F.softmax(outputs, dim=1).float() #1,2,256,256
			maskresult = F.one_hot(mask, model.n_classes).permute(0, 3, 1, 2).float() #1,2,256,256

			# diceloss = dice_score(outputs1,maskresult)
			diceloss = dice_loss(outputs1,maskresult,multiclass=True)
			output = F.interpolate(outputs, (512,512), mode='bilinear')
			mask = output.argmax(dim=1)
		# print("diceloss",diceloss)
		return diceloss, mask.squeeze(0)

	def trerr_single(model, image):
		model.eval()
		labels = torch.LongTensor([0, 1, 2, 3])
		inputs = torch.stack([image for _ in range(4)])
		inputs = rotate_batch_with_labels(image, labels)
		inputs, labels = inputs.cuda(), labels.cuda()
		with torch.no_grad():
			outputs = model(inputs.cuda())
			print("ouitputs", outputs)
			print("labels",labels)
			_, predicted = outputs.max(1)
			corr = predicted.eq(labels)
			print("corr",corr)
		return corr.cpu()


	def mask_to_image(mask: np.ndarray, mask_values):
		if isinstance(mask_values[0], list):
			out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
		elif mask_values == [0, 1]:
			out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
		else:
			out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

		if mask.ndim == 3:
			mask = np.argmax(mask, axis=0)

		for i, v in enumerate(mask_values):
			out[mask == i] = v
		return Image.fromarray(out)


	def dice_score(pred, target):
		smooth = 0.5
		num = pred.size(0)
		# 将灰度标签转换为二值标签
		pred = (pred > 0.5).float()
		target = (target > 0.5).float()
		m1 = pred.view(num, -1)  # Flatten
		m2 = target.view(num, -1)  # Flatten
		intersection = (m1 * m2).sum()
		dice_coeff = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
		return dice_coeff



	print('Running...')
	correct = []
	sshconf = []
	trerror = []
	start = time.time()
	if args.dset_size == 0:
		args.dset_size = len(teset)
	for i in range(0,args.dset_size): #这个是直接加载数据集
		image = teset[i]['image'] # 3,256,256
		image = image.unsqueeze(0) # 1,3,256,256
		mask = teset[i]['mask'] # 256,256

		sshconf.append(test_single(ssh, image, 0)[1])
		if sshconf[-1] < args.threshold:
			adapt_single(image)
		ts, mask = test_single_seg(net, image, mask) #mask 1,2,256,256
		mask = mask.cpu().numpy()
		mask_values = [0, 1]
		result = mask_to_image(mask, mask_values)

		result.save(args.dir_out + '/result2_%d.png' % i)
		correct.append(ts) # ok
		# trerror.append(trerr_single(ssh, image))

	end = time.time()
	print(f'Inference done! Time elapsed: {time.time() - start:.2f} seconds')
	rdict = {'cls_correct': correct, 'ssh_confide': sshconf,
			'cls_adapted':1-mean(correct), 'trerror': trerror}
	torch.save(rdict, args.outf + '/%s_%d_ada.pth' %(args.corruption, args.level))
	# print("adapt!!!!!!!!!!!!!!!!!!!!!!!!!!!!finish!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

