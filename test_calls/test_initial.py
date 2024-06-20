from __future__ import print_function
import argparse

import torch
from utils.misc_ttt import *
from utils.test_helpers_ttt import *
from predict import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--level', default=0, type=int)
	parser.add_argument('--corruption', default='original')
	parser.add_argument('--dataroot', default=r'D:\tempdataset\TTADataset\CHASE\test\images')
	parser.add_argument('--shared', default=None)
	parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
	parser.add_argument('--ground-truth', '-g', metavar='GROUND_TRUTH', nargs='+',
						help='Filenames of ground truth masks')
	parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
	########################################################################
	parser.add_argument('--depth', default=26, type=int)
	parser.add_argument('--width', default=1, type=int)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--group_norm', default=0, type=int)
	parser.add_argument('--grad_corr', action='store_true')
	parser.add_argument('--visualize_samples', action='store_true')
	########################################################################
	parser.add_argument('--outf', default='.')
	parser.add_argument('--resume', default=r'D:\python\UNet-TTA\checkpoints_CHASE\checkpoint_epoch10.pth')
	parser.add_argument('--none', action='store_true')
	parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
	parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
	parser.add_argument('--method', '-me', type=str, default='source', help='Method for adaptation')


	args = parser.parse_args()
	my_makedir(args.outf)
	import torch.backends.cudnn as cudnn
	cudnn.benchmark = True
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	#获取数据
	in_folder = args.input[0]  # Assuming the first argument is the input folder
	in_mask_folder = args.ground_truth[0] if args.ground_truth else None  # Ground truth mask folder, if specified
	out_folder = args.output[0] if args.output else None  # Output folder, if specified



	model, ext, head, ssh = build_model(args)

	print('Resuming from %s...' %(args.resume))
	ckpt = torch.load(args.resume,ssh) #这是权重地址
	mask_values = ckpt.pop('mask_values', [0, 1])
	model.load_state_dict(ckpt['net'])



	in_files = get_image_files(in_folder)
	in_mask_files = get_image_files(in_mask_folder) if in_mask_folder else None

	for i, filename in enumerate(in_files):

		logging.info(f'Predicting image {filename} ...')
		img = Image.open(filename)
		base_name = os.path.basename(filename)
		print(base_name)
		# 拼接基本名称到in_mask_folder路径中，打开同名文件
		mask_file = os.path.join(in_mask_folder, base_name)
		gt_mask = Image.open(mask_file) if in_mask_files else None

		mask, dice = predict_img(model=model,
								 full_img=img,
								 mask_img=gt_mask,
								 scale_factor=args.scale,
								 out_threshold=args.mask_threshold,
								 device=device)

		if not args.no_save:
			out_filename = os.path.join(out_folder, f'{os.path.basename(filename).split(".")[0]}_OUT.png') \
				if out_folder else f'{os.path.splitext(filename)[0]}_OUT.png'

			result = mask_to_image(mask, mask_values)
			result.save(out_filename)
			logging.info(f'Mask saved to {out_filename}')

		if args.viz:
			logging.info(f'Visualizing results for image {filename}, close to continue...')
			plot_img_and_mask(img, mask)

	print(f"average dice score: {np.mean(diceloss)}")



	print('Old test error ssh %.2f' %(ckpt['err_ssh']*100))
	head.load_state_dict(ckpt['head'])
	ssh_initial, ssh_correct, ssh_losses = [], [], []

	labels = [0,1,2,3]
	for label in labels:
		tmp = test(teloader, ssh, sslabel=label)
		ssh_initial.append(tmp[0])
		ssh_correct.append(tmp[1])
		ssh_losses.append(tmp[2])

	rdict = {'cls_initial': cls_initial, 'cls_correct': cls_correct, 'cls_losses': cls_losses,
				'ssh_initial': ssh_initial, 'ssh_correct': ssh_correct, 'ssh_losses': ssh_losses}
	torch.save(rdict, args.outf + '/%s_%d_inl.pth' %(args.corruption, args.level))

	if args.grad_corr:
		corr = test_grad_corr(teloader, net, ssh, ext)
		print('Average gradient inner product: %.2f' %(mean(corr)))
		torch.save(corr, args.outf + '/%s_%d_grc.pth' %(args.corruption, args.level))

