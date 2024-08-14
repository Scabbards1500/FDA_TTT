from __future__ import print_function
import argparse
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim

import multiprocessing
import torch.optim as optim
from utils.prepare_dataset_ttt import *
from utils.misc_ttt import *
from utils.test_helpers_ttt import *
from utils.rotation_ttt import *

import os
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch.backends.cudnn as cudnn

if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--dataroot', default=r'D:\tempdataset\CIFAR10')
    parser.add_argument('--shared', default="layer2")
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--group_norm', default=0, type=int)
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--nepoch', default=75, type=int)
    parser.add_argument('--milestone_1', default=50, type=int)
    parser.add_argument('--milestone_2', default=65, type=int)
    parser.add_argument('--rotation_type', default='rand')
    parser.add_argument('--outf', default='.')

    args = parser.parse_args()

    dir_img = Path(r"D:\tempdataset\TTADataset\MoNuSeg\train\images")
    dir_mask = Path(r'D:\tempdataset\TTADataset\MoNuSeg\train\masks')

    my_makedir(args.outf)

    cudnn.benchmark = True
    device = torch.device("cuda")
    net, ext, head, ssh = build_model(args)
    net.to(device)
    ssh.to(device)
    _, teloader = prepare_data(dir_img, dir_mask)
    _, trloader = prepare_data(dir_img, dir_mask)

    parameters = list(net.parameters()) + list(head.parameters())
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)
    criterion = nn.CrossEntropyLoss().to(device)

    all_err_cls = []
    all_err_ssh = []
    print('Running...')
    print('Error (%)\t\ttest\t\tself-supervised')
    for epoch in range(1, args.nepoch + 1):
        net.train()
        ssh.train()
        for batch in trloader:
            optimizer.zero_grad()
            images, true_masks = batch['image'].to(device), batch['mask'].to(device)

            outputs_cls = net(images)
            loss = criterion(outputs_cls, true_masks)

            if args.shared is not None:
                inputs_ssh, labels_ssh = rotate_batch(images, args.rotation_type)
                inputs_ssh, labels_ssh = inputs_ssh.to(device), labels_ssh.to(device)
                outputs_ssh = ssh(inputs_ssh)
                output = [outputs_ssh[0].argmax(),outputs_ssh[1].argmax(),outputs_ssh[2].argmax(),outputs_ssh[3].argmax() ]
                print("label",labels_ssh)
                print("output_ssh",outputs_ssh)
                print("output",output)
                loss_ssh = criterion(outputs_ssh, labels_ssh)
                loss += loss_ssh

            loss.backward()
            optimizer.step()

        err_cls = test(teloader, net)[0]
        err_ssh = 0 if args.shared is None else test_ssh(teloader, ssh, sslabel='expand')[0]
        all_err_cls.append(err_cls)
        all_err_ssh.append(err_ssh)
        scheduler.step()

        print(('Epoch %d/%d:' % (epoch, args.nepoch)).ljust(24) +
              '%.2f\t\t%.2f' % (err_cls * 100, err_ssh * 100))
        torch.save((all_err_cls, all_err_ssh), args.outf + '/loss.pth')
        plot_epochs(all_err_cls, all_err_ssh, args.outf + '/loss.pdf')

    state = {'err_cls': err_cls, 'err_ssh': err_ssh,
             'net': net.state_dict(), 'head': head.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, args.outf + '/ckpt.pth')
    torch.save(net.state_dict(),args.outf + '/net.pth')

