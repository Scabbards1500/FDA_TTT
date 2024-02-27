import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from glob import glob
from utils import tent
import torch.optim as optim
from utils.dice_score import dice_loss




def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--ground-truth', '-g', metavar='GROUND_TRUTH', nargs='+', help='Filenames of ground truth masks')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()

def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_folder = args.input[0]  # Assuming the first argument is the input folder
    in_mask_folder = args.ground_truth[0] if args.ground_truth else None  # Ground truth mask folder, if specified
    out_folder = args.output[0] if args.output else None  # Output folder, if specified

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    # model = setup_tent(model)
    model.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    model.load_state_dict(state_dict,strict=False)

    logging.info('Model loaded!')

    in_files = get_image_files(in_folder)
    in_mask_files = get_image_files(in_mask_folder) if in_mask_folder else None

    # 这里出现的问题就是只有输入图片，没有输入图片标签，所以只能预测图片但是改变不了它的准确率
    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=model,
                           full_img=img,
                           mask_img=img,
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

def get_image_files(folder_path):
    jpg = glob(os.path.join(folder_path, '*.png'))
    png = glob(os.path.join(folder_path, '*.jpg'))
    return  jpg + png


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5,
                mask_img = None):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        print("no_grad")
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    if mask_img is not None:
        ground_truth_mask = torch.from_numpy(BasicDataset.preprocess(None, ground_truth_mask, scale_factor, is_mask=True))
        ground_truth_mask = ground_truth_mask.unsqueeze(0)
        ground_truth_mask = ground_truth_mask.to(device=device, dtype=torch.float32)

    return mask[0].long().squeeze().numpy()

#tent---------------------------------------------------------------
def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    # print(f"model for evaluation: %s", model)
    return model

def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=5,
                           episodic=False)
    # print(f"model for adaptation: %s", model)
    # print(f"params for adaptation: %s", param_names)
    # print(f"optimizer for adaptation: %s", optimizer)
    return tent_model

def setup_optimizer(params, optimizer_method='Adam', lr=0.001, beta=0.9, momentum=0.9, dampening=0, weight_decay=0, nesterov=False):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    print("tent")
    if optimizer_method == 'Adam':
        return optim.Adam(params, lr=lr, betas=(beta, 0.999), weight_decay=weight_decay)
    elif optimizer_method == 'SGD':
        return optim.SGD(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
    else:
        raise NotImplementedError

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


if __name__ == '__main__':
    main()
