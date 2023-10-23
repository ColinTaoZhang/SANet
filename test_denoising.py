

"""
## Learning Enriched Features for Real Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## ECCV 2020
## https://arxiv.org/abs/2003.06792
"""


import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from models import *
import torch.nn.functional as F

# from networks.MIRNet_model import MIRNet
# from models import *
# from SGNet import *
from dataset_rgb import get_test_data
import utils
from skimage import img_as_ubyte
import scipy.io as sio
import h5py
import cv2
import matplotlib.pyplot as plt
import matplotlib

from image_utils import batch_metric

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
# parser.add_argument('--input_dir', default='/home/zhangtao/RGB_dataset/benchmark/Urban100/HR', type=str, help='Directory of validation images')
# parser.add_argument('--result_dir', default='./results_DM/Syn/', type=str, help='Directory for results')
# # parser.add_argument('--weights', default='./checkpoints_DM/Denoising/models/Unet_Ada_Res_Res32_Demosaic_Syn/model_best.pth', type=str, help='Path to weights')
# parser.add_argument('--weights', default='./checkpoints_DM/AJDD/Denoising/models/SGNet_Res32_Demosaic_Syn/model_best.pth', type=str, help='Path to weights')

# parser.add_argument('--input_dir', default='/traindata/PixelShift/testing', type=str, help='Directory of validation images')
# parser.add_argument('--result_dir', default='./results_DM/noise-free/test/', type=str, help='Directory for results')
# parser.add_argument('--weights', default='./checkpoints_DM/Denoising/models/Unet_Ada_Res_Res32_Demosaic_Real/model_latest.pth', type=str, help='Path to weights')
# # parser.add_argument('--weights', default='./checkpoints_DM/Denoising/models/CDMCNN_Res32_Demosaic_Real/model_latest.pth', type=str, help='Path to weights')

parser.add_argument('--input_dir', default='E:/demosaic/DM/NOISY/TEST', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results_DM/noisy/test/', type=str, help='Directory for results')
# parser.add_argument('--weights', default='./checkpoints_DM/AJDD/Denoising/models/Unet_Ada_Res_Res32_Demosaic_Real/model_epoch_90.pth', type=str, help='Path to weights')
parser.add_argument('--weights', default=r'E:\demosaic\SANet\checkpoints\model_best.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

utils.mkdir(args.result_dir)

test_dataset = get_test_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=0, drop_last=False)



# model_restoration = U_Net()
# model_restoration = Ada_U_Net(in_ch=3, mask_in_ch=3, ps=False)
model_restoration = Ada_Res_U_Net(in_ch=3, mask_in_ch=3, ps=False)
# model_restoration = A_U_Net(in_ch=3, ps=False)
# model_restoration = U_Net_IM()
# model_restoration = SGNet()
# model_restoration = BayerDemosaick()
# model_restoration = CDMCNN()
# model_restoration = JDNDMSR()
# model_restoration = U_Net()
# model_restoration = LiteISPNet()
utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration=nn.DataParallel(model_restoration).cuda()

model_restoration.eval()

# data = h5py.File('/home/zhangtao/code/PixelShift/DM_methods/ADMM/res/Ours/compareResults_Ours_sigma0_clean.mat', 'r')
# name = data['results']

with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].cuda()
        rgb_noisy = data_test[1].float().cuda()
        mask = data_test[4].cuda()
        input_unpack = data_test[5].float().cuda()
        filenames = data_test[2]

        # rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = model_restoration(input_unpack, mask)
        # rgb_restored, rgb_restored_green = model_restoration(rgb_noisy) # SGNet
        # rgb_restored, rgb_restored_green = model_restoration(input_unpack) # CDMCNN
        # rgb_restored = model_restoration(input_unpack) # DJN
        # rgb_restored = torch.clamp(rgb_restored,0,1)
        # rgb_gt /= 10
        # rgb_restored /= 10
        # rgb_gt[:,0] /= 2.16/3.0
        # rgb_gt[:,2] /= 2.0/3.0
        # rgb_restored[:,0] /= 2.16/3.0
        # rgb_restored[:,2] /= 2.0/3.0
     
        # psnr_val_rgb.append(utils.batch_PSNR(rgb_restored, rgb_gt, 1.))
        # psnr, ssim = utils.batch_metric(rgb_restored.cpu().detach().numpy(), rgb_gt.cpu().detach().numpy())

        rgb_gt = rgb_gt / 10
        rgb_restored = rgb_restored / 10

        # # rgb_restored = np.asarray(data[name[(ii+1)%13,0]]['admm']['RGB'], dtype=np.float32).transpose((0,2,1))/255.0 # noisy
        # rgb_restored = np.asarray(data[name[ii,0]]['flexisp']['RGB'], dtype=np.float32).transpose((0,2,1))#/255.0
        # # rgb_restored = np.expand_dims(rgb_restored, axis=0).transpose((0,2,3,1))
        # rgb_restored = torch.FloatTensor(rgb_restored).cuda().unsqueeze(0) * rgb_gt.max()
        # rgb_restored[:,0] /= 2.16
        # rgb_restored[:,2] /= 2.0

        rgb_gt = rgb_gt.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        # rgb_gt = utils.process(rgb_gt).permute(0, 2, 3, 1).cpu().detach().numpy()
        # rgb_restored = utils.process(rgb_restored).permute(0, 2, 3, 1).cpu().detach().numpy()

        psnr, ssim = batch_metric(rgb_restored, rgb_gt)
        psnr_val_rgb.append(psnr)
        ssim_val_rgb.append(ssim)

        if args.save_images:
            for batch in range(len(rgb_gt)):
                # sio.savemat(args.result_dir + filenames[batch].split('/')[-1][:-4] + '.mat', {'mosaic': rgb_mosaic[batch], 'gt': rgb_gt[batch]})

                # denoised_img = img_as_ubyte(rgb_gt[batch])
                # utils.save_img(args.result_dir + filenames[batch][-8:-4] + '_gt.png', denoised_img)

                # denoised_img = img_as_ubyte(rgb_noisy[batch])
                # utils.save_img(args.result_dir + filenames[batch][-8:-4] + '_noisy.png', denoised_img)

                denoised_img = img_as_ubyte(rgb_gt[batch])
                # utils.save_img(args.result_dir + filenames[batch][-8:-4] + '_restored.png', denoised_img)
                utils.save_img(args.result_dir + filenames[batch].split('/')[-1][:-4] + '.png', denoised_img)
                # utils.save_img(args.result_dir + filenames[batch][-8:-4] + '.png', denoised_img)

        with open(args.result_dir+'metric.txt', 'a') as fd:
            for i in range(len(filenames)):
                fd.write('ID: {}, PSNR: {}, SSIM: {}\n'.format(filenames[i].split('/')[-1][:-4], psnr, ssim))    

mean_psnr = sum(psnr_val_rgb)/len(psnr_val_rgb)
mean_ssim = sum(ssim_val_rgb)/len(ssim_val_rgb)
with open(args.result_dir + 'metric.txt', 'a') as fd:
    fd.write('\nPSNR: {}, SSIM: {}\n'.format(mean_psnr, mean_ssim))
print("PSNR: %.4f, SSIM: %.4f" %(mean_psnr, mean_ssim))

