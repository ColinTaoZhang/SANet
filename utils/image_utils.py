import torch
import numpy as np
import pickle
import cv2
from skimage.measure import compare_psnr, compare_ssim

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img.transpose((2,0,1))

def save_img(filepath, img):
    cv2.imwrite(filepath,img[:,:,::-1])

def mosaic(img):
    mosaic = np.zeros((img.shape[0]*2, img.shape[1]*2, 3), dtype=np.float32)
    mosaic[::2, ::2, 0] = img[:,:,0]
    mosaic[::2, 1::2, 1] = img[:,:,1]
    mosaic[1::2, ::2, 1] = img[:,:,3]
    mosaic[1::2, 1::2, 2] = img[:,:,2]
    return mosaic

def processing(img, gt=None):
    img[:,:,:,0] *= 2.16
    img[:,:,:,2] *= 2
    if not gt is None:
        return np.clip(img/np.amax(gt), 0, 1)
    else:
        return np.clip(img, 0, 1)

def processing2(img, gt_max=None, R_gain=2.16, B_gain=2.0):
    img[:,:,:,0] *= R_gain
    img[:,:,:,2] *= B_gain
    if not gt_max is None:
        return np.clip(img/gt_max, 0, 1)
    else:
        gt_max = np.amax(np.clip(img, 0, 1))
        return np.clip(img/gt_max, 0, 1), gt_max
        
def myPSNR(prd_img, tar_img):
    imdff = prd_img- tar_img
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(torch.max(tar_img)/rmse)
    return ps

def batch_PSNR(img1, img2, data_range=None):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR)

def batch_metric(img1, img2):
    PSNR = []
    SSIM = []
    for im1, im2 in zip(img1, img2):
        im1[:,:,0] *= 2.36  # R
        im1[:,:,2] *= 1.7   # B
        im2[:,:,0] *= 2.36
        im2[:,:,2] *= 1.7
        psnr = compare_psnr(im1, im2, data_range=1.0)
        ssim = compare_ssim(im1, im2, multichannel=True)

        # im1 = im1.transpose((1,2,0)) ### sRGB
        # im2 = im2.transpose((1,2,0))
        # psnr = (compare_psnr(im1[:,:,0], im2[:,:,0], data_range=1.0)+compare_psnr(im1[:,:,1], im2[:,:,1], data_range=1.0)+compare_psnr(im1[:,:,2], im2[:,:,2], data_range=1.0))/3
        # ssim = (compare_ssim(im1[:,:,0], im2[:,:,0], gaussian_weights=True)+compare_ssim(im1[:,:,1], im2[:,:,1], gaussian_weights=True)+compare_ssim(im1[:,:,2], im2[:,:,2], gaussian_weights=True))/3
        PSNR.append(psnr)
        SSIM.append(ssim)
    return sum(PSNR)/len(PSNR), sum(SSIM)/len(SSIM)

def apply_gains(bayer_images, wbs):
    N, C, _, _ = bayer_images.shape
    outs = bayer_images * wbs.view(N, C, 1, 1)
    return outs

def apply_ccms(images, ccms):
    images = images.permute(0, 2, 3, 1)
    images = images[:, :, :, None, :]
    ccms = ccms[:, None, None, :, :]
    outs = torch.sum(images * ccms, dim=-1)
    outs = outs.permute(0, 3, 1, 2)
    return outs

def gamma_compression(images, gamma=2.2):
    outs = torch.clamp(images, min=1e-8) ** (1.0 / gamma)
    outs = torch.clamp((outs*255).int(), min=0, max=255).float() / 255
    return outs

def binning(bayer_images):
    lin_rgb = torch.stack([
        bayer_images[:,0,...],
        torch.mean(bayer_images[:, [1,3], ...], dim=1),
        bayer_images[:,2,...]], dim=1)
    return lin_rgb

def process(bayer_images, wbs=None, ccms=None, gamma=2.2):
    if wbs is None:
        R_gain = 2.36 #bayer_images[:,1].mean()/bayer_images[:,0].mean()
        B_gain = 1.7 #bayer_images[:,1].mean()/bayer_images[:,2].mean()
        # print(R_gain, B_gain)
        wbs = torch.FloatTensor([[R_gain, 1.0, B_gain, 1.0]]).cuda()
    if ccms is None:
        ccms = torch.FloatTensor([[[1.54174, -0.35795, -0.18379],
                              [-0.13061, 1.54902, -0.41841],
                              [0.00474, -0.39966, 1.39492]]]).cuda()
    if bayer_images.shape[1] == 3:
        images = apply_gains(bayer_images, wbs[:,:-1])
        images = torch.clamp(images, min=0.0, max=1.0)
    else:
        bayer_images = apply_gains(bayer_images, wbs)
        bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
        images = binning(bayer_images)
    images = apply_ccms(images, ccms)
    images = torch.clamp(images, min=0.0, max=1.0)
    images = gamma_compression(images, gamma)
    images = torch.clamp(images, min=0.0, max=1.0)
    return images
