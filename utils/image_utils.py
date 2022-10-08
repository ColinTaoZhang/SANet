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