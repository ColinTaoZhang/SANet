import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random
import scipy.stats as stats
from os.path import join
import cv2
import exifread
import rawpy
from . import process

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

def read_raw(rawpath, pack=True, verbose=True):
    raw = rawpy.imread(rawpath)
    raw_image = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern

    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)
    
    H = raw_image.shape[0]
    W = raw_image.shape[1]

    packed_raw = np.stack((raw_image[R[0][0]:H:2,R[1][0]:W:2], #RGBG
                    raw_image[G1[0][0]:H:2,G1[1][0]:W:2],
                    raw_image[B[0][0]:H:2,B[1][0]:W:2],
                    raw_image[G2[0][0]:H:2,G2[1][0]:W:2]), axis=0).astype(np.float32)
    packed_raw -= 512.

    if pack:
        packed_raw = packed_raw / (16383.-512.)#, raw_pattern
        return packed_raw
        
    else:
        raw_image = (np.expand_dims(raw_image, axis=0) - 512.) / (16383.-512.)#, raw_pattern
        return raw_image

def read_raw_ARQ(rawpath):
    raw = rawpy.imread(rawpath)
    raw_image = raw.raw_image_visible.astype(np.float32)
    
    raw_image -= 512.0
    raw_image /= (16383.0-512.0)
    rgb = np.concatenate((raw_image[:,:,:1], 0.5*(raw_image[:,:,1:2]+raw_image[:,:,3:]), raw_image[:,:,2:3]), axis=2)
    rgb = rgb.transpose((2,0,1))

    return rgb

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform

        clean_files = []
        noisy_files = []
        clean_files += sorted([os.path.join(rgb_dir, 'GT', x) for x in os.listdir(os.path.join(rgb_dir, 'GT')) if x.endswith('.ARQ')])
        noisy_files += [x.replace('GT', '50').replace('.ARQ', '_1.ARW') for x in clean_files]

        self.noisy_filenames = noisy_files#[:1]
        self.clean_filenames = clean_files#[:1]

        self.clean = [np.float32(read_raw_ARQ(self.clean_filenames[index])) for index in range(len(self.clean_filenames))]
        self.noisy = [np.float32(read_raw(self.noisy_filenames[index])) for index in range(len(self.noisy_filenames))]        
        
        self.img_options=img_options

        ps = self.img_options['patch_size']
        self.mask = np.zeros((3, 2*ps, 2*ps), dtype=np.float32)
        self.mask[0, 0:2*ps:2, 0:2*ps:2] = 1
        self.mask[1, 0:2*ps:2, 1:2*ps:2] = 2
        self.mask[2, 1:2*ps:2, 0:2*ps:2] = 3
        self.mask[1, 1:2*ps:2, 1:2*ps:2] = 2
        self.mask = torch.from_numpy(self.mask)

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size * 50

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = self.clean[tar_index] #torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = self.noisy[tar_index] #torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = noisy.shape[1]
        W = noisy.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, 2*r:2*r + 2*ps, 2*c:2*c + 2*ps] * 10
        noisy = noisy[:, r:r + ps, c:c + ps] * 10 * 50

        _,h,w = noisy.shape
        noisy_ = np.zeros((3, h*2, w*2), dtype=np.float32)
        noisy_[0, 0:2*h:2, 0:2*w:2] = noisy[0]
        noisy_[1, 0:2*h:2, 1:2*w:2] = noisy[1]
        noisy_[2, 1:2*h:2, 1:2*w:2] = noisy[2]
        noisy_[1, 1:2*h:2, 0:2*w:2] = noisy[3]

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)
        noisy_ = torch.from_numpy(noisy_)

        return clean, noisy, clean_filename, noisy_filename, self.mask, noisy_


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        clean_files = []
        noisy_files = []
        clean_files += sorted([os.path.join(rgb_dir, 'GT', x) for x in os.listdir(os.path.join(rgb_dir, 'GT')) if x.endswith('.ARQ')])
        noisy_files += [x.replace('GT', '50').replace('.ARQ', '_1.ARW') for x in clean_files]

        clean_files = clean_files#[:1]
        noisy_files = noisy_files#[:1]
        
        self.noisy_filenames = noisy_files
        self.clean_filenames = clean_files

        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size        

        clean = np.float32(read_raw_ARQ(self.clean_filenames[tar_index]))
        noisy = np.float32(read_raw(self.noisy_filenames[tar_index]))
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        H = noisy.shape[1]//2
        W = noisy.shape[2]//2
        patch_size = 128
        clean = clean[:, 2*H-2*patch_size:2*H+2*patch_size, 2*W-2*patch_size:2*W+2*patch_size] * 10
        noisy = noisy[:, H-patch_size:H+patch_size, W-patch_size:W+patch_size] * 10 * 50

        _,h,w = noisy.shape
        noisy_ = np.zeros((3, h*2, w*2), dtype=np.float32)
        noisy_[0, 0:2*h:2, 0:2*w:2] = noisy[0]
        noisy_[1, 0:2*h:2, 1:2*w:2] = noisy[1]
        noisy_[2, 1:2*h:2, 1:2*w:2] = noisy[2]
        noisy_[1, 1:2*h:2, 0:2*w:2] = noisy[3]

        mask = np.zeros((3, 2*h, 2*w), dtype=np.float32)
        mask[0, 0:2*h:2, 0:2*w:2] = 1
        mask[1, 0:2*h:2, 1:2*w:2] = 2
        mask[2, 1:2*h:2, 0:2*w:2] = 3
        mask[1, 1:2*h:2, 1:2*w:2] = 2

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)
        mask = torch.from_numpy(mask)
        noisy_ = torch.from_numpy(noisy_)

        return clean, noisy, clean_filename, noisy_filename, mask, noisy_