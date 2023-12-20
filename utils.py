#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import gdal
import numpy as np
import cv2
import skimage
import torch
import math
from torch.utils.data.dataset import Dataset
import tifffile as tiff
import network


def create_generator(opt):
    # Initialize the networks
    generator = network.Generator(opt)
    print('Generator is created!')
    network.weights_init(generator, init_type=opt.init_type, init_gain=opt.init_gain)
    print('Initialize generator with %s type' % opt.init_type)
    return generator


def create_discriminator(opt):
    # Initialize the networks
    discriminator = network.PatchDiscriminator()
    print('Discriminator is created!')
    network.weights_init(discriminator, init_type=opt.init_type, init_gain=opt.init_gain)
    print('Initialize discriminator with %s type' % opt.init_type)
    return discriminator


def MAE(img1, img2, mask):
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    count = np.where(mask != 0)[0].shape[0]
    MAE = np.sum(np.abs(img1_np - img2_np)) / count
    # MAE = np.mean(np.abs(img1_np - img2_np))
    return MAE


def RMSE(img1, img2, mask):
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    count = np.where(mask != 0)[0].shape[0]
    MSE = np.sum((img1_np - img2_np) ** 2) / count
    # MSE = np.mean((img1_np - img2_np) ** 2)
    RMSE = MSE ** 0.5
    return RMSE


def MEE(img1, img2):
    # MAE_test = np.nanmean(np.abs(y_true-y_pred))
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    MEE = np.sum(np.max(np.abs(img1_np - img2_np)))
    return MEE


def readTif(dataset):
    data = gdal.Open(dataset)
    if dataset is None:
        print(data + "文件无法打开")
    return data


def writeTiff(im_data, im_geotrans, im_proj, path):
    datatype = gdal.GDT_Float32
    im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dem, dataset_rs, dataset_mask):
        super(TrainSetLoader, self).__init__()
        self.dataset_dem = dataset_dem
        self.dataset_rs = dataset_rs
        self.dataset_mask = dataset_mask

        self.dem_list = os.listdir(self.dataset_dem)
        self.rs_list = os.listdir(self.dataset_rs)
        self.mask_list = os.listdir(self.dataset_mask)
        # self.slope = Slope()

    def __getitem__(self, index):
        # img_void_left = np.array(tiff.imread(self.dataset_bad + '/' + self.bad_list[index] + '/spline.tif'),dtype=np.float32)
        img_hr_left = np.array(tiff.imread(self.dataset_dem + '/' + self.dem_list[index] + '/hr0.tif'),
                               dtype=np.float32)
        img_hr_right = np.array(tiff.imread(self.dataset_rs + '/' + self.rs_list[index] + '/hr1.tif'), dtype=np.float32)
        img_mask = np.array(tiff.imread(self.dataset_mask + '/' + self.mask_list[index] + '/corase.tif'),
                            dtype=np.float32)
        img_mask = 1 - img_mask
        img_hr_left, img_hr_right, img_mask = augumentation(img_hr_left, img_hr_right, img_mask)

        # H, W = img_hr_right.shape
        # img_hr_right = cv2.resize(img_hr_right, (H // 3, W // 3), interpolation=cv2.INTER_CUBIC)
        # img_hr_right = cv2.resize(img_hr_right, (H, W), interpolation=cv2.INTER_CUBIC)

        img_hr_left, _, _ = tensor(img_hr_left)
        img_hr_right, _, _ = tensor(img_hr_right)
        img_mask = torch.tensor(img_mask).unsqueeze(0)
        return img_hr_left, img_hr_right, img_mask

    def __len__(self):
        return len(self.dem_list)


class TestSetLoader(Dataset):
    def __init__(self, dataset_dem, dataset_rs, dataset_mask):
        super(TestSetLoader, self).__init__()
        self.dataset_dem = dataset_dem
        self.dataset_rs = dataset_rs
        self.dataset_mask = dataset_mask
        self.dem_list = os.listdir(self.dataset_dem)
        self.rs_list = os.listdir(self.dataset_rs)
        self.mask_list = os.listdir(self.dataset_mask)
        # self.slope = Slope()

    def __getitem__(self, index):
        groundtruth = readTif(self.dataset_dem + '/' + self.dem_list[index] + '/hr0.tif')
        geotrans = groundtruth.GetGeoTransform()
        proj = groundtruth.GetProjection()

        img_hr_left = np.array(tiff.imread(self.dataset_dem + '/' + self.dem_list[index] + '/hr0.tif'),
                               dtype=np.float32)
        img_hr_right = np.array(tiff.imread(self.dataset_rs + '/' + self.rs_list[index] + '/hr1.tif'), dtype=np.float32)
        img_mask = np.array(tiff.imread(self.dataset_mask + '/' + self.mask_list[index] + '/corase.tif'),
                            dtype=np.float32)
        img_mask = 1 - img_mask

        img_hr_left, img_max, img_min = tensor(img_hr_left)
        img_hr_right, rs_max, rs_min = tensor(img_hr_right)
        img_mask = torch.tensor(img_mask).unsqueeze(0)

        list = self.dem_list[index]
        return img_hr_left, img_hr_right, img_mask, img_max, img_min, geotrans, proj, list

    def __len__(self):
        return len(self.dem_list)


def img_normalize(img):
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = (img - img_min) / (img_max - img_min + 10)
    # img_norm.append(np.clip((img[i, :, :] - img_min[i]) / (img_max[i] - img_min[i]), 0, 1))

    return img_norm, img_max, img_min


def tensor(img):
    img, img_max, img_min = img_normalize(img)
    img = torch.from_numpy(img).unsqueeze(0)
    return img, img_max, img_min


def img_unnormalize(img, img_max, img_min):
    img_unnorm = img * (img_max - img_min + 10) + img_min
    img_unnorm = np.array(img_unnorm.squeeze(0).cpu())
    # img_unnorm = np.array(img_unnorm).transpose(2, 0, 1)
    return img_unnorm


def augumentation(hr_image_left, hr_image_right, img_mask):
    x = random.randint(1, 8)
    if x == 1:  # flip horizontally
        # img_void_left = img_void_left[::-1, :]
        hr_image_left = hr_image_left[::-1, :]
        hr_image_right = hr_image_right[::-1, :]
        img_mask = img_mask[::-1, :]

    if x == 2:  # flip vertically
        # img_void_left = img_void_left[:, ::-1]
        hr_image_left = hr_image_left[:, ::-1]
        hr_image_right = hr_image_right[:, ::-1]
        img_mask = img_mask[:, ::-1]

    if x == 3:
        # img_void_left = np.rot90(img_void_left, k=1, axes=(0, 1))
        hr_image_left = np.rot90(hr_image_left, k=1, axes=(0, 1))
        hr_image_right = np.rot90(hr_image_right, k=1, axes=(0, 1))
        img_mask = np.rot90(img_mask, k=1, axes=(0, 1))

    if x == 4:
        # img_void_left = np.rot90(img_void_left, k=2, axes=(0, 1))
        hr_image_left = np.rot90(hr_image_left, k=2, axes=(0, 1))
        hr_image_right = np.rot90(hr_image_right, k=2, axes=(0, 1))
        img_mask = np.rot90(img_mask, k=2, axes=(0, 1))

    if x == 5:
        # img_void_left = np.rot90(img_void_left, k=3, axes=(0, 1))
        hr_image_left = np.rot90(hr_image_left, k=3, axes=(0, 1))
        hr_image_right = np.rot90(hr_image_right, k=3, axes=(0, 1))
        img_mask = np.rot90(img_mask, k=3, axes=(0, 1))

    if x == 6:
        # img_void_left = img_void_left
        hr_image_left = hr_image_left
        hr_image_right = hr_image_right
        img_mask = img_mask

    if x == 7:
        # img_void_left = img_void_left[::-1, :]
        hr_image_left = hr_image_left[::-1, :]
        hr_image_right = hr_image_right[::-1, :]
        img_mask = img_mask[::-1, :]

        # img_void_left = np.rot90(img_void_left, k=1, axes=(0, 1))
        hr_image_left = np.rot90(hr_image_left, k=1, axes=(0, 1))
        hr_image_right = np.rot90(hr_image_right, k=1, axes=(0, 1))
        img_mask = np.rot90(img_mask, k=1, axes=(0, 1))

    if x == 8:
        # img_void_left = img_void_left[:, ::-1]
        hr_image_left = hr_image_left[:, ::-1]
        hr_image_right = hr_image_right[:, ::-1]
        img_mask = img_mask[:, ::-1]

        # img_void_left = np.rot90(img_void_left, k=1, axes=(0, 1))
        hr_image_left = np.rot90(hr_image_left, k=1, axes=(0, 1))
        hr_image_right = np.rot90(hr_image_right, k=1, axes=(0, 1))
        img_mask = np.rot90(img_mask, k=1, axes=(0, 1))

    return np.ascontiguousarray(hr_image_left), np.ascontiguousarray(hr_image_right), np.ascontiguousarray(img_mask)



