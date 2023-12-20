#!/usr/bin/env python
# coding: utf-8
import argparse
import os

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--results_path', type = str, default = './results', help = 'testing samples path that is a folder')
    parser.add_argument('--gan_type', type = str, default = 'GAN', help = 'the type of GAN for training')
    parser.add_argument('--gpu_ids', type = str, default = "0", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epoch', type = int, default = 40, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    # Network parameters
    parser.add_argument('--imgsize', type = int, default = 256, help = 'size of image')
    parser.add_argument('--in_channels', type = int, default = 2, help = 'input DEM + 1 channel mask')
    parser.add_argument('--out_channels', type = int, default =1, help = 'output DEM')
    parser.add_argument('--latent_channels', type = int, default = 48, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activation', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    # Dataset parameters
    parser.add_argument('--dem_dir', type=str, default='')
    parser.add_argument('--rs_dir', type=str, default='')
    parser.add_argument('--mask_dir', type=str, default='')
    parser.add_argument('--result_dir', type=str, default='./results/')
    opt = parser.parse_args()
    
    
    # ----------------------------------------
    #       Choose CUDA visible devices
    # ----------------------------------------
    
    # Enter main function
    import tester
    if opt.gan_type == 'GAN':
        tester.GAN_tester(opt)
    
