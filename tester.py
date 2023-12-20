#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import kornia
import tifffile as tiff
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils

gan_rmse = []
gan_mae = []
gan_mee = []


def GAN_tester(opt):
    # Save the model if pre_train == True
    def load_model_generator(net, epoch, opt):
        model_name = 'GAN_G_epoch%d_batchsize%d.pth' % (100, 4)
        model_name = os.path.join('./models', model_name)
        pretrained_dict = torch.load(model_name)
        generator.load_state_dict(pretrained_dict)

    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # Build networks
    generator = utils.create_generator(opt).eval()
    print('-------------------------Loading Pretrained Model-------------------------')
    load_model_generator(generator, opt.epoch, opt)
    print('-------------------------Pretrained Model Loaded-------------------------')

    # To device
    generator = generator.cuda()

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    test_set = utils.TestSetLoader(dataset_dem=opt.dem_dir, dataset_rs=opt.rs_dir, dataset_mask=opt.mask_dir)
    dataloader = DataLoader(dataset=test_set, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True,
                            pin_memory=True, drop_last=True)
    # print('The overall number of images equals to %d' % len(test_set))

    # ----------------------------------------
    #            Testing
    # ----------------------------------------
    # Testing loop
    for batch_idx, (dem, rs, mask, img_max, img_min, geotrans, proj, list) in enumerate(dataloader):
        dem = Variable(dem).cuda()
        rs = Variable(rs).cuda()
        mask = Variable(mask).cuda()
        bad_dem = dem * (1 - mask) + mask
        # Generator output
        with torch.no_grad():
            second_out = generator(bad_dem, rs, mask)
        # forward propagation
        second_out_dem = dem * (1 - mask) + second_out * mask  # in range [0, 1]

        img_max = img_max.cuda()
        img_min = img_min.cuda()

        second_out_dem = utils.img_unnormalize(second_out_dem, img_max, img_min)
        true_dem = utils.img_unnormalize(dem.squeeze(0), img_max, img_min)
        true_dem = torch.from_numpy(true_dem).unsqueeze(0).unsqueeze(0).cuda()

        # if 0.0 < a <= 0.2:
        #     print(a)
        # gan_rmse.append(utils.RMSE(true_dem, second_out_dem, mask))
        # gan_mae.append(utils.MAE(true_dem, second_out_dem, mask))
        # gan_mee.append(utils.MEE(true_dem, second_out_dem))
        # utils.writeTiff(second_out_dem, geotrans, proj[0], opt.result_dir + '/' + str(list) + '.tif')

    # print(' mean RMSE: ', float(np.array(gan_rmse).mean()))
    # print(' mean MAE: ', float(np.array(gan_mae).mean()))
    # print(' mean Emax: ', float(np.array(gan_mee).mean()))

