import os
import time
import datetime
import numpy as np
import random
import kornia
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import network
import utils
from loss import *
import cv2

ganloss = []
allloss = []
epoch_num = []

seed = 20
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def GAN_trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = opt.save_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Build networks
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)

    # Loss functions
    L1Loss = nn.L1Loss()
    adversarial_loss = AdversarialLoss(type='nsgan').cuda()
    style_Loss = StyleLoss().cuda()
    perceptual_loss = PerceptualLoss().cuda()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2),
                                   weight_decay=opt.weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2),
                                   weight_decay=opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(lr_in, optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Save the two-stage generator model
    def save_model_generator(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'GAN_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))

    # Save the dicriminator model
    def save_model_discriminator(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'GAN_D_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))

    # load the model
    def load_model(net, epoch, opt, type='G'):
        """Save the model at "checkpoint_interval" and its multiple"""
        if type == 'G':
            model_name = 'GAN_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        else:
            model_name = 'GAN_D_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        pretrained_dict = torch.load(model_name)
        net.load_state_dict(pretrained_dict)

    if opt.resume:
        load_model(generator, opt.resume_epoch, opt, type='G')
        load_model(discriminator, opt.resume_epoch, opt, type='D')
        print('--------------------Pretrained Models are Loaded--------------------')

    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    train_set = utils.TrainSetLoader(dataset_dem=opt.dem_dir, dataset_rs=opt.rs_dir, dataset_mask=opt.mask_dir)
    dataloader = DataLoader(dataset=train_set, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True,
                            pin_memory=True, drop_last=True)
    # ----------------------------------------
    #            Training
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Training loop
    for epoch in range(opt.resume_epoch, opt.epochs):
        for batch_idx, (dem, rs, mask) in enumerate(dataloader):
            dem = Variable(dem).cuda()
            rs = Variable(rs).cuda()
            mask = Variable(mask).cuda()
            bad_dem = dem * (1 - mask) + mask

            # Train Discriminator
            optimizer_d.zero_grad()

            # Generator output
            refine_dem = generator(bad_dem, rs, mask)

            # forward propagation
            fill_dem = dem * (1 - mask) + refine_dem * mask  # in range [0, 1]
            # Fake samples
            fake_scalar = discriminator(fill_dem.detach(), mask)
            # True samples
            true_scalar = discriminator(dem, mask)

            # Loss and optimize
            loss_fake = adversarial_loss(fake_scalar, False, True)
            loss_true = adversarial_loss(true_scalar, True, True)
            # Overall Loss and optimize
            loss_D = 0.5 * (loss_fake + loss_true)
            loss_D.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()

            # L1 Loss
            first_L1Loss = L1Loss(refine_dem * mask, dem * mask)
            Style_L1loss = style_Loss(fill_dem, dem)
            second_L1Loss = L1Loss(fill_dem * mask, dem * mask)
            second_PerceptualLoss = perceptual_loss(fill_dem, dem)
            # GAN Loss
            fake_scalar = discriminator(fill_dem, mask)
            GAN_Loss = adversarial_loss(fake_scalar, True, False)

            # Compute losses
            loss = first_L1Loss + second_L1Loss + 0.1 * second_PerceptualLoss + 0.1 * GAN_Loss + 250 * Style_L1loss

            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f]" %
                  ((epoch + 1), opt.epochs, batch_idx, len(dataloader), first_L1Loss.item(), second_L1Loss.item()))
            print("\r[D Loss: %.5f] [G Loss: %.5f] [Perceptual Loss: %.5f] time_left: %s" %
                  (loss_D.item(), GAN_Loss.item(), second_PerceptualLoss.item(), time_left))
            print("\r[Style Loss: %.5f]" %
                  (Style_L1loss.item()))

        # Learning rate decrease
        adjust_learning_rate(opt.lr_g, optimizer_g, (epoch + 1), opt)
        adjust_learning_rate(opt.lr_d, optimizer_d, (epoch + 1), opt)

        # Save the model
        if (epoch + 1) % 10 == 0 or epoch == 0:
            save_model_generator(generator, (epoch + 1), opt)
