#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn.init as init
from deform_conv import DeformConv2d
from attention import *
from GateConv import *


def weights_init(net, init_type='kaiming', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class Generator(BaseNetwork):
    def __init__(self, init_weights=True, use_spectral_norm=True):
        super(Generator, self).__init__()
        self.begin_dem = Begin(1, 64)
        self.begin_rs = Begin(1, 64)
        self.registration = Registration(64*4)
        self.maxpool = nn.MaxPool2d(3, padding=1, stride=2)
        self.transfer = RPAB(256)
        self.kca = AttentionModule(256)


        blocks = []
        for _ in range(8):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        self.decoder_1 = nn.Sequential(
            SNGatedDeConv2dWithActivation(scale_factor=2, in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            # nn.ReLU(True)
        )

        self.decoder_2 = nn.Sequential(
            SNGatedDeConv2dWithActivation(scale_factor=2, in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            # nn.ReLU(True)
        )

        self.decoder_3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            SNGatedConv2dWithActivation(in_channels=64, out_channels=1, kernel_size=7, padding=0, activation=None)
        )

        if init_weights:
            self.init_weights()

    def forward(self, bad_dem, rs, mask):
        dem1, dem2, dem3 = self.begin_dem(bad_dem, mask)
        rs1, rs2, rs3 = self.begin_rs(rs, mask)

        registration_rs = self.registration(dem3, rs3)

        mask_down = self.maxpool(self.maxpool(mask))

        encode_dem = self.kca(dem3, registration_rs, mask_down)
        transfer_dem = self.transfer(encode_dem, registration_rs)

        middle_dem = self.middle(transfer_dem)
        decode_dem_1 = self.decoder_1(middle_dem+dem3)
        decode_dem_2 = self.decoder_2(decode_dem_1+dem2)
        decode_dem_3 = self.decoder_3(decode_dem_2+dem1)
        first_dem = torch.sigmoid(decode_dem_3)

        return first_dem


class PatchDiscriminator(BaseNetwork):
    def __init__(self, in_channels=2, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(PatchDiscriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, dem, mask):
        x = torch.cat((dem, mask), 1)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)
        return outputs


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            SNGatedConv2dWithActivation(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation,
                                    bias=not use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            # nn.ReLU(True),

            nn.ReflectionPad2d(1),
            SNGatedConv2dWithActivation(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,
                                    bias=not use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class Registration(nn.Module):
    def __init__(self, channels):
        super(Registration, self).__init__()

        self.conv_down_1 = nn.Conv2d(2*channels, 2*channels, 3, 2, 1)
        self.conv1 = nn.Conv2d(2*channels, channels, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(2*channels, channels, 3, 1, 1)
        self.conv_up_1 = nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1)
        self.conv_down_rs = nn.Conv2d(channels, channels, 3, 2, 1)
        self.conv_offset = nn.Conv2d(channels, channels, 1, 1, 0)

        self.conv3 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.deconv1 = DeformConv2d(channels, channels, 3, 1, 1)
        self.deconv2 = DeformConv2d(channels, channels, 3, 1, 1)
        self.conv_up_2 = nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1)

    def forward(self, dem, rs):

        fea_1 = torch.cat([dem, rs], 1)
        fea_2 = self.conv_down_1(fea_1)
        fea_1 = self.conv1(fea_1)
        fea_2 = self.conv1_2(fea_2)
        fea_2_1 = self.conv_up_1(fea_2)
        fea = self.conv_offset(fea_1 + fea_2_1)

        # print(right_1)
        fea_1_1 = self.deconv1(rs, fea)
        rs_2 = self.conv_down_rs(rs)
        fea_2 = self.deconv2(rs_2, fea_2)
        fea_2_1 = self.conv_up_2(fea_2)

        registration_right = fea_1_1 + fea_2_1
        registration_right = self.conv3(registration_right)
        return registration_right


class Begin(nn.Module):
    def __init__(self, input, channels):
        super(Begin, self).__init__()
        self.pad = nn.ReflectionPad2d(3)
        self.conv1 = GatedConv2dWithActivation(in_channels=input*2, out_channels=channels, kernel_size=7, stride=1, padding=0)
        self.conv2 = GatedConv2dWithActivation(in_channels=channels, out_channels=channels*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = GatedConv2dWithActivation(in_channels=channels*2, out_channels=channels*4, kernel_size=4, stride=2, padding=1)

    def forward(self, x, y):
        x1 = self.conv1(self.pad(torch.cat((x, y), 1)))
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x1, x2, x3


class RPAB(nn.Module):
    def __init__(self, channels):
        super(RPAB, self).__init__()
        self.rb = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels // 16, channels, 3, 1, 1),
        )

        self.encode = nn.Sequential(
            # nn.Conv2d(channels, channels, 1, 1, 0),
            nn.Conv2d(channels, channels//16, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels//16, channels, 1, 1, 0),
        )
        self.sig = nn.Sigmoid()

    def forward(self, dem, rs):

        buffer_dem = self.encode(dem)
        buffer_rs = self.rb(rs)
        rs_to_dem = self.sig(buffer_dem)
        buffer_1 = buffer_rs * rs_to_dem
        buffer = buffer_1 + dem

        return buffer
