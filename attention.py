import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeConsistentAttention(nn.Module):
    def __init__(self, patch_size = 5, propagate_size = 5, stride = 1):
        super(KnowledgeConsistentAttention, self).__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        self.prop_kernels = None
        self.att_scores_prev = None
        self.masks_prev = None
        self.ratio = nn.Parameter(torch.ones(1))

    def forward(self, dem, rs, masks):
        bz, nc, h, w = dem.size()
        foreground = dem
        # conv_kernels_all = rs.view(bz, nc, w * h, 1, 1)

        expanded_tensor_rs = torch.cat((rs[:, :, 0:1, :], rs), dim=2)
        rs = torch.cat((expanded_tensor_rs[:, :, :, 0:1], expanded_tensor_rs), dim=3)
        conv_kernels_all = rs.unfold(2, 5, 5).unfold(3, 5, 5).reshape(bz, nc, -1, 5, 5)
        conv_kernels_all = conv_kernels_all.permute(0, 2, 1, 3, 4)

        expanded_tensor_dem = torch.cat((foreground[:, :, 0:1, :], foreground), dim=2)
        foreground = torch.cat((expanded_tensor_dem[:, :, :, 0:1], expanded_tensor_dem), dim=3)
        conv_kernels = foreground.unfold(2, 5, 5).unfold(3, 5, 5).reshape(bz, nc, -1, 5, 5)
        conv_kernels = conv_kernels.permute(0, 2, 1, 3, 4)

        feature_map = rs

        conv_kernels_rs = conv_kernels_all.reshape(-1, nc, 5, 5) + 0.0000001
        norm_factor = torch.sum(conv_kernels_rs**2, [1, 2, 3], keepdim = True)**0.5
        conv_kernels_rs = conv_kernels_rs/norm_factor

        conv_kernels_dem = conv_kernels.reshape(-1, nc, 5, 5) + 0.0000001

        conv_result_rs = F.conv2d(feature_map, conv_kernels_rs, padding=2)
        attention_scores = F.softmax(conv_result_rs, dim=1)
        feature_map = F.conv_transpose2d(attention_scores, conv_kernels_dem, stride=1, padding=2)
        feature_map = feature_map[:, :, 1:65, 1:65]
        final_output = feature_map*masks + dem*(1-masks)
        return final_output


class AttentionModule(nn.Module):
    def __init__(self, inchannel, patch_size_list = [1], propagate_size_list = [3], stride_list = [1]):
        assert isinstance(patch_size_list, list), "patch_size should be a list containing scales, or you should use Contextual Attention to initialize your module"
        assert len(patch_size_list) == len(propagate_size_list) and len(propagate_size_list) == len(stride_list), "the input_lists should have same lengths"
        super(AttentionModule, self).__init__()

        self.att = KnowledgeConsistentAttention(patch_size_list[0], propagate_size_list[0], stride_list[0])
        self.num_of_modules = len(patch_size_list)
        self.combiner = nn.Conv2d(inchannel * 2, inchannel, kernel_size = 1)
        
    def forward(self, foreground, rs, mask):
        outputs = self.att(foreground, rs, mask)
        outputs = torch.cat([outputs, foreground],dim = 1)
        outputs = self.combiner(outputs)
        return outputs