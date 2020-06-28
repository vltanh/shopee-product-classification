# Credit: https://github.com/fregu856/deeplabv3/s

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.conv_1x1_1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(hidden_dim)

        self.conv_3x3_1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3,
                                    stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(hidden_dim)

        self.conv_3x3_2 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3,
                                    stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(hidden_dim)

        self.conv_3x3_3 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3,
                                    stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(hidden_dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(hidden_dim)

        self.conv_1x1_3 = nn.Conv2d(5 * hidden_dim, output_dim, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(output_dim)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.interpolate(out_img,
                                size=(feature_map_h, feature_map_w),
                                mode="bilinear",
                                align_corners=False)

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))

        return out
