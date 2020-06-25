import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils import getter
from .modules.aspp import ASPP


class BaselineWithLinear(nn.Module):
    def __init__(self, extractor_cfg, nclasses, hidden_dim):
        super().__init__()
        self.nclasses = nclasses
        self.extractor = getter.get_instance(extractor_cfg)
        self.feature_dim = self.extractor.feature_dim
        self.hidden_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.nclasses)
        )

    def forward(self, x):
        x = self.extractor(x)
        return self.classifier(x)


class BaselineWithASPP(nn.Module):
    def __init__(self, extractor_cfg, nclasses, aspp_hidden_dim, aspp_output_dim):
        super().__init__()
        self.nclasses = nclasses
        self.extractor = getter.get_instance(extractor_cfg)
        self.feature_dim = self.extractor.feature_dim
        self.aspp_hidden_dim = aspp_hidden_dim
        self.aspp_output_dim = aspp_output_dim
        self.aspp = ASPP(self.feature_dim,
                         self.aspp_hidden_dim,
                         self.aspp_output_dim)
        self.classifier = nn.Linear(self.aspp_output_dim, self.nclasses)

    def forward(self, x):
        x = self.extractor.get_feature_map(x)
        x = self.aspp(x)
        x = F.adaptive_avg_pool2d(x, 1).view(-1, self.aspp_output_dim)
        return self.classifier(x)


class BaselineWithAttention(nn.Module):
    def __init__(self, extractor_cfg, nclasses, nheads):
        super().__init__()
        self.nclasses = nclasses
        self.extractor = getter.get_instance(extractor_cfg)
        self.feature_dim = self.extractor.feature_dim
        self.attn = nn.MultiheadAttention(self.feature_dim, nheads)
        self.classifier = nn.Linear(self.feature_dim, self.nclasses)

    def forward(self, x):
        x = self.extractor.get_feature_map(x)
        B, D, H, W = x.shape
        x = x.reshape(B, D, -1)  # x: B, D, HW
        x = x.permute(2, 0, 1)  # x: HW, B, D
        x, p = self.attn(x, x, x)  # x: HW, B, D
        x = x.transpose(0, 1)  # x: B, HW, D
        x = torch.mean(x, dim=1)  # x: B, D
        return self.classifier(x)
