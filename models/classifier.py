import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils import getter
from .modules.aspp import ASPP_Bottleneck


class BaselineWithLinear(nn.Module):
    def __init__(self, extractor_cfg, hidden_dim, nclasses):
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
    def __init__(self, extractor_cfg, hidden_dim, nclasses):
        super().__init__()
        self.nclasses = nclasses
        self.extractor = getter.get_instance(extractor_cfg)
        self.hidden_dim = hidden_dim
        self.aspp = ASPP_Bottleneck(self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.nclasses)

    def forward(self, x):
        x = self.extractor.get_feature_map(x)
        x = self.aspp(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(-1, self.hidden_dim)
        return self.classifier(x)
