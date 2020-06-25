import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils import getter
from .modules.aspp import ASPP


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
    def __init__(self, extractor_cfg, aspp_hidden_dim, aspp_output_dim, nclasses):
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
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(-1, self.aspp_output_dim)
        return self.classifier(x)
