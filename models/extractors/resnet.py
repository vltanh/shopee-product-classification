import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .extractor_network import ExtractorNetwork


class ResNetExtractor(ExtractorNetwork):
    arch = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'wide_resnet50_2': models.wide_resnet50_2,
        'wide_resnet101_2': models.wide_resnet101_2,
    }

    def __init__(self, version):
        super().__init__()
        assert version in ResNetExtractor.arch, \
            f'{version} is not implemented.'
        cnn = ResNetExtractor.arch[version](pretrained=True)
        self.extractor = nn.Sequential(*list(cnn.children())[:-2])
        self.feature_dim = cnn.fc.in_features

    def forward(self, x):
        x = self.get_feature_map(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, self.feature_dim)
        return x

    def get_feature_map(self, x):
        return self.extractor(x)
