import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.getter import get_instance

class BaselineClassifier(nn.Module):
    def __init__(self, extractor_cfg, nclasses):
        self.nclasses = nclasses
        self.extractor = get_instance(extractor_cfg)
        self.feature_dims = self.extractor.feature_dims
        self.classifier = nn.Linear(self.feature_dims, self.nclasses)

    def forward(self, x):
        x = self.extractor(x)
        return self.classifier()
