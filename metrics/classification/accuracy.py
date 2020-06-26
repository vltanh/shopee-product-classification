import torch
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


class Accuracy():
    def __init__(self, *args, **kwargs):
        self.reset()

    def calculate(self, output, target):
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum()
        sample_size = output.size(0)
        return correct, sample_size

    def update(self, value):
        self.correct += value[0]
        self.sample_size += value[1]

    def reset(self):
        self.correct = 0.0
        self.sample_size = 0.0

    def value(self):
        return self.correct / self.sample_size

    def summary(self):
        print(f'Accuracy: {self.value()}')


class ConfusionMatrix():
    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.reset()

    def calculate(self, output, target):
        pred = torch.argmax(output, dim=1)
        return confusion_matrix(target.cpu().numpy(),
                                pred.cpu().numpy(),
                                labels=range(self.nclasses))

    def update(self, value):
        self.cm += value

    def reset(self):
        self.cm = np.zeros(shape=(self.nclasses, self.nclasses))

    def value(self):
        return None

    def summary(self):
        print('Confusion Matrix:')
        print(self.cm)


class ClassificationReport():
    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.reset()

    def calculate(self, output, target):
        pred = torch.argmax(output, dim=1)
        return pred, target

    def update(self, value):
        self.preds += value[0].detach().cpu().numpy().tolist()
        self.labels += value[1].detach().cpu().numpy().tolist()

    def reset(self):
        self.preds = []
        self.labels = []

    def value(self):
        return None

    def summary(self):
        print(classification_report(self.labels,
                                    self.preds,
                                    labels=range(self.nclasses)))
