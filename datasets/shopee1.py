import torchvision.transforms as tvtf
from torch.utils import data
from PIL import Image

from .autoaugment import ImageNetPolicy

import csv
import os


class ShopeeDataset1(data.Dataset):
    def __init__(self, img_dir, csv_path, is_train):
        self.img_dir = img_dir

        lines = csv.reader(open(csv_path))
        next(lines)
        self.samples = list(lines)

        if is_train:
            self.transforms = tvtf.Compose([
                tvtf.Resize((224, 224)),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = tvtf.Compose([
                tvtf.Resize((224, 224)),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            ])

    def __getitem__(self, index):
        fn, label = self.samples[index]
        label = int(label)

        img_path = os.path.join(self.img_dir, fn)
        im = Image.open(img_path).convert('RGB')
        im = self.transforms(im)

        return im, label

    def __len__(self):
        return len(self.samples)
