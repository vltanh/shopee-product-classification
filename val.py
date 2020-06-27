import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as tvtf
from tqdm import tqdm

from datasets.shopee import ShopeeDataset
from metrics.classification.accuracy import Accuracy, ConfusionMatrix
from utils.getter import get_instance
from utils.device import move_to

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str,
                    help='path to the folder of query images')
parser.add_argument('-c', type=str,
                    help='path to the csv file')
parser.add_argument('-w', type=str,
                    help='path to weight files')
parser.add_argument('-g', type=int, default=None,
                    help='(single) GPU to use (default: None)')
parser.add_argument('-b', type=int, default=64,
                    help='batch size (default: 64)')
args = parser.parse_args()

# Device
dev_id = 'cuda:{}'.format(args.g) \
    if torch.cuda.is_available() and args.g is not None \
    else 'cpu'
device = torch.device(dev_id)

# Load model
config = torch.load(args.w, map_location=dev_id)
model = get_instance(config['config']['model']).to(device)
model.load_state_dict(config['model_state_dict'])

# Load data
dataset = ShopeeDataset(img_dir=args.d, csv_path=args.c, is_train=False)
dataloader = DataLoader(dataset, batch_size=args.b)

# Metrics
metrics = {
    'Accuracy': Accuracy(),
    'ConfusionMatrix': ConfusionMatrix(nclasses=42),
}

with torch.no_grad():
    for m in metrics.values():
        m.reset()

    model.eval()
    progress_bar = tqdm(dataloader)
    for i, (inp, lbl) in enumerate(progress_bar):
        inp = move_to(inp, device)
        lbl = move_to(lbl, device)
        outs = model(inp)
        for m in metrics.values():
            value = m.calculate(outs, lbl)
            m.update(value)

    print('+ Evaluation result')
    for m in metrics.values():
        m.summary()
