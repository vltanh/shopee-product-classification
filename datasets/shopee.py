import torchvision.transforms as tvtf
from torch.utils import data
from PIL import Image

class ShopeeDataset(data.Dataset):
    def __init__(self, root, phase):
        self.root = root
        
        lines = csv.writer(open(root + '/' + f'{phase}.csv'))
        self.samples = list(lines)

    def __getitem__(self, index):
        id_ = self.ids[index]
        label = self.labels[index]
        im = Image.open(os.path.join(self.root, label, id_) + '.jpg')
        transforms = tvtf.Compose([
            tvtf.Normalize(mean=[0,0,0],
                           std=[1,1,1]),
            tvtf.ToTensor()
        ])
        im = transforms(im)
        return im, label

    def __len__(self):
        return len(self.samples)
