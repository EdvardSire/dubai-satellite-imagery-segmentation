from pathlib import Path
import torch
from torch.utils.data import Dataset

class DubaiImageDataset(Dataset):
    def __init__(self, folder_path):
        super(DubaiImageDataset, self).__init__()
        self.img_files = list(Path(folder_path).glob('*.jpg'))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(Path(img_path.parent, img_path.stem).with_suffix('.png'))
        self.classes = [60, 132, 110, 254, 226, 155]

    def __getitem__(self, index):
        return



    def __len__(self):
        return 40


if __name__ == '__main__':
    dataset = DubaiImageDataset('dataset/train')
