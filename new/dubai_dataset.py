import numpy as np
from pathlib import Path
import torch
import json
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms

class DubaiImageDataset(Dataset):
    def __init__(self, folder_path):
        super(DubaiImageDataset, self).__init__()
        self.img_files = list(Path(folder_path).glob('*.jpg'))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(Path(img_path.parent, img_path.stem).with_suffix('.png'))
        
        with open(Path('dataset', 'classes.json'), 'r') as file:
            self.classes = json.loads(file.read())['classes']

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = read_image(img_path)
            mask = read_image(mask_path)
            label = np.zeros((len(self.classes), mask.shape[1], mask.shape[2]))
            for i, c in enumerate(self.classes):
                rcolor = int(c['color'][1:3], 16)
                test = np.array(mask[0] == rcolor).astype(np.int8)
                label[i][mask[0] == rcolor] = 1.0

            return data.float(), torch.Tensor(label).float()

    def __len__(self):
        return len(self.img_files)


if __name__ == '__main__':
    dataset = DubaiImageDataset('dataset/train')
    for i in range(dataset.__len__()):
        data, label = dataset.__getitem__(i)
        print(label.shape)
        print(data.shape)
