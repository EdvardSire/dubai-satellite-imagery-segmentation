import numpy as np
from pathlib import Path
import torch
import json
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

class DubaiImageDataset(Dataset):
    def __init__(self, folder_path):
        super(DubaiImageDataset, self).__init__()
        self.img_files = list(Path(folder_path).glob('*.jpg'))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(Path(img_path.parent, img_path.stem).with_suffix('.png'))
        self.classes = [60, 132, 110, 254, 226, 155]

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = cv2.imread(img_path, cv2.IMREAD_COLOR)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            data = data.transpose(2,0,1)
            mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = mask.transpose(2,0,1)
            label = np.zeros((len(self.classes), mask.shape[1], mask.shape[2]))
            for i, c in enumerate(self.classes):
                label[i][mask[0] == c] = 255.0

            return torch.Tensor(data).float(), torch.Tensor(label).float()

    def __len__(self):
        return len(self.img_files)


if __name__ == '__main__':
    dataset = DubaiImageDataset('dataset/train')
    data, label = dataset.__getitem__(0)
    import matplotlib.pyplot as plt
    plt.imshow(data.permute(1,2,0))
    plt.show()
    mask = cv2.imread(dataset.mask_files[0])
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = mask.transpose(2,0,1)
    plt.imshow(mask[0])
    plt.show()

    rcolor = 226
    l = np.zeros((mask.shape[1], mask.shape[2]))
    l[mask[0]==rcolor] = 255.0
    plt.imshow(l)
    plt.show()
    #for i in range(dataset.__len__()):
    #    data, label = dataset.__getitem__(i)
    #    print(label.shape)
    #    print(data.shape)
