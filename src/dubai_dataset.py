import sys
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import cv2


def show_CHW_image(image, window_name='window'):
    image = image.permute(1, 2, 0).numpy()  # (C, H, W) to (H, W, C)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_HW_image(image, window_name='window'):
    image = image.numpy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def make_divisible_by_32(size):
    if len(size) == 4:  # BCHW
        _, _, height, width = size
    elif len(size) == 3:  # CHW
        _, height, width = size
    else:
        assert len(size) == 2
        height, width = size

    height = math.ceil(height / 32) * 32
    width = math.ceil(width / 32) * 32
    return (height, width)


class DubaiDatasetBatchless(Dataset):
    '''
    cv2.imread return HWC
    torchvision.io.decode_image return CHW
    for training we want NCHW
    '''

    def __init__(self, folder_path, device='cpu'):
        super(DubaiDatasetBatchless, self).__init__()
        self.device = device
        self.image_paths = list(Path(folder_path).glob('*.jpg'))
        self.classes = [60, 132, 110, 254, 226, 155]
        self.images = list()
        self.unprocessed_masks = list()
        self.processed_masks = list()
        def image_tfs(image):
            tf = v2.Compose([
                v2.Resize(make_divisible_by_32(image.shape), interpolation=v2.InterpolationMode.BILINEAR),
                # https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionresnetv2.py
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            return tf(image)

        def mask_tfs(mask):
            tf = v2.Compose([
                v2.Resize(make_divisible_by_32(mask.shape), interpolation=v2.InterpolationMode.NEAREST)
            ])
            return tf(image)


        for image_file in self.image_paths:
            mask_file = Path(image_file.parent, image_file.stem).with_suffix('.png')
            image = cv2.imread(image_file.__str__()) # HWC, np.uint8
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_file.__str__()) # HWC, np.uint8
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            image = torch.tensor(image.transpose(2,0,1)) # CHW, torch.uint8
            mask = torch.tensor(mask.transpose(2,0,1)) # CHW, torch.uint8
            self.images.append(image)
            self.unprocessed_masks.append(mask)

        ###  Fix 0's in the masks 
        UNLABELED = 155
        DEBUG_MASK_VALUES = False
        if DEBUG_MASK_VALUES:
            for index, mask in enumerate(self.unprocessed_masks):
                first_image_channel = mask[0]
                tensor_to_match = torch.tensor(self.classes)
                matches = (first_image_channel.unsqueeze(-1) == tensor_to_match).any(dim=-1)
                non_matches_count = (~matches).sum().item()

                print(f'{index} has {non_matches_count} non-matches')
                print(first_image_channel[~matches])
                print()
                sys.stdout.flush()

        for mask in self.unprocessed_masks:
            first_image_channel = mask[0]
            first_image_channel[first_image_channel == 0] = UNLABELED # H*W
            h, w = first_image_channel.shape
            mask_per_class = torch.zeros((len(self.classes), h, w)) # NUM_CLASSES*H*W
            for i, c in enumerate(self.classes):
                mask_per_class[i][first_image_channel == c] = 1
            self.processed_masks.append(mask_per_class)



    def __getitem__(self, index):
        # TODO: proper preprocessing
        return (self.images[index]/255).to(self.device).unsqueeze(0), self.processed_masks[index].to(self.device).unsqueeze(0)


    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = DubaiDatasetBatchless(Path(__file__).parent.parent / 'dataset' / 'train')

    def preview_images():
        for i in range(dataset.__len__()):
            image, mask = dataset.__getitem__(i) # single batch item
            show_CHW_image(image.squeeze())
    # preview_images()

    def visualize_masks_for_training():
        for i in range(dataset.__len__()):
            _, mask = dataset.__getitem__(i) # single batch item 
            print(mask)
            sys.stdout.flush()
            for layer in mask.squeeze():
                show_HW_image(layer)
    visualize_masks_for_training()





