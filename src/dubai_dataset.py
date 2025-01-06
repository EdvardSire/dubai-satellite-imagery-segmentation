from pathlib import Path
import sys

import torch
import cv2
from torch.utils.data import Dataset


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

    def preview_images_and_masks():
        for i in range(dataset.__len__()):
            image, mask = dataset.__getitem__(i) # single batch item
            mask = mask.unsqueeze(0)
            show_CHW_image(image)
            show_CHW_image(mask)
    preview_images_and_masks()

    def visualize_masks_for_training():
        for i in range(dataset.__len__()):
            _, mask = dataset.__getitem__(i) # single batch item 
            show_HW_image(mask)
            for layer in mask:
                show_HW_image(layer)
    visualize_masks_for_training()





