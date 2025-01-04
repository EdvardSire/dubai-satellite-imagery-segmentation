from pathlib import Path
import sys

import torch
import cv2
from torch.utils.data import Dataset


def show_CHW_image(image, window_name="window"):
    image = image.permute(1, 2, 0).numpy()  # Convert to (H, W, C)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class DubaiDatasetBatchless(Dataset):
    """
    cv2.imread return HWC
    torchvision.io.decode_image return CHW
    for training we want NCHW
    """

    def __init__(self, folder_path):
        super(DubaiDatasetBatchless, self).__init__()
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

        for index, mask in enumerate(self.unprocessed_masks):
            first_image_channel = mask[0]
            tensor_to_match = torch.tensor(self.classes)
            matches = (first_image_channel.unsqueeze(-1) == tensor_to_match).any(dim=-1)
            non_matches_count = (~matches).sum().item()

            print(f'{index} has {non_matches_count} non-matches')
            print(first_image_channel[~matches])
            print()
            sys.stdout.flush()
        




    def __getitem__(self, index):
        return None


    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = DubaiDatasetBatchless(Path(__file__).parent.parent / 'dataset' / 'train')

    # for mask in dataset.unprocessed_masks:
    #     show_CHW_image(mask)

