from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import decode_image
import cv2
import torch
import sys


def show_torchvision_image(image, window_name="window"):
    image = image.permute(1, 2, 0).numpy()  # Convert to (H, W, C)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class DubaiDatasetBatchless(Dataset):
    def __init__(self, folder_path):
        super(DubaiDatasetBatchless, self).__init__()
        self.img_files = list(Path(folder_path).glob('*.jpg'))
        self.samples = []
        for img_file in self.img_files:
            mask_file = Path(img_file.parent, img_file.stem).with_suffix('.png')
            self.samples.append((decode_image(img_file.__str__()), decode_image(mask_file.__str__())))

        self.classes = [60, 132, 110, 254, 226, 155]


        show_torchvision_image(self.samples[46][1])

        for sample in self.samples:
            show_torchvision_image(sample[1])

        # ### Run this snippet to get an idea of the mistakes in the masks
        # # show_torchvision_image(self.samples[46][1])
        # for index, sample in enumerate(self.samples):
        #     first_image_channel = sample[1][0]
        #     print(first_image_channel.shape)
        #     tensor_to_match = torch.tensor(self.classes)
        #     matches = (first_image_channel.unsqueeze(-1) == tensor_to_match).any(dim=-1)
        #     non_matches_count = (~matches).sum().item()
        #     print(f'{index} has {non_matches_count} non-matches')
        #     print(first_image_channel[~matches])
        #     print()
        #     sys.stdout.flush()





    def __getitem__(self, index):
        return self.samples[index][0].unsqueeze_(0), None


    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    dataset = DubaiDatasetBatchless(Path(__file__).parent.parent / 'dataset' / 'train')
    img, label = dataset.__getitem__(0)
    print(img.shape)


    # img = decode_image(dataset.img_files[0].__str__())
    # mask = decode_image(dataset.img_files[0].__str__().replace("jpg", "png"))
    # show_torchvision_image(img)
    # show_torchvision_image(mask)


