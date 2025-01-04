from pathlib import Path
from torchvision.io import decode_image
import cv2

# Run with `python3 torchvision_decode_image_bug.py | sort`
if __name__ == "__main__":
    img_files = list(Path("dataset/train").glob('*.jpg'))
    for img_file in img_files:
        mask_file = Path(img_file.parent, img_file.stem).with_suffix('.png')
        img = decode_image(img_file.__str__())
        mask = decode_image(mask_file.__str__())
        cv_mask = cv2.imread(mask_file.__str__())
        print("torchvision.io.decode_image:", mask.shape, "cv2.imread:", cv_mask.shape, mask_file.name)

