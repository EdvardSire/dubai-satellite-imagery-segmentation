from dubai_dataset import DubaiDatasetBatchless

from pathlib import Path
import math
import sys
import torch
from torchvision.transforms import v2

import segmentation_models_pytorch as smp


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

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'; print(f'Device type: {device}')
    train_path = Path(__file__).parent.parent / 'dataset' / 'train'
    val_path = Path(__file__).parent.parent / 'dataset' / 'val'
    train_dataset = DubaiDatasetBatchless(train_path, device)
    val_dataset = DubaiDatasetBatchless(val_path, device)

    model = smp.Unet(
        encoder_name="inceptionresnetv2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=6,                      # model output channels (number of classes in your dataset)
    )
    weights = torch.load("weights/inceptionresnetv2-520b38e4.pth", weights_only=True)
    model.encoder.load_state_dict(weights)


    model.eval().to(device)
    image, mask = train_dataset.__getitem__(0)
    image_tfs = v2.Compose([
        v2.Resize(make_divisible_by_32(image.shape), interpolation=v2.InterpolationMode.BILINEAR)
        ])
    mask_tfs = v2.Compose([
        v2.Resize(make_divisible_by_32(mask.shape), interpolation=v2.InterpolationMode.NEAREST)
        ])
    image = image_tfs(image)
    mask = mask_tfs(mask.unsqueeze(0))

    NUM_EPOCHS = 2
    loss_function = torch.nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print(optimizer)


    model.train()
    for e in range(NUM_EPOCHS):
        running_loss = 0.
        last_loss = 0.
        for i, (images, masks) in enumerate(train_dataset):
            images = image_tfs(images)
            masks = image_tfs(masks)

            outputs = model.forward(images)
            loss = loss_function(outputs, masks)
            loss.backward()
            optimizer.step()
            print(loss)
            sys.stdout.flush()
