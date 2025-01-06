from dubai_dataset import DubaiDatasetBatchless

from pathlib import Path
import math
import sys
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import v2

import segmentation_models_pytorch as smp
from tqdm import tqdm


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

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'; print(f'Device type: {device}')
    train_path = Path(__file__).parent.parent / 'dataset' / 'train'
    val_path = Path(__file__).parent.parent / 'dataset' / 'val'
    train_dataset = DubaiDatasetBatchless(train_path, device)
    val_dataset = DubaiDatasetBatchless(val_path, device)

    model = smp.Unet(
        encoder_name='inceptionresnetv2',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=6,                      # model output channels (number of classes in your dataset)
    )
    weights = torch.load('weights/inceptionresnetv2-520b38e4.pth', weights_only=True)
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

    loss_function = torch.nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    LOGDIR=Path(__file__).parent.parent / 'runs'; LOGDIR.mkdir(exist_ok=True)
    paths = [path for path in LOGDIR.iterdir() if path.name.startswith('exp')]
    try:
        iternum = 1+int(max([iternum.__str__().split('_')[1] for iternum in paths]))
    except:
        iternum = 1

    writer = SummaryWriter(log_dir='runs/exp_batchless_no_preprocessing_{}'.format((iternum)))

    step = 0
    NUM_EPOCHS = 10
    for e in tqdm(range(NUM_EPOCHS)):
        running_loss = 0.0
        last_loss = 0.0

        model.train()
        for i, (images, masks) in enumerate(train_dataset):
            images = image_tfs(images)
            masks = image_tfs(masks)

            outputs = model.forward(images)
            loss = loss_function(outputs, masks)
            loss.backward()
            optimizer.step()
            step += 1

            if writer:
                writer.add_scalar('Loss/train', loss, step)

        model.eval()  
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_dataset:
                images = image_tfs(images)
                masks = mask_tfs(masks)

                outputs = model(images)
                loss = loss_function(outputs, masks)
                val_loss += loss.item()

        if writer:
            writer.add_scalar('Loss/val', val_loss, e)


    #done epochs
    torch.save(model, Path(writer.get_logdir()) / Path('model').with_suffix('.pt'))

