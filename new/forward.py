import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import torchvision
from torchvision import transforms
import torchvision
import segmentation_models_pytorch as smp

from dubai_dataset import DubaiImageDataset

training_set    = DubaiImageDataset('dataset/train')
validation_set  = DubaiImageDataset('dataset/val')

training_loader     = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True)
validation_loader   = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = smp.Unet(
    encoder_name="inceptionresnetv2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=6,                      # model output channels (number of classes in your dataset)
)
model.load_state_dict(torch.load("model.pt"))
model = model.to(DEVICE)

image = torchvision.io.read_image('image.png')
plt.imshow(image.permute(1,2,0))
plt.show()
#tfms = transforms.Compose([
#          transforms.ToTensor(),
#          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#img_tensor = tfms(image).unsqueeze(0)


model.eval()



tfms = transforms.Compose([
          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

img_tensor = tfms(image.float()).unsqueeze(0)

output = model.forward(image.float().to(DEVICE).unsqueeze(0))

output = output.squeeze().cpu().detach().numpy()

for i in range(6):
     pred_mask = np.asarray(output[i], dtype=np.uint8)*255
     #print(output)
     plt.imshow(pred_mask)
     plt.show()

