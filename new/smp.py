import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="inceptionresnetv2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=6,                      # model output channels (number of classes in your dataset)
)
model.encoder.load_state_dict(torch.load("inceptionresnetv2-imagenet.pth", weights_only=True))

img = Image.open('../image.png')



tfms = transforms.Compose([
        #transforms.Resize(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

img_tensor = tfms(img).unsqueeze(0)



output = model.forward(img_tensor)

output = output.squeeze().detach().numpy()

for i in range(6):
    pred_mask = np.asarray(output[i], dtype=np.uint8)*255
    #print(output)
    plt.imshow(pred_mask)
    plt.show()

