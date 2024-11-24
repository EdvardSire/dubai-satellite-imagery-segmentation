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

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy().astype(np.int32)
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(training_loader)
images, labels = next(dataiter)

# Create a grid from the images and show them
# img_grid = torchvision.utils.make_grid(images)
# matplotlib_imshow(img_grid, one_channel=False)

# img_grid = torchvision.utils.make_grid(labels)
# matplotlib_imshow(img_grid, one_channel=False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = smp.Unet(
    encoder_name="inceptionresnetv2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=6,                      # model output channels (number of classes in your dataset)
)
weights = torch.load("inceptionresnetv2-imagenet.pth", weights_only=True)
model.encoder.load_state_dict(weights)
model = model.to(DEVICE)

lossFunc = BCEWithLogitsLoss()
opt = Adam(model.parameters(), lr=0.5)

H = {"train_loss": [], "test_loss": []}

NUM_EPOCHS = 100
trainSteps = len(training_set)

print("[INFO] training the network...")
startTime = time.time()
for e in range(NUM_EPOCHS):
	model.train()
	totalTrainLoss = 0
	totalTestLoss = 0
	for (i, (x, y)) in tqdm(enumerate(training_loader)):
		(x, y) = (x.to(DEVICE), y.to(DEVICE))

		shape = x.shape[::-1]
		new_height = (shape[0] // 32) * 32
		new_width = (shape[1] // 32) * 32

		tfms = transforms.Compose([
		    transforms.Resize((new_width, new_height)),
		    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])
		tfms_y = transforms.Compose([
            transforms.Resize((new_width, new_height))
        ])
		x = tfms(x)
		y = tfms_y(y)

		pred = model(x)
		loss = lossFunc(pred, y)

		opt.zero_grad()
		loss.backward()
		opt.step()

		totalTrainLoss += loss
	
	avgTrainLoss = totalTrainLoss / trainSteps

	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())

	print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
	print("Train loss: {:.6f}".format(avgTrainLoss))

endTime = time.time()

print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

torch.save(model.state_dict(), "model.pt")

image = torchvision.io.read_image('image.png')
tfms = transforms.Compose([
          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

img_tensor = tfms(image.float())


model.eval()
output = model.forward(img_tensor.to(DEVICE).unsqueeze(0))

output = output.cpu().squeeze().detach().numpy()

for i in range(6):
    pred_mask = np.asarray(output[i, :, :])
    #print(output)
    plt.imshow(pred_mask)
    plt.show()

