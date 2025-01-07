from dubai_dataset import DubaiDatasetBatchless
from dubai_dataset import show_CHW_image

from pathlib import Path
import torch


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'; print(f'Device type: {device}')
    val_path = Path(__file__).parent.parent / 'dataset' / 'val'
    val_dataset = DubaiDatasetBatchless(val_path, device)

    
    model = torch.load(Path(__file__).parent.parent / 'runs' / 'exp_batchless_preprocessing_2/model.pt' )
    model.eval()
    with torch.no_grad():
        image, mask = val_dataset.__getitem__(0)
        image = val_dataset.image_tfs(image)
        mask = val_dataset.mask_tfs(mask)

        output = model.forward(image)
        print(output.shape)


        show_CHW_image(image.squeeze().cpu())
        for layer in output.squeeze():
            print(layer.shape, layer.dtype)
            print(torch.unique(layer))

            # cv2.imshow("", layer.detach().cpu().numpy())
            # cv2.waitKey(0)
        

