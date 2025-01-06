from dubai_dataset import DubaiDatasetBatchless
from pathlib import Path
import torch


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'; print(f'Device type: {device}')
    train_path = Path(__file__).parent.parent / 'dataset' / 'train'
    val_path = Path(__file__).parent.parent / 'dataset' / 'val'
    train_dataset = DubaiDatasetBatchless(train_path, device)
    val_dataset = DubaiDatasetBatchless(val_path, device)


