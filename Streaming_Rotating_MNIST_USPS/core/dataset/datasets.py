import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


class RotateDigits(Dataset):
    # define static rotating dataset class for MNIST and USPS
    def __init__(self, dataset, split):
        super().__init__()
        splits = ['train', 'valid', 'test']
        datasets = ['MNIST', 'USPS']
        assert split in splits
        assert dataset in datasets
        self.data, self.target = torch.load(os.path.join("./data/static", dataset, split + ".pt"))
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img.numpy(), mode='L')
        img = self.transform(img)
        label = self.target[index]
        return img, label
    
    def __len__(self):
        return self.data.shape[0]
        

class MovRotateDigits(Dataset):
    # define streaming rotating dataset class for MNIST and USPS
    def __init__(self, dataset, split):
        super().__init__()
        splits = ['train', 'valid', 'test']
        datasets = ['MNIST', 'USPS']
        assert split in splits
        assert dataset in datasets
        self.data, self.target = torch.load(os.path.join("./data/stream", dataset, split + "_seq.pt"))

    def __getitem__(self, index):
        seq = self.data[index]
        seq = (seq - 0.5) / 0.5    # normalize ((0.5,), (0.5))
        label = self.target[index]
        return seq, label
    
    def __len__(self):
        return self.data.shape[0]