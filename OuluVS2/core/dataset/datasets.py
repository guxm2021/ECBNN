import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def OuluUniPadding(root="./data"):
    """
    Pad the sequence to 30/40 frames
    """
    save_folder = os.path.join(root, "unipadding")
    os.makedirs(save_folder, exist_ok=True)
    splits = ["train", "valid", "test"]
    for split in splits:
        if split == "valid":
            target_frames = 30
        else:
            target_frames = 40
        # load data
        ori_path = os.path.join(root, split + ".pt")
        ori_data, targets, angles = torch.load(ori_path)  # data: (num, frames, 44, 50)
        print(ori_data.shape)
        
        # compuet number of frames to be padded 
        pad_frames = target_frames - ori_data.shape[1]
        
        # replicate the first frame and the last frame
        pad_img_right = ori_data[:, [-1]].repeat(1, pad_frames, 1, 1)
        
        # concatenate data
        pad_data = torch.cat([ori_data, pad_img_right], dim=1)  # frame dimension

        # save data
        save_path = os.path.join(save_folder, split + ".pt")
        torch.save((pad_data, targets, angles), save_path)


class OuluLipReading(Dataset):
    """
    Dataset for Oulu Lip reading dataset
    """
    def __init__(self, path_to_data, select_angles={0, 30, 45, 60, 90}, shuffle=False):
        self.data, self.targets, self.angles = torch.load(path_to_data)
        mask = [(self.angles[i].item() in select_angles)
                for i in range(len(self.angles))]
        self.data = self.data[mask]
        self.targets = self.targets[mask]
        self.angles = self.angles[mask]
        if shuffle:
            permutation = np.random.permutation(len(self.targets))
            self.data = self.data[permutation]
            self.targets = self.targets[permutation]
            self.angles = self.angles[permutation]
        self.data = self.data.float()   # transform float64 into float32
        print(f"load data from {path_to_data} and data shape is: {self.data.shape}")

    def __getitem__(self, index):
        normalized_img = self.data[index]
        target = self.targets[index]
        angle_ratio = np.array([self.angles[index].numpy() / 90.0], dtype=np.float32)
        return normalized_img, target, angle_ratio, angle_ratio

    def __len__(self):
        return len(self.data)