import os
import os.path as osp

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms



class DatasetLoader(Dataset):

    def __init__(self, setname, args=None):
        DATASET_DIR = os.path.join(args.data_dir, 'FC100/')
        if setname == 'train':
            THE_PATH = osp.join(DATASET_DIR, 'train')
            label_list = os.listdir(THE_PATH)
        elif setname == 'test':
            THE_PATH = osp.join(DATASET_DIR, 'test')
            label_list = os.listdir(THE_PATH)
        elif setname == 'val':
            THE_PATH = osp.join(DATASET_DIR, 'val')
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Wrong setname.')
        self.wnids = label_list

        data = []
        label = []

        if 'num_patch' not in vars(args).keys():
            print('no num_patch parameter, set as default: 9')
            self.num_patch = 9
        else:
            self.num_patch = args.num_patch

        folders = [osp.join(THE_PATH, label) for label in label_list if os.path.isdir(osp.join(THE_PATH, label))]

        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)
        self.data = data
        self.label = label
        self.num_class = len(set(label))

        image_size = 84
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                 np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        patch_list = []
        for _ in range(self.num_patch):
            patch_list.append(self.transform(Image.open(path).convert('RGB')))
        patch_list = torch.stack(patch_list, dim=0)
        return patch_list, label


if __name__ == '__main__':
    pass