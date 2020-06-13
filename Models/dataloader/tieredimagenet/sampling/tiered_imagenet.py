import os
import os.path as osp

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class tieredImageNet(Dataset):

    def __init__(self, setname, args=None):
        TRAIN_PATH = osp.join(args.data_dir, 'tiered_imagenet/train')
        VAL_PATH = osp.join(args.data_dir, 'tiered_imagenet/val')
        TEST_PATH = osp.join(args.data_dir, 'tiered_imagenet/test')
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        elif setname == 'test':
            THE_PATH = TEST_PATH
        elif setname == 'val':
            THE_PATH = VAL_PATH
        else:
            raise ValueError('Unkown setname.')
        data = []
        label = []
        folders = [osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if
                   os.path.isdir(osp.join(THE_PATH, label))]
        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        if 'num_patch' not in vars(args).keys():
            self.num_patch = 9
            print('no num_patch parameter, set as default:',self.num_patch)
        else:
            self.num_patch = args.num_patch

        # Transformation
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

    def __getitem__(self, i):  # return the ith data in the set.
        path, label = self.data[i], self.label[i]
        patch_list = []
        for _ in range(self.num_patch):
            patch_list.append(self.transform(Image.open(path).convert('RGB')))

        patch_list = torch.stack(patch_list, dim=0)

        return patch_list, label


if __name__ == '__main__':
    pass
