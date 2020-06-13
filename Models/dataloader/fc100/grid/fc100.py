import os.path as osp
from PIL import Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
import torch
import random


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

        self.setname = setname
        self.wnids = label_list

        if 'patch_list' not in vars(args).keys():
            self.patch_list = [2, 3]
            print('do not assign parch_list , set as default:', self.patch_list)
        else:
            self.patch_list = args.patch_list

        if 'patch_ratio' not in vars(args).keys():
            self.patch_ratio = 2
            print('do not assign patch_ratio, set as default:', self.patch_ratio)
        else:
            self.patch_ratio = args.patch_ratio

        data = []
        label = []

        folders = [osp.join(THE_PATH, label) for label in label_list if os.path.isdir(osp.join(THE_PATH, label))]

        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if setname == 'val' or setname == 'test':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))

            ])


        elif setname == 'train':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
            ])

    def __len__(self):
        return len(self.data)

    def get_grid_location(self, size, ratio, num_grid):
        '''

        :param size: size of the height/width
        :param ratio: generate grid size/ even divided grid size
        :param num_grid: number of grid
        :return: a list containing the coordinate of the grid
        '''
        raw_grid_size = int(size / num_grid)
        enlarged_grid_size = int(size / num_grid * ratio)

        center_location = raw_grid_size // 2

        location_list = []
        for i in range(num_grid):
            location_list.append((max(0, center_location - enlarged_grid_size // 2),
                                  min(size, center_location + enlarged_grid_size // 2)))
            center_location = center_location + raw_grid_size

        return location_list

    def get_pyramid(self, img, num_patch):
        if self.setname == 'val' or self.setname == 'test':
            num_grid = num_patch
            grid_ratio = self.patch_ratio

        elif self.setname == 'train':
            num_grid = num_patch
            grid_ratio = 1 + 2 * random.random()


        else:
            raise ValueError('Unkown set')

        w, h = img.size
        grid_locations_w = self.get_grid_location(w, grid_ratio, num_grid)
        grid_locations_h = self.get_grid_location(h, grid_ratio, num_grid)

        patches_list = []
        for i in range(num_grid):
            for j in range(num_grid):
                patch_location_w = grid_locations_w[j]
                patch_location_h = grid_locations_h[i]
                left_up_corner_w = patch_location_w[0]
                left_up_corner_h = patch_location_h[0]
                right_down_cornet_w = patch_location_w[1]
                right_down_cornet_h = patch_location_h[1]
                patch = img.crop((left_up_corner_w, left_up_corner_h, right_down_cornet_w, right_down_cornet_h))
                patch = self.transform(patch)
                patches_list.append(patch)

        return patches_list

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]

        image = Image.open(path).convert('RGB')

        patch_list = []
        for num_patch in self.patch_list:
            patches = self.get_pyramid(image, num_patch)
            patch_list.extend(patches)
        patch_list = torch.stack(patch_list, dim=0)

        return patch_list, label


if __name__ == '__main__':
    pass