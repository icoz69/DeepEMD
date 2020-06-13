import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch
import random
import numpy as np


class MiniImageNet(Dataset):

    def __init__(self, setname, args):
        IMAGE_PATH = os.path.join(args.data_dir, 'miniimagenet/images')
        SPLIT_PATH = os.path.join(args.data_dir, 'miniimagenet/split')
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1
        self.setname=setname
        self.wnids = []

        if 'patch_list' not in vars(args).keys():
            self.patch_list=[2,3]
            print('do not assign num_patch , set default:',self.patch_list)
        else:
            self.patch_list=args.patch_list

        if 'patch_ratio' not in vars(args).keys():
            self.patch_ratio = 2
            print('do not assign patch_ratio, set as default:',self.patch_ratio)
        else:
            self.patch_ratio=args.patch_ratio


        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data#data path of all data
        self.label = label #label of all data
        self.num_class = len(set(label))


        if setname=='val' or setname=='test':

            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([image_size,image_size]),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])) ])


        elif setname=='train':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([image_size,image_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
            ])

        else:
            raise ValueError('no such set')

    def __len__(self):
        return len(self.data)

    def get_grid_location(self,size, ratio, num_grid):
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



    def get_pyramid(self,img,num_patch):
        if self.setname == 'val' or self.setname == 'test':
            num_grid=num_patch
            grid_ratio=self.patch_ratio

        elif self.setname=='train':
            num_grid=num_patch
            grid_ratio=1+2*random.random()


        else:
            raise ValueError('Unkown set')



        w, h = img.size
        grid_locations_w=self.get_grid_location(w,grid_ratio,num_grid)
        grid_locations_h=self.get_grid_location(h,grid_ratio,num_grid)

        patches_list=[]
        for i in range(num_grid):
            for j in range(num_grid):
                patch_location_w=grid_locations_w[j]
                patch_location_h=grid_locations_h[i]
                left_up_corner_w=patch_location_w[0]
                left_up_corner_h=patch_location_h[0]
                right_down_cornet_w=patch_location_w[1]
                right_down_cornet_h = patch_location_h[1]
                patch=img.crop((left_up_corner_w,left_up_corner_h,right_down_cornet_w,right_down_cornet_h))
                patch=self.transform(patch)
                patches_list.append(patch)


        return patches_list


    def __getitem__(self, i):# return the ith data in the set.
        path, label = self.data[i], self.label[i]

        image=Image.open(path).convert('RGB')

        patch_list=[]
        for num_patch in self.patch_list:
            patches=self.get_pyramid(image,num_patch)
            patch_list.extend(patches)
        patch_list=torch.stack(patch_list,dim=0)

        return patch_list, label

if __name__ == '__main__':
    pass