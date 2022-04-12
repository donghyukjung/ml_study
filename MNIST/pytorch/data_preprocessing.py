import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob


def get_label_from_path(DATA_PATH_LIST):
    label_list = []
    for path in DATA_PATH_LIST:
        label = path.split('/')[-2]
        label_list.append(label)
    return label_list


class mnistDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = path
        self.num_path = [''for i in range(10)]
        self.num_list = [[]for i in range(10)]
        if train:
            for i in range(10):
                self.num_path[i] = path+'/train/'+str(i)
        else:
            for i in range(10):
                self.num_path[i] = path+'/test/'+str(i)

        for i in range(10):
            self.num_list[i] = glob.glob(self.num_path[i]+'/*.png')
        self.transform = transform
        self.img_list = []
        self.class_list = []
        for i in range(10):
            self.img_list.extend(self.num_list[i])
            self.class_list.extend([i]*len(self.num_list[i]))
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label = self.class_list[idx]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label
