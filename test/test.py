import glob 
import torch 
from torchvision import transforms 
from PIL import Image 
from torch.utils.data import Dataset, DataLoader 
class catdogDataset(Dataset): 
    def __init__(self, path, train=True, transform=None): 
        self.path = path 
        if train: 
            self.cat_path = path + '/cat/train' 
            self.dog_path = path + '/dog/train' 
        else: 
            self.cat_path = path + '/cat/test' 
            self.dog_path = path + '/dog/test' 
        self.cat_img_list = glob.glob(self.cat_path + '/*.png') 
        self.dog_img_list = glob.glob(self.dog_path + '/*.png') 
        self.transform = transform 
        self.img_list = self.cat_img_list + self.dog_img_list 
        self.class_list = [0] * len(self.cat_img_list) + [1] * len(self.dog_img_list) 
    def __len__(self): 
        return len(self.img_list) 
    def __getitem__(self, idx): 
        img_path = self.img_list[idx] 
        label = self.class_list[idx] 
        img = Image.open(img_path) 
        if self.transform is not None: 
            img = self.transform(img) 
        return img, label 
if __name__ == "__main__": 
    transform = transforms.Compose( [ transforms.ToTensor(), ] ) 
    dataset = catdogDataset(path='./cat_and_dog', train=True, transform=transform) 
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=False) 
    for epoch in range(2): 
        print(f"epoch : {epoch} ") 
        for batch in dataloader: 
            img, label = batch 
            print(img.size(), label)
