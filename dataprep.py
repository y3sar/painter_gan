import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
import os
import random
import torch.nn as nn

class PaintDataset(Dataset):
    def __init__(self, root, transform):
        super().__init__()
        self.root = root
        self.img_names = os.listdir(self.root)
        self.transform = transform


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        label_tensor = torch.tensor(1.)
        img = Image.open(self.root+self.img_names[idx])
        img = self.transform(img)
        return (img, label_tensor)





train_transform = transforms.Compose([transforms.Resize((64, 64)), 
                                        transforms.ToTensor(),
                                        ])
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



if __name__ == '__main__':

    image_path = 'images/Sandro_Botticelli/'

    dataset = PaintDataset(image_path, train_transform)




        





