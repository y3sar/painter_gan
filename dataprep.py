import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random

class PaintDataset(Dataset):
    def __init__(self, generator, transform):
        super().__init__()
        self.generator = generator
        self.img_names = os.listdir('images/Sandro_Botticelli')
        self.labels = [0,1]

    def __getitem__(self, idx):
        label = random.choice(self.labels)
        label_tensor = torch.tensor([0, 0])
        if label:
            img = Image.open(self.img_names[idx])
            img = transforms(img)
            label_tensor = label_tensor[label].add_(1)
            return (img, label_tensor)

        img = self.generator(torch.randn(1, 3, 64, 64))
        label_tensor = label_tensor[label].add_(1)
        return (img, label_tensor)




train_transform = transforms.Compose([transforms.Resize((172, 172)), 
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                        transforms.ToTensor()
                                        ])









        





