import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage




class Generator(nn.Module):
    def __init__(self):
        super().__init__()


        self.conv_block = nn.Sequential(nn.ConvTranspose2d(1, 33, 3, 2),
                                    nn.ReLU(True)
                                    nn.ConvTranspose2d(33, 130, 3, 2),

                                    nn.ConvTranspose2d(130, 33, 3, 2),
                                    nn.Conv2d(33, 3, 5, 3),
                                    )

    def forward(self, x):
        x = self.main(x)
        


        return x




img = torch.randn(1, 1, 64, 64)

gen = Generator()

print(gen(img).shape)

