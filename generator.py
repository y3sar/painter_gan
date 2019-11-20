import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage




class Generator(nn.Module):
    def __init__(self):
        super().__init__()


        self.conv_block = nn.Sequential(

                                    nn.ConvTranspose2d(100, 512, 4, 1, 0),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),

                                    nn.ConvTranspose2d(512, 256, 4, 2, 1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True),

                                    nn.ConvTranspose2d(256, 128, 4, 2, 1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(True),

                                    nn.ConvTranspose2d(128, 64, 4, 2, 1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True),

                                    nn.ConvTranspose2d(64, 3, 4, 2, 1),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU(True),

                                    nn.ConvTranspose2d(3, 3, 4, 2, 1),
                                    nn.Tanh(),

                                    

                                    )

    def forward(self, x):
        x = self.conv_block(x)
        


        return x


if __name__ == '__main__':

    img = torch.randn(1, 100, 1, 1)

    gen = Generator()

    print(gen(img).shape)

