import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()


        self.main = nn.Sequential(
                                    nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, True),
                                    nn.BatchNorm2d(64),

                                    nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, True),
                                    nn.BatchNorm2d(128),

                                    nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, True),
                                    nn.BatchNorm2d(256),
                                    
                                    nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, True),
                                    nn.BatchNorm2d(512),

                                    nn.Conv2d(512, 1, 4, 1, bias=False),
                                    nn.Sigmoid()



                                    )


       

    def forward(self, x):
        x = self.main(x)
        return x


if __name__ == '__main__':

    img = torch.randn(1, 3, 64, 64)
    dis = Discriminator()
    print(dis(img))




