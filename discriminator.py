import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()


        self.main = nn.Sequential(
                                    nn.Conv2d(3, 33, 3, 2),
                                    nn.LeakyReLU(True),
                                    nn.BatchNormalization(33),

                                    nn.Conv2d(33, 130, 3, 2),
                                    nn.LeakyReLU(True),
                                    nn.BatchNormalization(130),

                                    nn.Conv2d(130, 130, 3, 2),
                                    nn.LeakyReLU(True),
                                    nn.BatchNormalization(130)
                                    )

        self.linear_block = nn.Sequential(nn.Linear(52000, 1200),
                                          nn.Linear(1200, 450),
                                          nn.Linear(450, 120),
                                          nn.Linear(120, 50),
                                          nn.Linear(50, 2)
                                          )

       

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size()[0], -1)
        x = self.linear_block(x)
        return x


if __name__ == '__main__':

    img = torch.randn(1, 3, 172, 172)
    dis = Discriminator()
    print(dis(img).shape)




