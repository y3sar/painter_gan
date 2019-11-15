import torch.nn as nn
import torch.optim as optim
from discriminator import Discriminator
from generator import Generator
from dataprep import PaintDataset, train_transform


painter = Generator()
paint_expert = Discriminator()
image_dataset = PaintDataset(painter, transform=train_transform)
optimizer_D = Adam(paint_expert.parameters(), lr=0.01)
optimizer_G = Adam(painter.parameters(), lr=0.01)
criterion = nn.BCELoss()






def train(epochs, generator, discriminator, gen_optimizer, dis_optimizer, criterion, real_dataset):
    epoch_loss = 0
    for epoch in range(epochs):
        for image, label in dataloader:
            prediction = discriminator(image)
            loss = criterion(prediction, label)
            epoch_loss += loss.item()
            loss.backward(
            




