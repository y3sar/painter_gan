import torch.nn as nn
import torch.optim as optim
from discriminator import Discriminator
from generator import Generator
import torch
from dataprep import PaintDataset, train_transform, weights_init
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms



image_path = 'images/Sandro_Botticelli/'
painter = Generator()
paint_expert = Discriminator()

painter.apply(weights_init)
paint_expert.apply(weights_init)

image_dataset = PaintDataset(image_path, transform=train_transform)
image_loader = DataLoader(image_dataset, batch_size=16, shuffle=True)



optimizer_D = optim.Adam(paint_expert.parameters(), lr=0.01)
optimizer_G = optim.Adam(painter.parameters(), lr=0.01)
criterion = nn.BCELoss()






def train(epochs, generator, discriminator, gen_optimizer, dis_optimizer, criterion, dataloader):
    epoch_loss = 0
    batch_size = dataloader.batch_size
    for epoch in range(epochs):
        k = 0

        #Training the Discriminator on real data
        for image, label in dataloader:

            """Train on real data"""
            discriminator.zero_grad()
            prediction = discriminator(image).view(-1)
            loss_real = criterion(prediction, label)
            epoch_loss += loss_real.item()
            loss_real.backward()
            

            """Training on fake data"""

            #Generate the random noise and generate fake paintings with generator
            fake_label = torch.full((batch_size,), 0) 
            noise = torch.randn(batch_size, 100, 1, 1)
            fake_painting = generator(noise)

            #forward propogate and calculate the loss on the fake data
            fake_pred = discriminator(fake_painting).view(-1)
            loss_fake = criterion(fake_pred, fake_label)
            loss_fake.backward()

            errD = loss_real + loss_fake
            dis_optimizer.step()
            print("discriminator-loss---", errD.item())

            k += 1
            if k > 3:
                break

        """Train the Generator to maximize the loss"""
        generator.zero_grad()
        real_label = torch.full((batch_size,), 1) #generator will optimize for function in which the discriminator thinks the labels for the fake images are true.

        noise = torch.randn(batch_size, 100, 1, 1)

        fake_painting = generator(noise)
        pred = discriminator(fake_painting.detach()).view(-1)
        print(pred)
        generator_loss = criterion(pred, real_label)
        generator_loss.backward()
        print('generator_loss--', generator_loss.item())
        gen_optimizer.step()
        

        """Save a fake painting for inspection"""

        noise = torch.randn(batch_size, 100, 1, 1)
        fake_painting = generator(noise)
        fake_painting = fake_painting[0].squeeze(0)
        pil_image = transforms.ToPILImage()(fake_painting)
        pil_image.save('fake_paintings/painting_'+str(epoch)+'.jpg') 

        
        
        
            
train(125, painter, paint_expert, optimizer_G, optimizer_D, criterion, image_loader)
         
            




