import torch.nn as nn
import torch.optim as optim
from discriminator import Discriminator
from generator import Generator
import torch
from dataprep import PaintDataset, train_transform, weights_init
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

image_path = 'images/Sandro_Botticelli/'
painter = Generator().to(device)
paint_expert = Discriminator().to(device)

fixed_noise = torch.randn(1, 100, 1, 1, device=device)

painter.apply(weights_init)
paint_expert.apply(weights_init)

image_dataset = PaintDataset(image_path, transform=train_transform)

image_loader = DataLoader(image_dataset, batch_size=128, shuffle=True)





optimizer_D = optim.Adam(paint_expert.parameters(), lr=0.002, betas=(0.5, 0.999))
optimizer_G = optim.Adam(painter.parameters(), lr=0.002, betas=(0.5, 0.999))
criterion = nn.BCELoss()






def train(epochs, generator, discriminator, gen_optimizer, dis_optimizer, criterion, dataloader):
    batch_size = dataloader.batch_size
    for epoch in range(epochs):
        k = 0
        print( "*************Epoch " + str(epoch) +"*****************")

        #Training the Discriminator on real data
        for image, label in dataloader:

            """Train on real data"""
            discriminator.zero_grad()
            image = image.cuda()
            real_label = torch.full((batch_size,), 1., device=device)

            real_pred = discriminator(image.cuda()).view(-1)
            loss_real = criterion(real_pred, real_label)
            loss_real.backward()
            

            """Training on fake data"""

            #Generate the random noise and generate fake paintings with generator
            fake_label = torch.full((batch_size,), 0., device=device) 
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_painting = generator(noise)

            #forward propogate and calculate the loss on the fake data
            fake_pred = discriminator(fake_painting.detach()).view(-1)
            loss_fake = criterion(fake_pred, fake_label)
            loss_fake.backward()

            errD = loss_real + loss_fake
            dis_optimizer.step()
            print("discriminator-loss---", errD.item())
            break


        """Train the Generator to maximize the loss"""
        generator.zero_grad()
        real_label = torch.full((batch_size,), 1., device=device) #generator will optimize for function in which the discriminator thinks the labels for the fake images are true.

        #noise = torch.randn(batch_size, 100, 1, 1)

        #fake_painting = generator(noise)
        pred = discriminator(fake_painting).view(-1)
        generator_loss = criterion(pred, real_label)
        generator_loss.backward()
        print()
        print('generator_loss--', generator_loss.item())
        gen_optimizer.step()
        

        """Save the fake painting for inspection"""
        fake_painting = generator(fixed_noise).detach()

        fake_painting = fake_painting[0].squeeze(0)
        pil_image = transforms.ToPILImage()(fake_painting.cpu())
        pil_image.save('fake_paintings/painting_'+str(epoch)+'.jpg') 

        
train(4255, painter, paint_expert, optimizer_G, optimizer_D, criterion, image_loader)
        
        
            
         
            




