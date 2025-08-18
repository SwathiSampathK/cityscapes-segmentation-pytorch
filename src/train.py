# Auto-extracted from notebook: training & evaluation loop
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

# Creating the DataLoaders
batch_size = 4
train_loader = DataLoader(traindata,batch_size)
vaild_loader = DataLoader(valdata,1)

lr = 0.01
epochs = 30

# Choosing the loss function to be Mean Square Error Loss
lossfunc = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_acc = []
val_acc = []
train_loss = []
val_loss = []

for i in range(epochs):

    trainloss = 0
    valloss = 0

    for img,label in tqdm(train_loader):
        '''
            Traning the Model.
        '''
        optimizer.zero_grad()
        img = img.to(device)
        label = label.to(device)
        output = model(img)
        loss = lossfunc(output,label)
        loss.backward()
        optimizer.step()
        trainloss+=loss.item()

    if(i%5==0):
        show(img,output,label)

    train_loss.append(trainloss/len(train_loader))

    for img,label in tqdm(vaild_loader):
        '''
            Validation of Model.
        '''
        img = img.to(device)
        label = label.to(device)
        output = model(img)
        loss = lossfunc(output,label)
        valloss+=loss.item()

    val_loss.append(valloss/len(vaild_loader))

    print("epoch : {} ,train loss : {} ,valid loss : {} ".format(i,train_loss[-1],val_loss[-1]))

# Defining Training Tools

def train_pix2pix(pix2pix_model, train_loader, val_loader, epochs, lambda_l1, display_interval):

    start_time = time.time()
    train_g_losses, train_d_losses = [], []
    val_g_losses, val_d_losses = [], []

    for epoch in range(epochs):

        # training
        pix2pix_model.train()
        train_g_loss, train_d_loss = 0.0, 0.0

        for idx, (segmented_images, real_images) in enumerate(tqdm(train_loader, desc=f"Training {epoch+1}/{epochs}")):
            segmented_images = segmented_images.to(device)
            real_images = real_images.to(device)

            # train model discriminator
            d_optimizer.zero_grad()
            fake_images = pix2pix_model.generator(segmented_images)
            real_output = pix2pix_model.discriminator(segmented_images, real_images)
            fake_output = pix2pix_model.discriminator(segmented_images, fake_images)
            d_loss = pix2pix_model.discriminator_loss(real_output, fake_output)
            d_loss.backward()
            d_optimizer.step()

            # train model generator
            g_optimizer.zero_grad()
            fake_images = pix2pix_model.generator(segmented_images)
            fake_output = pix2pix_model.discriminator(segmented_images, fake_images)
            g_loss = pix2pix_model.generator_loss(fake_output, fake_images, real_images, lambda_l1)
            g_loss.backward()
            g_optimizer.step()

            train_d_loss += d_loss.item()
            train_g_loss += g_loss.item()

        # average train loss per epoch
        train_d_losses.append(train_d_loss / len(train_loader))
        train_g_losses.append(train_g_loss / len(train_loader))

        # validating
        pix2pix_model.eval()
        val_g_loss, val_d_loss = 0.0, 0.0

        with torch.no_grad():
            for idx, (segmented_images, real_images) in enumerate(tqdm(val_loader, desc=f"Validating {epoch+1}/{epochs}")):
                segmented_images = segmented_images.to(device)
                real_images = real_images.to(device)

                fake_images = pix2pix_model.generator(segmented_images)
                real_output = pix2pix_model.discriminator(segmented_images, real_images)
                fake_output = pix2pix_model.discriminator(segmented_images, fake_images)

                d_loss = pix2pix_model.discriminator_loss(real_output, fake_output)
                g_loss = pix2pix_model.generator_loss(fake_output, fake_images, real_images, lambda_l1)

                val_d_loss += d_loss.item()
                val_g_loss += g_loss.item()

        # average val loss per epoch
        val_d_losses.append(val_d_loss / len(val_loader))
        val_g_losses.append(val_g_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{epochs} | G Train Loss: {train_g_losses[-1]:.4f} | D Train Loss: {train_d_losses[-1]:.4f} | G Val Loss: {val_g_losses[-1]:.4f} | D Val Loss: {val_d_losses[-1]:.4f}")

        # display sample results after each interval
        if (epoch + 1) % display_interval == 0 or (epoch + 1) == epochs:
            display_generated_samples(pix2pix_model, train_loader, num_samples=3)

    print(f"Training Completed in {time.time() - start_time:.2f} seconds.")
    return train_g_losses, train_d_losses, val_g_losses, val_d_losses


def display_generated_samples(model, loader, num_samples=10, title="Generated Samples"):

    model.eval()
    with torch.no_grad():
        segmented_images, real_images = next(iter(loader))
        segmented_images = segmented_images[:num_samples].to(device)
        real_images = real_images[:num_samples].to(device)

        fake_images = model.generator(segmented_images)

        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        fig.suptitle(title, fontsize=16)

        for i in range(num_samples):
            axes[i, 0].imshow((segmented_images[i].permute(1, 2, 0).cpu().numpy() + 1) / 2)         # denormalizing
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis('off')
            axes[i, 1].imshow((real_images[i].permute(1, 2, 0).cpu().numpy() + 1) / 2)              # denormalizing
            axes[i, 1].set_title("Target Image")
            axes[i, 1].axis('off')
            axes[i, 2].imshow((fake_images[i].permute(1, 2, 0).cpu().numpy() + 1) / 2)              # denormalizing
            axes[i, 2].set_title("Generated Image")
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.show()

# Training Pix2Pix Model

num_epochs = 50
learning_rate = 2e-4
lambda_l1 = 100
display_interval = 10

g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

train_g_losses, train_d_losses, val_g_losses, val_d_losses = train_pix2pix(pix2pix, train_loader, val_loader, num_epochs, lambda_l1, display_interval)

