# Auto-extracted from notebook: model definitions (UNet/SegNet/FCN/etc.)
import torch
import torch.nn as nn
import torchvision.models as models

class Convblock(nn.Module):

      def __init__(self,input_channel,output_channel,kernal=3,stride=1,padding=1):

        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernal,stride,padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel,output_channel,kernal),
            nn.ReLU(inplace=True),
        )


      def forward(self,x):
        x = self.convblock(x)
        return x

class UNet(nn.Module):

    def __init__(self,input_channel,retain=True):

        super().__init__()

        self.conv1 = Convblock(input_channel,32)
        self.conv2 = Convblock(32,64)
        self.conv3 = Convblock(64,128)
        self.conv4 = Convblock(128,256)
        self.neck = nn.Conv2d(256,512,3,1)
        self.upconv4 = nn.ConvTranspose2d(512,256,3,2,0,1)
        self.dconv4 = Convblock(512,256)
        self.upconv3 = nn.ConvTranspose2d(256,128,3,2,0,1)
        self.dconv3 = Convblock(256,128)
        self.upconv2 = nn.ConvTranspose2d(128,64,3,2,0,1)
        self.dconv2 = Convblock(128,64)
        self.upconv1 = nn.ConvTranspose2d(64,32,3,2,0,1)
        self.dconv1 = Convblock(64,32)
        self.out = nn.Conv2d(32,3,1,1)
        self.retain = retain

    def forward(self,x):

        # Encoder Network

        # Conv down 1
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1,kernel_size=2,stride=2)
        # Conv down 2
        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2,kernel_size=2,stride=2)
        # Conv down 3
        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3,kernel_size=2,stride=2)
        # Conv down 4
        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4,kernel_size=2,stride=2)

        # BottelNeck
        neck = self.neck(pool4)

        # Decoder Network

        # Upconv 1
        upconv4 = self.upconv4(neck)
        croped = self.crop(conv4,upconv4)
        # Making the skip connection 1
        dconv4 = self.dconv4(torch.cat([upconv4,croped],1))
        # Upconv 2
        upconv3 = self.upconv3(dconv4)
        croped = self.crop(conv3,upconv3)
        # Making the skip connection 2
        dconv3 = self.dconv3(torch.cat([upconv3,croped],1))
        # Upconv 3
        upconv2 = self.upconv2(dconv3)
        croped = self.crop(conv2,upconv2)
        # Making the skip connection 3
        dconv2 = self.dconv2(torch.cat([upconv2,croped],1))
        # Upconv 4
        upconv1 = self.upconv1(dconv2)
        croped = self.crop(conv1,upconv1)
        # Making the skip connection 4
        dconv1 = self.dconv1(torch.cat([upconv1,croped],1))
        # Output Layer
        out = self.out(dconv1)

        if self.retain == True:
            out = F.interpolate(out,list(x.shape)[2:])

        return out

    def crop(self,input_tensor,target_tensor):
        # For making the size of the encoder conv layer and the decoder Conv layer same
        _,_,H,W = target_tensor.shape
        return transform.CenterCrop([H,W])(input_tensor)

# initializing the model
model = UNet(3).float().to(device)

# Defining Discriminator Class

class Discriminator(nn.Module):

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        # helper function to construct layers quickly
        def conv_block(in_c, out_c, stride):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # due to concatenated input of segmented+real, in_channels=in_channels*2
        self.model = nn.Sequential(
            nn.Conv2d(in_channels*2, 64, kernel_size=4, stride=2, padding=1),     # C64, no BatchNorm
            nn.LeakyReLU(0.2, inplace=True),

            conv_block(64, 128, stride=2),                                        # C128
            conv_block(128, 256, stride=2),                                       # C256
            conv_block(256, 512, stride=1),                                       # C512 (stride 1 for 70x70 patches)

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),                # Final layer
            nn.Sigmoid()
        )


    def forward(self, x, y):

        concatenated = torch.cat([x, y], dim=1)
        verdict = self.model(concatenated)

        return verdict

# Defining Generator Class (via DownSample and UpSample Classes)

class DownSample(nn.Module):

    def __init__(self, in_channels, out_channels, apply_batchnorm=True):
        super(DownSample, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not(apply_batchnorm))]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.down = nn.Sequential(*layers)


    def forward(self, x):

        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, apply_dropout=False):
        super(UpSample, self).__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))

        self.up = nn.Sequential(*layers)


    def forward(self, x, skip):

        x = self.up(x)
        x = torch.cat([x, skip], dim=1)                                         # skip connection
        return x


class Generator(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()

        # Encoder (DownSampling)
        self.down1 = DownSample(in_channels, 64, apply_batchnorm=False)         # C64
        self.down2 = DownSample(64, 128)                                        # C128
        self.down3 = DownSample(128, 256)                                       # C256
        self.down4 = DownSample(256, 512)                                       # C512
        self.down5 = DownSample(512, 512)                                       # C512
        self.down6 = DownSample(512, 512)                                       # C512
        self.down7 = DownSample(512, 512)                                       # C512
        self.down8 = DownSample(512, 512)                                       # C512

        # Decoder (Upsampling)
        self.up1 = UpSample(512, 512, apply_dropout=True)                       # CD512
        self.up2 = UpSample(1024, 512, apply_dropout=True)                      # CD1024
        self.up3 = UpSample(1024, 512, apply_dropout=True)                      # CD1024
        self.up4 = UpSample(1024, 512)                                          # C1024
        self.up5 = UpSample(1024, 256)                                          # C1024
        self.up6 = UpSample(512, 128)                                           # C512
        self.up7 = UpSample(256, 64)                                            # C256

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )


    def forward(self, x):

        # Encoder forward
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder forward + skip connections (U-Net)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

 #Defining Pix2Pix Model Class and Cost Functions

class Pix2Pix(nn.Module):

    def __init__(self, generator, discriminator):
        super(Pix2Pix, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()


    def generator_loss(self, fake_output, fake_target, real_target, lambda_l1=100):

        adversarial_loss = self.criterion_gan(fake_output, torch.ones_like(fake_output, device=fake_output.device))
        l1_loss = self.criterion_l1(fake_target, real_target)

        total_loss = adversarial_loss + lambda_l1 * l1_loss
        return total_loss


    def discriminator_loss(self, real_output, fake_output):

        real_loss = self.criterion_gan(real_output, torch.ones_like(real_output, device=real_output.device))
        fake_loss = self.criterion_gan(fake_output, torch.zeros_like(fake_output, device=fake_output.device))

        total_loss = (real_loss + fake_loss) * 0.5
        return total_loss

# Instantiating Pix2Pix Model

generator = Generator().to(device)
discriminator = Discriminator().to(device)
pix2pix = Pix2Pix(generator, discriminator).to(device)

print(pix2pix)

# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.block(x)
        pooled, indices = self.pool(x)
        return pooled, indices, x.size()

# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, indices, output_size):
        x = self.unpool(x, indices, output_size=output_size)
        return self.block(x)

# SegNet Model
class SegNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=20):
        super(SegNet, self).__init__()
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 512)
        self.decoder5 = DecoderBlock(512, 512)
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, num_classes)

    def forward(self, x):
        e1, ind1, size1 = self.encoder1(x)
        e2, ind2, size2 = self.encoder2(e1)
        e3, ind3, size3 = self.encoder3(e2)
        e4, ind4, size4 = self.encoder4(e3)
        e5, ind5, size5 = self.encoder5(e4)
        d5 = self.decoder5(e5, ind5, size5)
        d4 = self.decoder4(d5, ind4, size4)
        d3 = self.decoder3(d4, ind3, size3)
        d2 = self.decoder2(d3, ind2, size2)
        d1 = self.decoder1(d2, ind1, size1)
        return d1

import time

# Loss function
criterion = nn.CrossEntropyLoss()
# Define your SegNet model
class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        # Define layers here (e.g., Conv2d, MaxPool2d, etc.)

    def forward(self, x):
        # Define the forward pass
        return x

# Initialize the model and optimizer
segnet = SegNet().to(device)
segnet_optimizer = optim.Adam(segnet.parameters(), lr=1e-3)

# Training function
def train_segnet(segnet_model, train_loader, val_loader, epochs, display_interval):
    segnet_model = segnet_model.to(device)
    start_time = time.time()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        segnet_model.train()
        train_loss = 0.0

        # Training loop
        for images, masks in tqdm(train_loader, desc=f"Training {epoch + 1}/{epochs}"):
            images = images.to(device)
            masks = masks.to(device)

            segnet_optimizer.zero_grad()
            outputs = segnet_model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            segnet_optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        # Validation loop
        segnet_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Validating {epoch + 1}/{epochs}"):
                images = images.to(device)
                masks = masks.to(device)

                outputs = segnet_model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

        # Visualize predictions every interval
        if (epoch + 1) % display_interval == 0 or (epoch + 1) == epochs:
            display_segmentation_results(segnet_model, train_loader, num_samples=3)

    print(f"Training Completed in {time.time() - start_time:.2f} seconds.")
    return train_losses, val_losses


# Visualization function
def display_segmentation_results(model, loader, num_samples=5, title="Segmentation Results"):
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(loader))
        images = images[:num_samples].to(device)
        masks = masks[:num_samples]

        predictions = model(images).argmax(dim=1).cpu()

        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        fig.suptitle(title, fontsize=16)

        for i in range(num_samples):
            axes[i, 0].imshow(images[i].permute(1, 2, 0).cpu().numpy())
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis('off')
            axes[i, 1].imshow(masks[i].numpy())
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            axes[i, 2].imshow(predictions[i].numpy())
            axes[i, 2].set_title("Predicted Mask")
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.show()

# Parameters
num_epochs = 50
learning_rate = 1e-3
display_interval = 10

# Define optimizer and loss function
optimizer = optim.Adam(SegNet.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()  # Assuming segmentation classes are encoded as class indices

# Train the SegNet model
train_losses, val_losses = train_segnet(SegNet, train_loader, val_loader, num_epochs, optimizer, criterion, display_interval)
# Visualize results on validation set
display_segmentation_results(SegNet, val_loader, num_samples=5)

# Define hyperparameters
num_epochs = 50
learning_rate = 1e-3
display_interval = 10

# Optimizer for SegNet
segnet_optimizer = optim.Adam(segnet.parameters(), lr=learning_rate)

# Training Function
def train_segnet(segnet_model, train_loader, val_loader, epochs, optimizer, display_interval):
    segnet_model = segnet_model.to(device)
    criterion = nn.CrossEntropyLoss()  # Loss function for semantic segmentation
    start_time = time.time()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        segnet_model.train()
        train_loss = 0.0

        # Training loop
        for images, masks in tqdm(train_loader, desc=f"Training {epoch + 1}/{epochs}"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = segnet_model(images)

            # Reshape outputs and masks for loss calculation
            outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.size(1))
            masks = masks.view(-1)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        # Validation loop
        segnet_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Validating {epoch + 1}/{epochs}"):
                images = images.to(device)
                masks = masks.to(device)

                outputs = segnet_model(images)

                # Reshape outputs and masks for loss calculation
                outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.size(1))
                masks = masks.view(-1)

                loss = criterion(outputs, masks)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

        # Visualize predictions every interval
        if (epoch + 1) % display_interval == 0 or (epoch + 1) == epochs:
            display_segmentation_results(segnet_model, train_loader, num_samples=3)

    print(f"Training Completed in {time.time() - start_time:.2f} seconds.")
    return train_losses, val_losses

# Visualization Function
def display_segmentation_results(model, loader, num_samples=5, title="Segmentation Results"):
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(loader))
        images = images[:num_samples].to(device)
        masks = masks[:num_samples].to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)  # Convert logits to class indices

        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        fig.suptitle(title, fontsize=16)

        for i in range(num_samples):
            axes[i, 0].imshow(images[i].permute(1, 2, 0).cpu().numpy())
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(masks[i].cpu().numpy(), cmap="gray")
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(predictions[i].cpu().numpy(), cmap="gray")
            axes[i, 2].set_title("Predicted Mask")
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.show()

# Training the model
train_losses, val_losses = train_segnet(segnet, train_loader, val_loader, num_epochs, segnet_optimizer, display_interval)

