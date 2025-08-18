# Auto-extracted from notebook: dataset & transforms
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import os

class MyDataset(Dataset):

    def __init__(self, images_path ,transform_img=None ,transform_label=None):

        self.images_path = images_path
        self.transform_img = transform_img
        self.transform_label = transform_label

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):

        img = plt.imread(self.images_path[idx])
        image,label = img[:,:int(img.shape[1]/2)],img[:,int(img.shape[1]/2):]

        if self.transform_img:
            image = self.transform_img(image)

        if self.transform_label:
            label = self.transform_label(label)

        return image, label

# Creating Dataset Class for Cityscapes (Pix2Pix)

class CityscapesDataset(Dataset):

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        self.transform = transform
        self.files = sorted(os.listdir(self.data_dir))
        self.image_paths = []
        self.label_paths = []


    def __len__(self):

        return len(self.files)


    def __getitem__(self, idx):

        image_path = os.path.join(self.data_dir, self.files[idx])
        image = Image.open(image_path).convert("RGB")

        width, height = image.size
        real_image = image.crop((0, 0, width // 2, height))
        segmented_image = image.crop((width //2, 0, width, height))

        if self.transform:
            real_image = self.transform(real_image)
            segmented_image = self.transform(segmented_image)

        return segmented_image, real_image

# Loading Cityscapes Pix2Pix Dataset

SIZE = 256
batch_size = 32
root_dir = "/content/cityscapes_data"
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'val')

data_transforms = transform.Compose([
    transform.Resize((SIZE, SIZE)),
    transform.ToTensor(),
    transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),                     # normalizing range (-1, 1)
])

train_dataset = CityscapesDataset(train_dir, transform=data_transforms)
val_dataset = CityscapesDataset(val_dir, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)

# Creating Dataset Class for Cityscapes

class CityscapesDataset(Dataset):

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        self.transform = transform
        self.files = sorted(os.listdir(self.data_dir))
        self.image_paths = []
        self.label_paths = []


    def __len__(self):

        return len(self.files)


    def __getitem__(self, idx):

        image_path = os.path.join(self.data_dir, self.files[idx])
        image = Image.open(image_path).convert("RGB")

        width, height = image.size
        real_image = image.crop((0, 0, width // 2, height))
        segmented_image = image.crop((width //2, 0, width, height))

        if self.transform:
            real_image = self.transform(real_image)
            segmented_image = self.transform(segmented_image)

        return segmented_image, real_image

# Loading Cityscapes Dataset

SIZE = 256
batch_size = 32
root_dir = "/content/cityscapes_data/cityscapes_data"
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'val')

data_transforms = transform.Compose([
    transform.Resize((SIZE, SIZE)),
    transform.ToTensor(),
    transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),                     # normalizing range (-1, 1)
])

train_dataset = CityscapesDataset(train_dir, transform=data_transforms)
val_dataset = CityscapesDataset(val_dir, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)

