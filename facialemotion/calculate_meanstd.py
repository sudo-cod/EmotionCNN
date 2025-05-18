import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os


current_dir = os.getcwd()

dataset = ImageFolder(
    root=f'{current_dir}/dataset',
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
)

loader = DataLoader(dataset, batch_size=32, shuffle=False)

# calculate mean/std
mean = torch.zeros(3)
std = torch.zeros(3)
for images, _ in loader:
    for c in range(3):  # Iterate over RGB channels
        mean[c] += images[:, c, :, :].mean()
        std[c] += images[:, c, :, :].std()
mean /= len(loader)
std /= len(loader)

print("Mean (RGB):", mean.tolist())
print("Std (RGB):", std.tolist())

'''
Mean (RGB): [0.6788272857666016, 0.5195830464363098, 0.4711763262748718]
Std (RGB): [0.1942940503358841, 0.19551515579223633, 0.19221417605876923]
'''