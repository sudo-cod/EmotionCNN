import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import os


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
current_dir = os.getcwd()

# Hyper parameters
num_epochs = 50
num_classes = 4
batch_size = 100
learning_rate = 0.001

train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.678, 0.520, 0.471], std=[0.194, 0.196, 0.192])
])

val_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.678, 0.520, 0.471], std=[0.194, 0.196, 0.192])
])

dataset = ImageFolder(root=f'{current_dir}/dataset')

labels = np.array([label for _, label in dataset])          # dataset labels
indices = np.arange(len(labels))                            # 0 … len-1
train_indices, val_indices = train_test_split(indices, test_size=0.1,
                                              stratify=labels, random_state=42)

from torch.utils.data import Subset
train_dataset = Subset(dataset, train_indices)
train_dataset.dataset.transform = train_transform
val_dataset  = Subset(dataset, val_indices)
val_dataset.dataset.transform = val_transform

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader   = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)


# Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # input size to fc: 128×128 → /2 /2 /2 = 16, channels 64
        self.fc = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)         # propagate through new layer
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy of the model on the validation set: {100 * correct / total:.2f} %')

# Save the model checkpoint
torch.save(model.state_dict(), fr'{current_dir}/model.ckpt')
