import os

import torch
from torchvision import transforms
from PIL import Image
current_dir = os.getcwd()
# Define model architecture (same as training)
class ConvNet(torch.nn.Module):
    def __init__(self, num_classes=4):
        super(ConvNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = torch.nn.Linear(32 * 32 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Load the trained model
model = ConvNet(num_classes=4)
model.load_state_dict(torch.load(f'{current_dir}/model.ckpt'))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.678, 0.520, 0.471], std=[0.194, 0.196, 0.192])
])


image = Image.open(f"{current_dir}/WechatIMG839.jpg").convert('RGB')
image = transform(image).unsqueeze(0)  # Add batch dimension


# Predict
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
    predicted_class = predicted.item()

class_names = ['none', 'pouting', 'smile', 'openmouth']  # Adjust to your classes
print(f"Predicted emotion: {class_names[predicted_class]}")