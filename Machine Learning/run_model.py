import torch
from torch import nn
import cv2
import numpy as np
from pathlib import Path

# Define the model architecture
class CataractModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super(CataractModel, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(hidden_units)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, 2 * hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(2 * hidden_units)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(2 * hidden_units, 4 * hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4 * hidden_units, output_shape, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(output_shape)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28 * output_shape, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # Output 2 classes (Cataract, Normal)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.classifier(x)
        return x

# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CataractModel(3, 10, 2).to(device)

# Load the model's state_dict
model_path = 'C:/Users/PRANAY/Downloads/devjams/EyeCataractDetectModel.pth'
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # Change to (C, H, W)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Function to make predictions
def predict(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        pred_logits = model(image)
        pred_probs = torch.softmax(pred_logits, dim=1)
        pred_classes = pred_probs.argmax(dim=1)
    return pred_classes.cpu().numpy()[0], pred_probs.cpu().numpy()

# Example usage
if __name__ == "__main__":
    image_path = 'C:/Users/PRANAY/Downloads/devjams/detect_example.jpg'  # Corrected path
    predicted_class, probabilities = predict(image_path)
    
    class_names = ['Normal', 'Cataract']
    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Probabilities: {probabilities}")
