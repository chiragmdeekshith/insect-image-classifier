import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, models

# Load the model architecture (MobileNetV2) and modify the classifier
model = models.mobilenet_v2(pretrained=False)  # Initialize the model architecture without pre-trained weights

# Modify the final classifier layer to match the number of output classes (2 for binary classification)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

# Load the trained model weights (state_dict) into the model
model_path = '4_mobilenet_v2_model.pth'  # Path to the trained model file
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load the weights

model.eval()  # Set the model to evaluation mode

# Define the transformations for the input data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ImageNet
])

# Load the test dataset
dataset_path = '../tiny_datasets/BeeOrWasp/training_data'  # Path to the dataset
test_dataset = datasets.ImageFolder(dataset_path, transform=transform)

# DataLoader for iterating through the dataset
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Function to evaluate the model on the test data
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get the predicted labels
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')


# Evaluate the model
evaluate_model(model, test_loader)
