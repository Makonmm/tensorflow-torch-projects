import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Parameters
LEARNING_RATE = 0.001  # Learning rate for the optimizer
BATCH_SIZE = 256        # Number of samples per batch
DISPLAY_STEP = 10       # Interval for displaying training statistics
NUM_CLASSES = 10        # Number of output classes (CIFAR-10 classes)
NUM_EPOCHS = 25         # Number of epochs for training

# Normalizing the data and loading the CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),  # Convert images to PyTorch tensors
     # Normalize the images to have mean and std for RGB channels
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the CIFAR-10 dataset
train_dataset = datasets.CIFAR10(
    root='./data_cifar', train=True, download=True, transform=transform)  # Training dataset
test_dataset = datasets.CIFAR10(
    root='./data_cifar', train=False, download=True, transform=transform)  # Testing dataset

# Create data loaders for training and testing
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True)  # Shuffle the training data
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE, shuffle=False)  # No shuffle for test data

# Define the CNN architecture


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define layers
        # First convolutional layer
        # Input channels = 3 for RGB images
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        # First fully connected layer
        # Adjusted input size after pooling
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, NUM_CLASSES)  # Output layer
        self.pool = nn.MaxPool2d(2)  # Max pooling layer
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        # Define the forward pass
        # Apply conv1, ReLU, and max pooling
        x = self.pool(self.relu(self.conv1(x)))
        # Apply conv2, ReLU, and max pooling
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 5 * 5)  # Adjusted for CIFAR-10
        # Apply first fully connected layer and ReLU
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Apply output layer
        return x  # Return output logits


# Create the model instance
model = CNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function (cross-entropy loss for classification tasks)
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    for step, (images, labels) in enumerate(train_loader):
        # Move images and labels to the GPU (if available)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(images)  # Forward pass: compute predicted outputs
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update model parameters

        # Display training statistics
        if (step + 1) % DISPLAY_STEP == 0:
            # Get predicted class indices
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == labels).float().mean()  # Compute accuracy
            print(f"Epoch [{epoch + 1} / {NUM_EPOCHS}], STEP [{step +
                  1}], LOSS: {loss.item():.4f}, ACCURACY: {accuracy:.4f}")

print("END")

# Evaluation mode
model.eval()  # Set the model to evaluation mode

# Testing loop
with torch.no_grad():  # Disable gradient calculation for evaluation
    correct = 0  # Initialize correct predictions counter
    total = 0    # Initialize total samples counter
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # Forward pass for test data
        # Get predicted class indices
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)  # Update total samples
        # Count correct predictions
        correct += (predicted == labels).sum().item()

    # Calculate and print the test accuracy
    print(f"TEST ACCURACY: {100 * correct / total:.2f}")
