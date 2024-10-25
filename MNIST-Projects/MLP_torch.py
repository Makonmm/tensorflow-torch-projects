import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

LEARNING_RATE = 0.001
BATCH_SIZE = 100
DISPLAY_STEP = 1
TRAINING_EPOCHS = 35
NUM_CLASSES = 10
INPUT_SIZE = 784
HIDDEN_SIZE1 = 256
HIDDEN_SIZE2 = 256


# Normalizing the data and loading the MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE, shuffle=False)

# Define the MLP model


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE1)
        self.fc2 = nn.Linear(HIDDEN_SIZE1, HIDDEN_SIZE2)
        self.fc_out = nn.Linear(HIDDEN_SIZE2, NUM_CLASSES)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc_out(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model instance

model = MLP().to(device)

# Loss function

criterion = nn.CrossEntropyLoss()

# Optimizer

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Lists to store epochs and costs

epoch_set = []
avg_set = []

# Training the model

for epoch in range(TRAINING_EPOCHS):
    avg_cost = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        cost = criterion(outputs, labels)
        cost.backward()
        optimizer.step()

        avg_cost += cost.item() / len(train_loader)

    if (epoch + 1) % DISPLAY_STEP == 0:
        print(f"EPOCH: {epoch + 1:04d}, COST: {avg_cost:.9f}")
    avg_set.append(avg_cost)
    epoch_set.append(epoch + 1)
print("END")


# Plotting the cost history
plt.plot(epoch_set, avg_set, 'o', label='Training Phase')
plt.ylabel('COST')
plt.xlabel('EPOCH')
plt.title('Training Cost')
plt.legend()
plt.show()


# Evaluating model's accuracy

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"MODEL ACCURACY: {100 * correct / total:.4f}")
