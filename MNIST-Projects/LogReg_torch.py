import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Hyperparameters

LEARNING_RATE = 0.01
BATCH_SIZE = 100
DISPLAY_STEP = 1
TRAINING_EPOCHS = 35
NUM_CLASSES = 10
INPUT_SIZE = 784

# Normalizing the data and loading the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE, shuffle=False)

# Define the model (Logistic Regression)


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.W = nn.Parameter(torch.zeros(INPUT_SIZE, NUM_CLASSES))
        self.b = nn.Parameter(torch.zeros(NUM_CLASSES))

    def forward(self, x):
        return torch.softmax(torch.matmul(x, self.W) + self.b, dim=1)


# Create model instance

model = LogisticRegression()

# Loss function

criterion = nn.CrossEntropyLoss()

# Optimizer

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Lists to store epochs and costs

avg_set = []
epoch_set = []

# Training loop´

for epoch in range(TRAINING_EPOCHS):
    avg_cost = 0
    total_batch = len(train_loader)

    for images, labels in train_loader:
        images = images.view(-1, INPUT_SIZE)

        optimizer.zero_grad()
        outputs = model(images)
        cost = criterion(outputs, labels)
        cost.backward()
        optimizer.step()

        avg_cost += cost.item() / total_batch

    if (epoch + 1) % DISPLAY_STEP == 0:
        print(f"EPOCH: {epoch + 1:04d}, COST: {avg_cost:.9f}")

    avg_set.append(avg_cost)
    epoch_set.append(epoch + 1)

print("END")

# Plotting the cost history
plt.plot(epoch_set, avg_set, 'o', label='Logistic Regression (TRAINING PHASE)')
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
        images = images.view(-1, INPUT_SIZE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"MODEL ACCURACY: {100 * correct / total}")
