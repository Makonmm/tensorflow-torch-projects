import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 25
N_INPUT = 32 * 32 * 3
N_STEPS = 1
N_HIDDEN = 512
N_CLASSES = 10

# Normalizing and loading the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalização
])

train_dataset = datasets.CIFAR10(
    root='./data_cifar', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(
    root='./data_cifar', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model architecture


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.lstm1 = nn.LSTM(N_INPUT, N_HIDDEN, batch_first=True)
        self.lstm2 = nn.LSTM(N_HIDDEN, N_HIDDEN, batch_first=True)
        self.fc = nn.Linear(N_HIDDEN, N_CLASSES)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        # Last output
        out = self.fc(out[:, -1, :])
        return out


# Creating model, loss func and optimizer
model = RNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Model training
for epoch in range(EPOCHS):
    model.train()
    for step, (batch_xs, batch_ys) in enumerate(train_loader):
        # Moving data to GPU
        batch_xs, batch_ys = batch_xs.to(device), batch_ys.to(device)

        # Reshape
        batch_xs = batch_xs.view(-1, N_STEPS, N_INPUT)

        optimizer.zero_grad()
        logits = model(batch_xs)  # Passing through network
        loss = criterion(logits, batch_ys)  # Loss calc
        loss.backward()  # Grads calc
        optimizer.step()  # Weights update

        if step % 100 == 0:
            _, predicted = torch.max(logits, 1)
            acc = (predicted == batch_ys).float().mean()
            print(f"EPOCH: {
                  epoch + 1}, STEP: {step}, LOSS: {loss.item():.6f}, ACCURACY: {acc:.5f}")

# Evaluating data test
model.eval()
TEST_LOSS = 0
CORRECT = 0

with torch.no_grad():
    for batch_xs, batch_ys in test_loader:
        # Moving data to GPU
        batch_xs, batch_ys = batch_xs.to(device), batch_ys.to(device)

        # Reshape
        batch_xs = batch_xs.view(-1, N_STEPS, N_INPUT)
        test_logits = model(batch_xs)
        TEST_LOSS += criterion(test_logits, batch_ys).item()
        _, predicted = torch.max(test_logits, 1)
        CORRECT += (predicted == batch_ys).sum().item()

TEST_LOSS /= len(test_loader)
test_accuracy = CORRECT / len(test_dataset)

print(f"TEST LOSS: {TEST_LOSS:.6f}, TEST ACCURACY: {test_accuracy:.5f}")
