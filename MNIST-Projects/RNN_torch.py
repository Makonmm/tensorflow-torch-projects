import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 5
N_INPUT = 28
N_STEPS = 28
N_HIDDEN = 256
N_CLASSES = 10

# Normalizing the data and loading the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the images
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

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
        out = self.fc(out[:, -1, :])
        return out


# Creating model, loss func and the optimizer
model = RNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Model training
for epoch in range(EPOCHS):
    model.train()
    for step, (batch_xs, batch_ys) in enumerate(train_loader):
        # Move data to device
        batch_xs, batch_ys = batch_xs.to(device), batch_ys.to(device)

        # Entry reshape
        batch_xs = batch_xs.view(-1, N_STEPS, N_INPUT)

        optimizer.zero_grad()
        logits = model(batch_xs)  # Passing through the network
        loss = criterion(logits, batch_ys)  # Loss calculation
        loss.backward()  # Gradients calculation
        optimizer.step()  # Updating weights

        if step % 100 == 0:
            _, predicted = torch.max(logits, 1)
            acc = (predicted == batch_ys).float().mean()
            print(f"EPOCH: {
                  epoch + 1}, STEP: {step}, LOSS: {loss.item():.6f}, ACCURACY: {acc:.5f}")

# Evaluating model (test data)
model.eval()
TEST_LOSS = 0
CORRECT = 0

with torch.no_grad():
    for batch_xs, batch_ys in test_loader:
        # Move data to device
        batch_xs, batch_ys = batch_xs.to(device), batch_ys.to(device)

        # Entry reshape
        batch_xs = batch_xs.view(-1, N_STEPS, N_INPUT)
        test_logits = model(batch_xs)
        TEST_LOSS += criterion(test_logits, batch_ys).item()
        _, predicted = torch.max(test_logits, 1)
        CORRECT += (predicted == batch_ys).sum().item()

TEST_LOSS /= len(test_loader)
test_accuracy = CORRECT / len(test_dataset)

print(f"TEST LOSS: {TEST_LOSS:.6f}, TEST ACCURACY: {test_accuracy:.5f}")
