import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from regnet import RegNetX, RegNetY

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="cifar10", help="Name of the dataset")
parser.add_argument("--regnet_type", default="regnetx_200mf", help="Type of RegNet model to use")
parser.add_argument("--batch_size", type=int, default=128, help="Size of the training batch")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for optimizer")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizer")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
parser.add_argument("--workers", type=int, default=4, help="Number of worker threads for DataLoader")
args = parser.parse_args()

# Define the transforms for the dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the dataset
if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    num_classes = 10
elif args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    num_classes = 100
else:
    raise ValueError("Invalid dataset name")

# Define the data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

# Instantiate the RegNet model
if args.regnet_type.startswith("regnetx"):
    model = RegNetX(args.regnet_type)
elif args.regnet_type.startswith("regnety"):
    model = RegNetY(args.regnet_type)
else:
    raise ValueError("Invalid RegNet type")

# Move the model to the device
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# Train the model
for epoch in range(args.epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
                # Move inputs and labels to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # Print every 100 mini-batches
            print("[Epoch %d, Batch %5d] Loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    # Evaluate the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("Accuracy on test set: %.2f%%" % accuracy)

print("Training complete")

