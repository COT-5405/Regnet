import torch
import argparse
from models import RegNet


# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate a RegNet model on the CIFAR-10 or CIFAR-100 dataset")
parser.add_argument("dataset", choices=["cifar10", "cifar100"], help="Dataset to use (CIFAR-10 or CIFAR-100)")
parser.add_argument("model_path", help="Path to saved model")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size for evaluation (default: 100)")
args = parser.parse_args()


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load test set
if args.dataset == "cifar10":
    num_classes = 10
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
elif args.dataset == "cifar100":
    num_classes = 100
    testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transforms.ToTensor())

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)


# Load saved model
model = RegNet(num_classes=num_classes)
model.load_state_dict(torch.load(args.model_path))
model.to(device)


# Evaluate model on test set
model.eval()
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
