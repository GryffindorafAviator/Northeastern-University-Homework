import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from PIL import Image
import sys


# Loading data and transform them
def load_data():
    training_set = torchvision.datasets.CIFAR10(
        root='./data.cifar10',
        train=True,
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(
        training_set,
        batch_size=64,
        shuffle=True
    )

    test_set = torchvision.datasets.CIFAR10(
        root='./data.cifar10',
        train=False,
        download=True,
        transform=transform
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size=64,
        shuffle=True
    )

    return train_dataloader, test_dataloader


# Defining the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Calculate the accuracy and training loss
def accuracy(input_data, criterion):
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in input_data:
            images = data[0].to(device)
            labels = data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            running_loss += criterion(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total, running_loss / len(input_data)


# Training function
def train():
    train_dataloader, test_dataloader = load_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    total_epoch = 10

    net.to(device)

    print("LOOP    Train Loss  Train Acc%  Test Loss  Test Acc%")

    for epoch in range(total_epoch):
        running_loss = 0.0

        for i, data in enumerate(train_dataloader, 0):
            net.train()
            inputs = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)
        train_acc, _ = accuracy(train_dataloader, criterion)
        test_acc, test_loss = accuracy(test_dataloader, criterion)

        print('%d/%d    %.2f        %d%%         %.2f       %d%%' % (epoch + 1,
                                                                     total_epoch,
                                                                     train_loss,
                                                                     train_acc,
                                                                     test_loss,
                                                                     test_acc))

        model_path = './model/model.ckpt'
        torch.save(net.state_dict(), model_path)


# Test function
def test():
    with torch.no_grad():
        image_path = sys.argv[2]
        model_path = './model/model.ckpt'

        if not os.path.isfile('./model/model.ckpt'):
            sys.stderr.write("no model saved")
            sys.exit()

        if not os.path.isfile(image_path):
            sys.stderr.write("no image found")
            sys.exit()

        image = Image.open(image_path)
        image = transform(image)
        # fit the CIFAR10 format
        image = image.view(1, 3, 32, 32)
        net.load_state_dict(torch.load(model_path))

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        output = net(image)
        _, predicted = torch.max(output, 1)

        print("Prediction result: %s" % (classes[predicted]))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    net = NeuralNetwork()

    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test":
        test()
