import os
import torch
import torch.nn as nn
from model import ResNet, ResidualBlock
from data_preprocessing import preprocess_data


def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def train(number_of_epochs: int = 10, device: str = 'cpu') -> None:
    # Create the model
    model = ResNet(ResidualBlock, num_classes=10).to(device)

    # Create the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # Create the training dataloader
    train_dataset = preprocess_data()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Start Training
    for epoch in range(1, number_of_epochs+1):  # loop over the dataset multiple times
        train_epoch(model, device, train_loader, optimizer, epoch)

        # Update learning rate
        scheduler.step()

    # Save the model
    torch.save(model.state_dict(), os.path.join(os.path.dirname(os.getcwd()), 'models/trained_classifier.pth'))


if __name__ == '__main__':
    train(number_of_epochs=300)
