import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class CNN(nn.Module):
    def __init__(self, device, generator):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, device=device)
        self.bn1 = nn.BatchNorm2d(32, device=device)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, device=device)
        self.bn2 = nn.BatchNorm2d(32, device=device)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 6 * 6, 256, device=device)
        self.fc2 = nn.Linear(256, 2, device=device)
        self.init_layers(generator)

    def init_layers(self, generator):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu', generator=generator)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu', generator=generator)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu', generator=generator)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='linear', generator=generator)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def train_one_epoch(model, optimizer, train_loader, criterion, device) -> float:
    """
    Train the model for one epoch
    :param model: model to train
    :param optimizer: torch.optim.Adagrad(model.parameters(), lr = 0.015)
    :param train_loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.CrossEntropyLoss()
    :param device: device to use
    :return: loss
    """
    model.train()

    num_batches = len(train_loader)
    pbar = tqdm(total=num_batches, dynamic_ncols=True)

    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        pbar.update(1)
        pbar.set_description(f'Batch {i + 1:4d}/{num_batches:4d}, '
                             f'Batch size: {inputs.size(0):4d}, '
                             f'Loss: {loss.item():.5f}, ')

    pbar.close()

    return running_loss / len(train_loader)
