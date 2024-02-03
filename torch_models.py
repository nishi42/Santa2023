"""
This file contains the PyTorch models for the Rubik's Cube dataset.
And also the dataset class for the Rubik's Cube dataset.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd

class RubikCubeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = []
        self.transform = transform

        df = pd.read_csv(csv_file, header=0)
        
        for index, row in df.iterrows():
            # Parse the tuple string to actual tuple
            state = ast.literal_eval(row[0])
            state_table = preprocess_state(state)
            state_tensor = torch.tensor(state_table)
            label = row[1]
            self.data.append((state_tensor, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        state = self.data[idx][0]
        solution = self.data[idx][1]

        if self.transform:
            state = self.transform(state)
            solution = self.transform(solution)
        
        return state, solution

# Define the model
class RubikCubeNet(nn.Module):
    def __init__(self, inputs_size=33*33):
        super(RubikCubeNet, self).__init__()
        self.fc1 = nn.Linear(inputs_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Multi Input Model
class RubikCubePredictMoves(nn.Module):
    def __init__(self, inputs_size=33*33):
        super(RubikCubePredictMoves, self).__init__()

        self.input_layer = nn.Linear(inputs_size, 1024)
        # Use bi-directional LSTM
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.selu(self.bn1(self.fc1(x)))
        x = F.selu(self.bn2(self.fc2(x)))
        x = F.selu(self.fc3(x))
        x = self.fc4(x)
        return x

class RubikCubePredictMoves1DCNN(nn.Module):
    def __init__(self, input_length=33*33):
        super(RubikCubePredictMoves1DCNN, self).__init__()

        self.input_length = input_length

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        # 計算された特徴量の数に基づいて次元を調整する必要があります
        self.fc1 = nn.Linear(128 * input_length, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        # 入力xを3Dテンソルに変形（例：[batch_size, 1, input_length]）
        x = x.view(-1, 1, self.input_length)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Multi Input Model
class RubikCubeMultiInput(nn.Module):
    def __init__(self, input_size=33*33):
        super(RubikCubeMultiInput, self).__init__()

        self.input_layer2 = nn.Linear(2*2*6, 1024)
        self.input_layer3 = nn.Linear(3*3*6, 1024)
        self.input_layer4 = nn.Linear(4*4*6, 1024)
        self.input_layer5 = nn.Linear(5*5*6, 1024)
        self.input_layer6 = nn.Linear(6*6*6, 1024)
        self.input_layer7 = nn.Linear(7*7*6, 1024)
        self.input_layer8 = nn.Linear(8*8*6, 1024)
        self.input_layer9 = nn.Linear(9*9*6, 1024)
        self.input_layer10 = nn.Linear(10*10*6, 1024)
        self.input_layer19 = nn.Linear(19*19*6, 1024)
        self.input_layer33 = nn.Linear(33*33*6, 1024)

        # Use bi-directional LSTM
        self.fc1 = nn.Linear(1024, 512)
        self.bilstm1 = nn.LSTM(512, 512, batch_first=True, bidirectional=True)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x, inputs_size):
        if inputs_size == 2*2*6:
            x = x.view(-1, 2*2*6)
            x = F.relu(self.input_layer2(x))
        elif inputs_size == 3*3*6:
            x = x.view(-1, 3*3*6)
            x = F.relu(self.input_layer3(x))
        elif inputs_size == 4*4*6:
            x = x.view(-1, 4*4*6)
            x = F.relu(self.input_layer4(x))
        elif inputs_size == 5*5*6:
            x = x.view(-1, 5*5*6)
            x = F.relu(self.input_layer5(x))
        elif inputs_size == 6*6*6:
            x = x.view(-1, 6*6*6)
            x = F.relu(self.input_layer6(x))
        elif inputs_size == 7*7*6:
            x = x.view(-1, 7*7*6)
            x = F.relu(self.input_layer7(x))
        elif inputs_size == 8*8*6:
            x = x.view(-1, 8*8*6)
            x = F.relu(self.input_layer8(x))
        elif inputs_size == 9*9*6:
            x = x.view(-1, 9*9*6)
            x = F.relu(self.input_layer9(x))
        elif inputs_size == 10*10*6:
            x = x.view(-1, 10*10*6)
            x = F.relu(self.input_layer10(x))
        elif inputs_size == 19*19*6:
            x = x.view(-1, 19*19*6)
            x = F.relu(self.input_layer19(x))
        elif inputs_size == 33*33*6:
            x = x.view(-1, 33*33*6)
            x = F.relu(self.input_layer33(x))
        else:
            print('Error')
            return

        x = F.relu(self.fc1(x))
        x, _ = self.bilstm1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    """
    Trains the given model on the training data for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        device (torch.device): The device to be used for training.
        train_loader (torch.utils.data.DataLoader): The data loader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model parameters.
        epoch (int): The current epoch number.

    Returns:
        None
    """
    model.train() # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # Move the data to device
        optimizer.zero_grad() # Clear the gradient
        output = model(data) # Predict the output
        loss = F.mse_loss(output, target) # Calculate the loss
        loss.backward() # Backpropagation
        optimizer.step() # Update the parameters
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
# Define Testing
def test(model, device, test_loader):
    """
    Evaluate the performance of a model on a test dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        device (torch.device): The device to run the evaluation on.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.

    Returns:
        None
    """
    model.eval() # Set the model to evaluation mode
    test_loss = 0
    with torch.no_grad(): # Turn off gradients for validation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) # Move the data to device
            output = model(data) # Predict the output
            test_loss += F.mse_loss(output, target, reduction='sum').item() # Calculate the loss
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}\n')