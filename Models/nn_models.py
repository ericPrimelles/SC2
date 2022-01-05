import torch.nn as nn
import torch.nn.functional as F

class BeaconCNN2(nn.Module):
    """
    NN model specifically for beacon mini game
    """
    def __init__(self, *args):
        super(BeaconCNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(24, 1, kernel_size=3, stride=1, padding=1)
        self.name = 'BeaconCNN'

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = self.conv3(x)
        return x

class FeatureCNNFCBig(nn.Module):
    """
    CNN model based on feature inputs
    """
    def __init__(self, n_features, screen_size=64):
        super(FeatureCNNFCBig, self).__init__()

        self.conv1 = nn.Conv2d(n_features, 12, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(screen_size ** 2, screen_size ** 2)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(screen_size ** 2, screen_size ** 2)
        self.name = f'FeatureCNNFCBig{n_features}'
        self.screen_size = screen_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.screen_size ** 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
