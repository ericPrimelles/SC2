'''
Script que contiene los modelos neuronales implementados
'''
# Modelos
import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.Module):
    """
    Modelo neuronal para el minijuego Move to Beacon
    """

    def __init__(self, *args):
        super(DQNModel, self).__init__()
        """
        Consta de 3 redes convolucionales, toman un estado, lo expanden y devuelven una acción
        """
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(24, 1, kernel_size=3, stride=1, padding=1)
        self.name = 'BeaconCNN'

    def forward(self, x):
        # Función Foward
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = self.conv3(x)
        return x


class D3QNModel(nn.Module):
    """
    Modelo Dueling DQN para minijuego Defeat to Roaches
    """

    def __init__(self, n_features, screen_size=64):
        super(D3QNModel, self).__init__()
        # Misma estructura
        self.conv1 = nn.Conv2d(n_features, 12, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1)
        self.flt = nn.Flatten()
        self.fc1 = nn.Linear(screen_size ** 2, screen_size ** 2)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(screen_size ** 2, screen_size ** 2)

        # Añadiendo las redes de estimación de valor y ventaja
        self.v1 = nn.Linear(screen_size ** 2, screen_size ** 2)
        self.a1 = nn.Linear(screen_size ** 2, screen_size ** 2)
        self.v2 = nn.Linear(screen_size ** 2, 1)
        self.a2 = nn.Linear(screen_size ** 2, screen_size ** 2)
        self.name = f'D3QNModel{n_features}'
        self.screen_size = screen_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.flt(x))
        #x = x.view(-1, self.screen_size ** 2)
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        a = F.relu(self.a1(features))
        a = self.a2(a)
        v = F.relu(self.v1(features))
        v = self.v2(v)

        x = v + a - a.mean(1).unsqueeze(1).expand(x.size(0), self.screen_size ** 2)  # Forma Dueling Deep Q-Learning
        return x
