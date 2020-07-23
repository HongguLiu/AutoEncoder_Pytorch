import torch
from torch import nn

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, 4*out_channels, kernel_size=3, stride=1, padding=1)

def conv5x5(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=1)

def batch_norm(in_channels):
    return nn.BatchNorm2d(in_channels)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
    def forward(self, x):
        return x.view(-1, 1024, 8, 8)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = conv5x5(3, 64)
        self.bn1 = batch_norm(64)
        self.conv2 = conv5x5(64, 128)
        self.bn2 = batch_norm(128)
        self.conv3 = conv5x5(128, 256)
        self.bn3 = batch_norm(256)
        self.conv4 = conv5x5(256, 512)
        self.bn4 = batch_norm(512)
        self.conv5 = conv5x5(512, 1024)
        self.bn5 = batch_norm(1024)
        self.relu = nn.ReLu()
        self.linear1 = nn.Linear(8*8*1024, 1024)
        self.linear2 = nn.Linear(1024, 8*8*1024)
        self.upsample = nn.PixelShuffle(2)
        self.flatten = Flatten()
        self.reshape = Reshape()
        self.deconv1 = conv3x3(1024, 512)
        self.deconv2 = conv3x3(512, 256)
        self.deconv3 = conv3x3(256, 128)
        self.deconv4 = conv3x3(128, 64)
        self.deconv5 = conv3x3(64 ,3)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        x = self.conv1(x) #(64, 128, 128)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x) #(128, 64, 64)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x) #(256 ,32 ,32)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x) #(512 ,16 ,16)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x) #(1024 ,8 ,8)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.flatten(x)
        encoded = self.linear1(x)
        
        x = self.linear2(encoded)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.upsample(x) #(512, 16, 16)
        x = self.deconv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.upsample(x) #(256, 32, 32)
        x = self.deconv3(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.upsample(x) #(128, 64, 64)
        x = self.deconv4(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.upsample(x) #(64, 128, 128)
        x = self.deconv5(x)
        x = self.relu(x)
        x = self.upsample(x) #(3, 256, 256)
        decoded = self.sigmoid(x)

        return encoded, decoded


