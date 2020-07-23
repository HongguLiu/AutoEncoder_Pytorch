import torch
from torch import nn


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding = dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def Maxpooling(kernel_size=3,stride=2):
    return nn.MaxPool2d(kernel_size, stride)
def Upsample(scale=2):
    return nn.PixelShuffle(scale)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.stride = stride

    def forward(self, x):
        indentity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out+=indentity
        #out = self.pooling(x)
        out = self.relu(out)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.conv1 = BasicBlock(64, 64)
        self.conv2 = BasicBlock(128, 128)
        self.conv3 = BasicBlock(512 ,512)
        self.linear = nn.Linear(16*16*1024, 2048)
        self.linear2 = nn.Linear(2048, 16*16*1024)

    
    def forward(self, x):
        x = conv3x3(3, 64)(x)
        x = Maxpooling(x) #64,128
        x = self.conv1(x) #64,128
        x = conv1x1(64, 128) #128, 128
        x = Maxpooling(x) #128, 64
        x = self.conv2(x) #128, 64
        x = conv1x1(128, 512) #512, 64
        x = Maxpooling(x) #512, 32
        x = self.conv3(x) #512, 32
        x = conv1x1(512, 1024) #1024, 32
        x = Maxpooling(x) #1024, 16
        x = x.view(x.size(0), -1)

        encoded = self.linear(x)
        
        x = self.linear2(encoded)
        x = x.view(-1, 1024, 16, 16)
        x = Upsample(x)
        x = conv1x1(1024, 512) #512, 32
        x = self.conv3(x) #512, 32
        x = Upsample(x) #512, 64
        x = conv1x1(512, 128) #128, 64
        x = self.conv2(x) #128, 64
        x = Upsample(x) #128, 128
        x = conv1x1(128, 64) #64, 128
        x = self.conv1(x) #64, 128
        x = Upsample(x) #64, 256
        decoded = conv3x3(64, 3)(x)




        return encoded, decoded



class AutoEncoder_mnist(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoEncoder_mnist, self).__init__()
        self.encoder1 = torch.nn.Linear(input_dim, 14*14)
        self.encoder2 = torch.nn.Linear(14*14, 32)
        self.encoder3 = torch.nn.Linear(32, hidden_dim)
        self.relu = torch.nn.ReLU()

        self.decoder1 = torch.nn.Linear(hidden_dim, 32)
        self.decoder2 = torch.nn.Linear(32, 14*14)
        self.decoder3 = torch.nn.Linear(14*14, input_dim)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.encoder1(x)
        #x = self.relu(x)
        x = self.encoder2(x)
        #x = self.relu(x)
        x = self.encoder3(x)
        encoded = self.relu(x)
        x = self.decoder1(encoded)
        x = self.decoder2(x)
        x = self.decoder3(x)
        decoded = self.sigmoid(x)
        return encoded, decoded