import torch
import torchvision
from networks.autoencoders import AutoEncoder_mnist
import pdb

input_dim = 28*28
hidden_dim = 3
num_epoches = 100
batch_size = 128

train_data = torchvision.datasets.MNIST(root="../Non-local_pytorch/mnist", train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root="../Non-local_pytorch/mnist", train=False, transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

net = AutoEncoder_mnist(input_dim, hidden_dim)
net.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(num_epoches):
    for i, (images, _) in enumerate(train_loader):
        images = images.view(-1, 28*28)
        images = images.cuda()
        #pdb.set_trace()
        encoded, decoded = net(images)
        loss = criterion(images, decoded)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1) % 100 ==0:
            print('Epoch [%d/%d], Iter[%d/%d] Loss:%.4f'%(epoch+1, num_epoches, i+1, len(train_data)//batch_size, loss.item()))

    val_loss = 0
    for i, (images, _) in enumerate(test_loader):
        images = images.view(-1, 28*28)
        images = images.cuda()
        encoded, decoded = net(images)
        loss = criterion(images, decoded)
        val_loss += loss.item()
    print('Epoch [%d/%d], Val Loss:%.4f'%(epoch+1, num_epoches, val_loss/batch_size))