import torch
import torchvision
from torchvision import transforms
from networks.autoencoders import AutoEncoder
import pdb
from datasets import MyDataset

input_dim = 28*28
hidden_dim = 64
num_epoches = 100
batch_size = 128

transformer = transforms.Compose([
    transforms.resize((256, 256)),
    transforms.ToTensor(),
])

train_data = MyDataset(txt_path="../Non-local_pytorch/mnist", transform=transformer)
val_data = MyDataset(txt_path="../Non-local_pytorch/mnist", train=False, transform=transformer)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

net = AutoEncoder(input_dim, hidden_dim)
net.cuda()
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(num_epoches):
    for i, (images) in enumerate(train_loader):
        #images = images.view(-1, 320*320)
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
    for i, (images) in enumerate(val_loader):
        #images = images.view(-1, 320*320)
        images = images.cuda()
        encoded, decoded = net(images)
        loss = criterion(images, decoded)
        val_loss += loss.item()
    print('Epoch [%d/%d], Val Loss:%.4f'%(epoch+1, num_epoches, val_loss/batch_size))