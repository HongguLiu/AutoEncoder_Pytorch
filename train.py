import torch
import torchvision
from torchvision import transforms
from networks.autoencoders import AutoEncoder
import pdb
from datasets import MyDataset
import os
from torch.optim import lr_scheduler

num_epoches = 100
batch_size = 256
learning_rate = 2e-4

transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_data = MyDataset(txt_path="./data_list/real_face_train.txt", transform=transformer)
val_data = MyDataset(txt_path="./data_list/real_face_val.txt", transform=transformer)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

model = AutoEncoder()
model = model.to(device)
model = torch.nn.DataParallel(model)
criterion = torch.nn.L1Loss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
best_loss = 1000

for epoch in range(num_epoches):
    model.train()
    for i, (images) in enumerate(train_loader):
        #images = images.view(-1, 320*320)
        images = images.to(device)
        #pdb.set_trace()
        encoded, decoded = model(images)
        loss = criterion(images, decoded)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1) % 100 ==0:
            print('Epoch [%d/%d], Iter[%d/%d] Loss:%.4f'%(epoch+1, num_epoches, i+1, len(train_data)//batch_size, loss.item()))

    model.eval()
    val_loss = 0
    for i, (images) in enumerate(val_loader):
        #images = images.view(-1, 320*320)
        images = images.to(device)
        encoded, decoded = model(images)
        loss = criterion(images, decoded)
        val_loss += loss.item()
        if val_loss/batch_size < best_loss:
            best_epoch = epoch
            best_loss = val_loss/batch_size
            best_model_wts = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    if not (epoch+1) % 10:
        model_wts = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        torch.save(model_wts, os.path.join('./output', str(epoch+1)+'_'+str(val_loss/batch_size)[0:6]+'_'+'ae_real.pth'))
    scheduler.step(val_loss)
    print('Epoch [%d/%d], Val Loss:%.4f'%(epoch+1, num_epoches, val_loss/len(val_data)))
torch.save(best_model_wts, os.path.join('./output', str(best_epoch+1)+'_'+str(best_loss/batch_size)[0:6]+'_'+'best_ae.pth'))