import torch
import torchvision
from torchvision import transoforms
from networks.autoencoders import AutoEncoder
import pdb

input_dim = 28*28
hidden_dim = 32


transformer = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

test_path = './test_images/000_003/'
save_path = './test_images/000_003_ae/'
if not os.path.exsits(save_path):
    os.makedirs(save_path)

image_list = os.listdir(test_path)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

model = AutoEncoder(input_dim, hidden_dim)
model.load_state_dict(torch.load('./output/14_0.0226_ae_real.pth'))

for image in image_list:
    img = cv2.imread(os.path.join(test_path, image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transformer(img)
    img.to(device)
    output = model(img)
    output = output.cpu().numpy()
    output = int(output*255)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_path, image), output)