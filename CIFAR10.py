import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers = 0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers = 0)

class Net(nn.Module):
    def __init__ (self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,64,4, stride=1, padding=2)
        self.batch_normalize = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,64,4, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p=0.5)
        self.conv3 = nn.Conv2d(64,64,4, stride=1, padding=2)
        self.conv4 = nn.Conv2d(64,64,4, stride=1, padding=2)
        self.conv5 = nn.Conv2d(64,64,4, stride=1, padding=2) 
        self.conv6 = nn.Conv2d(64,64,3, stride=1, padding=0)
        self.conv7 = nn.Conv2d(64,64,3, stride=1, padding=0)
        self.conv8 = nn.Conv2d(64,64,3, stride=1, padding=0)
        self.full_conn_1 = nn.Linear(64*4*4, 500)
        self.full_conn_2 = nn.Linear(500, 500)
        self.full_conn_3 = nn.Linear(500, 10)
        
    def forward(self, x):
        x = self.batch_normalize(F.relu(self.conv1(x)))
        x = self.drop(self.pool(F.relu(self.conv2(x))))
        x = self.batch_normalize(F.relu(self.conv3(x)))
        x = self.drop(self.pool(F.relu(self.conv4(x))))
        x = self.batch_normalize(F.relu(self.conv5(x)))
        x = self.drop(F.relu(self.conv6(x)))
        x = self.batch_normalize(F.relu(self.conv7(x)))
        x = self.drop(self.batch_normalize(F.relu(self.conv8(x))))
        x = x.view(-1, 64*4*4)
        x = F.relu(self.full_conn_1(x))
        x = F.relu(self.full_conn_2(x))
        x = self.full_conn_3(x)
        return x

net = Net()
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# training the model

for epoch in range(200):  # loop over the dataset multiple times

    total_right = 0
    total = 0

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        total_right += (predicted == labels.data).float().sum()
        
    print("Training Accuracy for epoch {} : {}".format(epoch+1,total_right/total))

    if (epoch+1) % 5 == 0:
        torch.save(net, 'save_params.ckpt')

# test the model

my_model = torch.load('save_params.ckpt')

total_right = 0
total = 0


with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = my_model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        total_right += (predicted == labels.data).float().sum()

print('Test accuracy: %d %%' % (
    100 * total_right / total))




