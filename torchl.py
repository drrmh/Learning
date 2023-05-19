import torch
import torchvision
import torchvision.transforms as transforms

import  numpy as np
import math
import torch.nn as nn

from matplotlib import pyplot as plt


trans = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

b_s = 8

trainset = torchvision.datasets.CIFAR10(root = 'H:/CIFAR10', train =True, download= True, transform=trans)

trainingloader = torch.utils.data.DataLoader(trainset, batch_size= b_s, shuffle= True, num_workers=2)

tests = torchvision.datasets.CIFAR10(root = 'H:/CIFAR10', train =True, download= True, transform=trans)

tstloader = torch.utils.data.DataLoader(tests, batch_size= b_s, shuffle= True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(classes)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainingloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(b_s)))


from loss import criterion, optimizer,net

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainingloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/2000:.3f}')
            running_loss = 0.0

print('Finished Training')

