# %% Importing Dependencies

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np




# %% Transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5))])




# %% data

trainset = torchvision.datasets.FashionMNIST(
    root='data',
    download=True,
    train=True,
    transform=transform)

testset = torchvision.datasets.FashionMNIST(
    root='data',
    download=True,
    train=False,
    transform=transform)




# %% dataLoaders

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')




# %% helper function for showing images with plt

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img/2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))




# %% Define model

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu((self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()




# %% Optimizer and loss

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)




# %% TensorBoard summary setup

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/fashion_mnist_experiment_1')




# %% Showing images using tensorboard

dataiter = iter(trainloader)
images, labels = next(dataiter)
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
writer.add_image("four_fashion_items", img_grid)
writer.add_graph(model, images)
writer.close()




# %% Tensorboard Projector

def select_n_random(data, labels, n=100):
    
    assert len(data) == len(labels)
    
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

images, labels = select_n_random(trainset.data, trainset.targets)
class_labels = [classes[lab] for lab in labels]

features = images.view(-1, 28 * 28)
writer.add_embedding(features, 
                     metadata=class_labels,
                     label_img=images.unsqueeze(1))
writer.close()


# %% Tracking model training with Tensorboard

images = images.unsqueeze(1).float()

def images_to_probs(model, images):
    output = model(images)
    _, pred_labels = torch.max(output, 1)
    preds = np.squeeze(pred_labels.numpy())
    return preds, [F.softmax(logits, dim=0)[max_label].item() for max_label, logits in zip(preds, output)]

def plot_classes_preds(model, images, labels):
    preds, probs = images_to_probs(model, images)
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx]*100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx]==labels[idx].item() else "red")
        )
    return fig
    
# %% Train loop

model.train()
running_loss = 0.0

for epoch in range(1):
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            writer.add_scalar('training loss',    # graph title
                              running_loss / 1000,   # y axis
                              epoch * len(trainloader) + i)  # x axis
            
            writer.add_figure('predictions vs labels',
                              plot_classes_preds(model, inputs, labels),
                              global_step=epoch * len(trainloader) + i)
            running_loss = 0.0
    print("Finished Training")



#  %% test loop

model.eval()
class_probs = []
class_labels = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = model(images)
        class_probs_batch = [F.softmax(logit, dim=0) for logit in output] # here we want the entire probability
        class_probs.append(class_probs_batch)                             # distribution, not just max
        class_labels.append(labels)                                                             

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])     # combine prob batches, then concatenate them
test_labels = torch.cat(class_labels)




# %% Adding PR Curves to Tensorboard

def add_pr_curve_tensorboard(class_index, test_probs, test_label, global_step=0):

    tensorboard_truth = test_label == class_index  # check which images have class_index as label
    tensorboard_probs = test_probs[:, class_index] # get probability of class_index for each input

    writer.add_pr_curve(classes[class_index],
                         tensorboard_truth,
                         tensorboard_probs,
                         global_step=global_step)
    writer.close()

for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_labels)
# %%
