import argparse
parser = argparse.ArgumentParser()
parser.add_argument('augmentation')
parser.add_argument('--tau', type=float, default=0.15)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

import torch
import torchvision
from torchvision.transforms import v2
from time import time
import matplotlib.pyplot as plt
import data

alpha = 1.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

######################################### DATA #########################################

transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((int(224*1.1), int(224*1.1)), antialias=True),
    v2.RandomCrop((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.ColorJitter(0.1, 0.1),
    v2.ToDtype(torch.float32, scale=True),
])

tr = ds = data.Smear(r"C:\Users\Beatriz\Desktop\3 year (2023-2024)\2ºsemestre\estagio\data", transforms)

# since we want to do augmentation across all the labels, we want to iterate across
# images from each class at the same time
ds_per_class = [torch.utils.data.Subset(tr, [i for i, l in enumerate(ds.labels) if l == k]) for k in range(ds.num_classes)]
ds_per_class = [torch.utils.data.DataLoader(d, 32, True, pin_memory=True) for d in ds_per_class]  # , num_workers=4

######################################### MODEL #########################################

model = torchvision.models.resnet18(weights='DEFAULT')
model.fc = torch.nn.Linear(512, ds.num_classes)
model.to(device)

###################################### ORDINAL AUG ######################################

def exp(num_classes, center_class, tau):
    x = torch.arange(num_classes, dtype=torch.float)
    return torch.nn.functional.softmax(-torch.abs(center_class[:, None] - x[None, :]) / tau, 1)

def none(images):  # images.shape = [N, K]. return.shape = [N]
    device = images.device
    n = images.shape[0]
    num_classes = images.shape[1]
    labels = torch.randint(0, num_classes, [n], device=device)
    return images[range(n), labels], labels

def mixup(images):
    device = images.device
    n = images.shape[0]
    num_classes = images.shape[1]
    labels = torch.zeros((n, num_classes))
    class_1 = torch.randint(0, num_classes, [n], device=device)
    class_2 = torch.randint(0, num_classes-1, [n], device=device)
    class_2[class_2 >= class_1] = 1+class_2[class_2 >= class_1]

    beta = torch.distributions.beta.Beta(torch.tensor(alpha), torch.tensor(alpha))
    temp = beta.sample([n])
    labels[range(n), class_1] = temp
    labels[range(n), class_2] = 1-temp

    mixup_images = torch.sum(images * labels[:, :, None, None, None], 1)
    return mixup_images, labels

def ordinal_adjacent_mixup(images):
    device = images.device
    n = images.shape[0]
    num_classes = images.shape[1]
    labels = torch.zeros((n, num_classes))
    class_1 = torch.randint(0, num_classes-1, [n], device=device)
    class_2 = 1+class_1  # <-- difference between this and the previous one

    beta = torch.distributions.beta.Beta(torch.tensor(alpha), torch.tensor(alpha))
    temp = beta.sample([n])
    labels[range(n), class_1] = temp
    labels[range(n), class_2] = 1-temp

    mixup_images = torch.sum(images * labels[:, :, None, None, None], 1)
    return mixup_images, labels

def ordinal_exponential_mixup(images):
    device = images.device
    n = images.shape[0]
    num_classes = images.shape[1]

    labels = torch.randint(0, num_classes, [n], device=device) + 1
    labels = exp(num_classes, labels, args.tau)

    mixup_images = torch.sum(images * labels[:, :, None, None, None], 1)
    return mixup_images, labels

def each_nested(images):
    # images: (7, 3, 224, 224)
    pass

def nested(batch_images):
    # images: (32, 7, 3, 224, 224)
    return torch.stack([each_nested(images) for images in batch_images], 0)


# TODO:
# - nested
# - jaime

######################################### LOOP #########################################

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
aug = globals()[args.augmentation]

nits = 10000
avg_loss = avg_acc = 0
tic = time()
for it in range(nits):
    images = torch.stack([next(iter(d))[0] for d in ds_per_class], 1)
    images = images.to(device)
    old_images = images
    # images: input (32, 7, 3, 224, 224) => output (32, 3, 224, 224)
    images, labels = aug(images)

    if args.debug:
        plt.subplot(2, 7, 1)
        plt.imshow(images[0].permute(1, 2, 0))
        for k in range(7):
            plt.subplot(2, 7, k+8)
            plt.imshow(old_images[0, k].permute(1, 2, 0))
        plt.show()

    preds = model(images)  # (N, K)
    loss = criterion(preds, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    avg_loss += float(loss) / len(tr)
    if len(labels.shape) == 2:
        labels = labels.argmax(1)
    avg_acc += float((labels == preds.argmax(1)).float().mean()) / len(tr)

    if (it+1) % 100 == 0:
        toc = time()
        print(f'Iteration {it+1}/{nits} - {toc-tic:.0f}s - Avg loss: {avg_loss} - Avg acc: {avg_acc}')
        avg_loss = avg_acc = 0
        tic = time()

torch.save(model, f'model-{args.augmentation}.pth')
