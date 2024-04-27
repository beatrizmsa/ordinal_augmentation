import argparse
parser = argparse.ArgumentParser()
parser.add_argument('augmentation')
args = parser.parse_args()

import torch
import torchvision
from torchvision.transforms import v2
from time import time
import data

alpha = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

######################################### DATA #########################################

transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((int(224*1.1), int(224*1.1))),
    v2.RandomCrop((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.ColorJitter(25, 25),
    v2.ToDtype(torch.float32, scale=True),
])

tr = ds = data.Smear('/nas-ctm01/homes/bmsa/data', transforms)

# since we want to do augmentation across all the labels, we want to iterate across
# images from each class at the same time
ds_per_class = [torch.utils.data.Subset(tr, [i for i, l in enumerate(ds.labels) if l == k]) for k in range(ds.num_classes)]
ds_per_class = [torch.utils.data.DataLoader(d, 32, True, pin_memory=True, num_workers=4) for d in ds_per_class]

######################################### MODEL #########################################

model = torchvision.models.resnet18(weights='DEFAULT')
model.fc = torch.nn.Linear(512, ds.num_classes)
model.to(device)

###################################### ORDINAL AUG ######################################

def none(images):
    device = images.device
    n = images.shape[0]
    num_classes = images.shape[1]
    labels = torch.randint(0, num_classes, [n], device=device)
    return images[range(len(images)), labels], labels

def ordinal_mixup(images):
    device = images.device
    n = images.shape[0]
    num_classes = images.shape[1]
    lables = torch.zeros(num_classes)
    class_1 = torch.randint(0,num_classes,[n], device=device)
    class_2 = torch.randint(0,num_classes,[n], device=device)

    beta = torch.distributions.beta.Beta(torch.tensor(alpha), torch.tensor(alpha))

    labels[class_1] = beta.sample()
    labels[class_2] = 1 - beta.sample()
    
    return 

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
    # images = (32, 7, 3, 224, 224) => (32, 7, 224, 224)
    images, labels = aug(images)

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
