import argparse
import math
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('augmentation')
parser.add_argument('--tau', type=float, default=0.15)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=32)
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

print('Loading data...')
ds = getattr(data, args.dataset)

transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((int(224*1.1), int(224*1.1)), antialias=True),
    v2.RandomCrop((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(0.5 if ds.allow_vflips else 0),
    v2.ColorJitter(0.1, 0.1),
    v2.ToDtype(torch.float32, scale=True),
])

ds = ds('/nas-ctm01/homes/bmsa/data', transforms)
tr, _ = torch.utils.data.random_split(ds, [0.8, 0.2], torch.Generator().manual_seed(42))

# since we want to do augmentation across all the labels, we want to iterate across
# images from each class at the same time
tr_labels = [tr[i][1] for i in range(len(tr))]
tr_per_class = [torch.utils.data.Subset(tr, [i for i, l in enumerate(tr_labels) if l == k]) for k in range(ds.num_classes)]
tr_per_class = [torch.utils.data.DataLoader(d, args.batchsize, True, pin_memory=True) for d in tr_per_class]  # , num_workers=4

######################################### MODEL #########################################

model = torchvision.models.resnet18(weights='DEFAULT')
model.fc = torch.nn.Linear(512, ds.num_classes)
model.to(device)

###################################### ORDINAL AUG ######################################

def exp(num_classes, center_class, tau, device):
    x = torch.arange(num_classes, dtype=torch.float, device= device)
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
    labels = torch.zeros((n, num_classes), device=device)
    class_1 = torch.randint(0, num_classes, [n], device=device)
    class_2 = torch.randint(0, num_classes-1, [n], device=device)
    class_2[class_2 >= class_1] = 1+class_2[class_2 >= class_1]

    beta = torch.distributions.beta.Beta(torch.tensor(alpha), torch.tensor(alpha))
    temp = beta.sample([n]).to(device)
    labels[range(n), class_1] = temp
    labels[range(n), class_2] = 1-temp

    mixup_images = torch.sum(images * labels[:, :, None, None, None], 1)
    return mixup_images, labels

def ordinal_adjacent_mixup(images):
    device = images.device
    n = images.shape[0]
    num_classes = images.shape[1]
    labels = torch.zeros((n, num_classes), device=device)
    class_1 = torch.randint(0, num_classes-1, [n], device=device)
    class_2 = 1+class_1  # <-- difference between this and the previous one

    beta = torch.distributions.beta.Beta(torch.tensor(alpha), torch.tensor(alpha))
    temp = beta.sample([n]).to(device)
    labels[range(n), class_1] = temp
    labels[range(n), class_2] = 1-temp

    mixup_images = torch.sum(images * labels[:, :, None, None, None], 1)
    return mixup_images, labels

def ordinal_exponential_mixup(images):
    device = images.device
    n = images.shape[0]
    num_classes = images.shape[1]

    labels = torch.randint(0, num_classes, [n], device=device) + 1
    labels = exp(num_classes, labels, args.tau, device)

    mixup_images = torch.sum(images * labels[:, :, None, None, None], 1)
    return mixup_images, labels

def calcular_distance(H,W,area):
    raiz_discriminante = math.sqrt((2*(H + W))**2 - 16* area)
    distance = (-2*(H + W) + raiz_discriminante) / - 8
    return distance

def each_nested(images, probabilities, x1, y1, x2, y2, output):
    H, W = images.shape[2:]
    for k in range(1, len(probabilities)):
        w = int(W*sum(probabilities[k:]))
        h = int(H*sum(probabilities[k:]))
        x1 = torch.randint(x1, x2-w+1, ())
        y1 = torch.randint(y1, y2-h+1, ())
        x2 = x1+w
        y2 = y1+h
        output[:, y1:y2, x1:x2] = images[k][:, y1:y2, x1:x2]

def vectorize(fn):
    def f(batch_images):
        result = [fn(images) for images in batch_images]
        return torch.stack([r[0] for r in result], 0), torch.stack([r[1] for r in result], 0)
    return f

def normalize_each_nested(images):
    device = images.device
    output = images[0].clone()
    num_classes = images.shape[0]
    center_class = torch.randint(0, num_classes, [1], device=device)
    probabilities = exp(num_classes, center_class, args.tau, device)[0]
    
    each_nested(images, probabilities, 0 , 0 , images[0].shape[2], images[0].shape[1], output)
    return output, probabilities

nested = vectorize(normalize_each_nested)

def random_ranges(intervals):
    intervals = [(v1, v2) for v1, v2 in intervals if v2 > v1]
    total_size = sum(v2-v1 for v1, v2 in intervals)
    r = intervals[0][0] + torch.randint(0, total_size, ())
    for (_, i1), (i2, _) in zip(intervals, intervals[1:]):
        # invalid interval, pass through
        r[r >= i1] = r[r >= i1] + (i2-i1)
    return r

def intersects(x1, y1, w1, h1, x2, y2, w2, h2):
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

def jaime_ordinal_cutmix(images, probabilities, center):
    output = images[center].clone()

    H, W = output.shape[1:]
    wl = int(W * sum(probabilities[:center]))
    hl = int(H * sum(probabilities[:center]))
    wr = int(W * sum(probabilities[center + 1:]))
    hr = int(H * sum(probabilities[center + 1:]))

    while True:
        restrict_x = torch.rand(()) < 0.5
        x1_left = random_ranges([(0, W - wl - wr + 1), (wr, W - wl)]) if restrict_x and W - wl - wr + 1 < wr else torch.randint(0, W - wl, ())
        y1_left = random_ranges([(0, H - hl - hr + 1), (hr, H - hl)]) if not restrict_x and H - hl - hr + 1 < hr else torch.randint(0, H - hl, ())
        
        x1_right = random_ranges([(0, x1_left - wr), (x1_left + wl, W - wr)]) if restrict_x else torch.randint(0, W - wr, ())
        y1_right = random_ranges([(0, y1_left - wr), (y1_left + hl, H - hr)]) if not restrict_x else torch.randint(0, H - hr, ())
        
        if not intersects(x1_left, y1_left, wl, hl, x1_right, y1_right, wr, hr):
            break

    each_nested(images[:center], probabilities[:center], x1_left, y1_left, x1_left + wl, y1_left + hl, output)
    each_nested(images[center + 1:], probabilities[center + 1:], x1_right, y1_right, x1_right + wr, y1_right + hr, output)
    
    return output

def normalize_jaime_ordinal_cutmix(images):
    device = images.device
    num_classes = images.shape[0]
    center_class = torch.randint(0, num_classes, [1], device=device)
    probabilities = exp(num_classes, center_class, args.tau,device)[0]
    output = jaime_ordinal_cutmix(images, probabilities, center_class[0])
    return output, probabilities

jaime = vectorize(normalize_jaime_ordinal_cutmix)

# TODO:
# - nested
# - jaime

######################################### LOOP #########################################

print('Training...')
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
aug = globals()[args.augmentation]

nits = int(round(len(tr) * args.epochs / args.batchsize))
nprint = len(tr) // args.batchsize
avg_loss = avg_acc = 0
tic = time()
for it in range(nits):
    images = torch.stack([next(iter(d))[0] for d in tr_per_class], 1)
    images = images.to(device)
    old_images = images
    # images: input (32, 7, 3, 224, 224) => output (32, 3, 224, 224)
    images, labels = aug(images)
    #print('images:', images.min(), images.max(), 'labels:', labels)

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
    avg_acc += float((labels == preds.argmax(1)).float().mean()) / nprint

    if (it+1) % nprint == 0:
        toc = time()
        print(f'Iteration {it+1}/{nits} - {toc-tic:.0f}s - Avg loss: {avg_loss} - Avg acc: {avg_acc}')
        avg_loss = avg_acc = 0
        tic = time()

torch.save(model, f'model-{args.dataset}-{args.augmentation}.pth')

'''
import torchmetrics

acc = torchmetrics.Accuracy(...)

for images, labels in ts:
    with torch.no_grad():
        preds = model(images)
    acc.update(preds.argmax(1), labels)
print(acc.compute())
'''

