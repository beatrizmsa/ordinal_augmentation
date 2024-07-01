import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('augmentation')
parser.add_argument('--latent-aug', action='store_true')
parser.add_argument('--epochs', type=int, default=200)
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
print('Splitting by classes...')
ds.only_labels = True
tr_labels = [tr[i][1] for i in range(len(tr))]
ds.only_labels = False
tr_per_class = [torch.utils.data.Subset(tr, [i for i, l in enumerate(tr_labels) if l == k]) for k in range(ds.num_classes)]
tr_per_class = [torch.utils.data.DataLoader(d, args.batchsize, True, pin_memory=True) for d in tr_per_class]

######################################### MODEL #########################################

model = torchvision.models.resnet18(weights='DEFAULT')
model.fc = torch.nn.Linear(512, ds.num_classes)
model.to(device)

###################################### ORDINAL AUG ######################################

def exp(num_classes, center_class, tau, device):
    x = torch.arange(num_classes, dtype=torch.float, device= device)
    return torch.nn.functional.softmax(-torch.abs(center_class[:, None] - x[None, :]) / tau, 1)

def poisson_probs(num_classes, target_class, tau, device):
    # https://proceedings.mlr.press/v70/beckham17a/beckham17a.pdf
    _lambda = target_class[:, None]+0.5  # to ensure mode falls only on this class
    j = torch.arange(num_classes, dtype=torch.float, device=device)[None]
    log_pmf = j*torch.log(_lambda) - _lambda - torch.lgamma(j+1)
    return torch.softmax(log_pmf/tau, 1)

def none(images, tau):  # images.shape = [N, K]. return.shape = [N]
    device = images.device
    n = images.shape[0]
    num_classes = images.shape[1]
    labels = torch.randint(0, num_classes, [n], device=device)
    return images[range(n), labels], labels

def mixup(images, tau):
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

def normalize_cutmix(images,tau):
    device = images.device
    num_classes = images.shape[0]
    labels = torch.zeros((num_classes), device=device)
    class_1 = torch.randint(0, num_classes, [], device=device)
    class_2 = torch.randint(0, num_classes-1, [], device=device)

    class_2[class_2 >= class_1] = 1 + class_2[class_2 >= class_1]

    lamba = torch.distributions.beta.Beta(torch.tensor(alpha), torch.tensor(alpha))
    temp = lamba.sample().to(device)
    labels[class_1] = temp
    labels[class_2] = 1-temp

    H, W = images.shape[2:]
    
    rw = int(W * torch.sqrt(1 - temp))
    rh = int(H * torch.sqrt(1 - temp))
    rx = torch.randint(0, W - rw,())
    ry = torch.randint(0, H - rh, ())

    output = images[class_1].clone()

    output[:, ry:ry+rh, rx:rx+rw] = images[class_2][:, ry:ry+rh, rx:rx+rw]

    return output, labels

def ordinal_adjacent_mixup(images, tau):
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

def ordinal_exponential_mixup(images, tau):
    device = images.device
    n = images.shape[0]
    num_classes = images.shape[1]

    labels = torch.randint(0, num_classes, [n], device=device) + 1
    labels = poisson_probs(num_classes, labels, tau, device)

    mixup_images = torch.sum(images * labels[:, :, None, None, None], 1)
    return mixup_images, labels


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
    def f(batch_images, tau):
        result = [fn(images, tau) for images in batch_images]
        return torch.stack([r[0] for r in result], 0), torch.stack([r[1] for r in result], 0)
    return f

def normalize_each_nested(images, tau):
    device = images.device
    output = images[0].clone()
    num_classes = images.shape[0]
    center_class = torch.randint(0, num_classes, [1], device=device)
    probabilities = poisson_probs(num_classes, center_class, tau, device)[0]
    
    each_nested(images, probabilities, 0 , 0 , images[0].shape[2], images[0].shape[1], output)
    return output, probabilities


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
        x1_left = torch.randint(0, W - wl, ())
        y1_left = torch.randint(0, H - hl, ())
        
        x1_right = torch.randint(0, W - wr, ())
        y1_right = torch.randint(0, H - hr, ())
        
        if not intersects(x1_left, y1_left, wl, hl, x1_right, y1_right, wr, hr):
            break

    imag = images[:center + 1].tolist()
    imag = torch.tensor(imag[::-1])
    prob = probabilities[:center + 1].tolist()
    prob = torch.tensor(prob[::-1])
    each_nested(imag, prob, x1_left, y1_left, x1_left + wl, y1_left + hl, output)
    each_nested(images[center:], probabilities[center:], x1_right, y1_right, x1_right + wr, y1_right + hr, output)

    return output

def normalize_jaime_ordinal_cutmix(images, tau):
    device = images.device
    num_classes = images.shape[0]
    center_class = torch.randint(0, num_classes, [1], device=device)
    probabilities = poisson_probs(num_classes, center_class,tau,device)[0]
    output = jaime_ordinal_cutmix(images, probabilities, center_class[0])
    return output, probabilities

nested = vectorize(normalize_each_nested)
jaime = vectorize(normalize_jaime_ordinal_cutmix)
cutmix = vectorize(normalize_cutmix)

######################################### LOOP #########################################

print('Training...')
criterion = torch.nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model.fc.parameters())
optimizer2 = torch.optim.Adam(set(model.parameters()) - set(model.fc.parameters()), 1e-4)
aug = globals()[args.augmentation]

nits = int(round(len(tr) * args.epochs / args.batchsize))
nprint = len(tr) // args.batchsize
avg_loss = avg_acc = 0
tic = time()

for it in range(nits):
    tau = (1-(it+1)/nits)*1 + 0.00001

    images = torch.stack([next(iter(d))[0] for d in tr_per_class], 1)
    images = images.to(device)

    if not args.latent_aug:
        
        old_images = images
        # images: input (32, 7, 3, 224, 224) => output (32, 3, 224, 224)
        images, labels = aug(images, tau)
        #print('images:', images.min(), images.max(), 'labels:', labels)
        preds = model(images)
    
    if args.debug:
        plt.subplot(2, ds.num_classes, 1)
        plt.imshow(images[0].permute(1, 2, 0))
        for k in range(ds.num_classes):
            plt.subplot(2, ds.num_classes, k+ds.num_classes+1)
            plt.imshow(old_images[0, k].permute(1, 2, 0))
        plt.show()

    if args.latent_aug:
        # images.shape = (32, 7, 3, 224, 224)
        x = images.reshape(-1, 3, 224, 224)  # images.shape = (32*7, 3, 224, 224)
        for layer in list(model.children())[:-1]:
            x = layer(x)
        x = x.reshape(images.shape[0], images.shape[1], *x.shape[1:])
        x.to(device)
        x, labels = aug(x, tau)

        x = torch.mean(x, (2, 3))
        preds = model.fc(x)


    loss = criterion(preds, labels)
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss.backward()
    optimizer1.step()
    optimizer2.step()
    
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
