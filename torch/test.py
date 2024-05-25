import argparse
import torch
import torchvision
import torchmetrics
from torchvision.transforms import v2
import data

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('dataset')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

######################################### DATA #########################################

transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
])

ds = getattr(data, args.dataset)('/nas-ctm01/homes/bmsa/data', transforms)
_, ts = torch.utils.data.random_split(ds, [0.8, 0.2], torch.Generator().manual_seed(42))
ts = torch.utils.data.DataLoader(ts, 32, True, pin_memory=True)

######################################### MODEL #########################################

model = torch.load(args.model, map_location=device)

######################################### LOOP #########################################

acc = torchmetrics.Accuracy(task="multiclass", num_classes=ds.num_classes)
acc = acc.to(device)

for images, labels in ts:
    with torch.no_grad():
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
    acc.update(preds.argmax(1), labels)
print(f"Accuracy on all data: {acc.compute()}")