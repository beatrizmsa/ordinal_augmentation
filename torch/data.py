import torch, torchvision
from skimage.io import imread
import os

class Smear:
    # https://mde-lab.aegean.gr/index.php/downloads/
    num_classes = 7
    classes = ['normal_superficiel', 'normal_intermediate', 'normal_columnar', 'light_dysplastic',
               'moderate_dysplastic', 'severe_dysplastic', 'carcinoma_in_situ']

    def __init__(self, root, transform=None):
        self.root = os.path.join(root, 'smear2005', 'New database pictures')
        self.files = [(klass, fname) for klass in self.classes for fname in os.listdir(os.path.join(self.root, klass))
            if fname.endswith('.BMP')]
        self.labels = [self.classes.index(klass) for klass, _ in self.files]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        dname, fname = self.files[i]
        label = self.labels[i]
        image = imread(os.path.join(self.root, dname, fname))
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    ds = Smear('/nas-ctm01/homes/bmsa/data')
    x, y = ds[0]
    print('x:', type(x), x.shape, x.dtype, x.min(), x.max())
    print('y:', y)