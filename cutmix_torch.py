import torch, torchvision
import matplotlib.pyplot as plt

ds = torchvision.datasets.STL10('/data/toys')
images = []
for i in range(10):
    for x, y in ds:
        if y == i:
            x = torchvision.transforms.functional.pil_to_tensor(x)/255
            images.append(x)
            break

def nested_ordinal_cutmix(images, probabilities, x1, y1, x2, y2, output):
    _, H, W = output.shape
    for k in range(1, len(probabilities)):
        w = int(W*sum(probabilities[k:]))
        h = int(H*sum(probabilities[k:]))
        x1 = torch.randint(x1, x2-w+1, ())
        y1 = torch.randint(y1, y2-h+1, ())
        x2 = x1+w
        y2 = y1+h
        output[:, y1:y2, x1:x2] = images[k][:, y1:y2, x1:x2]

def random_ranges(intervals):
    intervals = [(v1, v2) for v1, v2 in intervals if v2 > v1]
    total_size = sum(v2-v1 for v1, v2 in intervals)
    r = intervals[0][0] + torch.randint(0, total_size, ())
    for (_, i1), (i2, _) in zip(intervals, intervals[1:]):
        # invalid interval, pass through
        r[r >= i1] = r[r >= i1] + (i2-i1)
    return r

def jaime_ordinal_cutmix(images, probabilities, center):
    output = images[center].clone()
    _, H, W = output.shape
    wl = int(W*sum(probabilities[:center]))
    hl = int(H*sum(probabilities[:center]))
    wr = int(W*sum(probabilities[center+1:]))
    hr = int(H*sum(probabilities[center+1:]))
    # either constrain the x or y axis. no need to constrains both at the same time.
    # to be valid: x1 >= wr or W-x2 >= wr
    # therefore invalid: x1 < wr and x1 > W-wl-wr
    # in other words: x1 cannot be in [W-wl-wr+1, wr-1]
    # if invalid2 < invalid1, then it can be placed anywhere
    restrict_x = torch.rand(()) < 0.5
    x1 = random_ranges([(0, W-wl-wr+1), (wr, W-wl)]) if restrict_x and W-wl-wr+1 < wr else torch.randint(0, W-wl, ())
    y1 = random_ranges([(0, H-hl-hr+1), (hr, H-hl)]) if not restrict_x and H-hl-hr+1 < hr else torch.randint(0, H-hl, ())
    nested_ordinal_cutmix(images[:center][::-1], probabilities[:center][::-1], x1, y1, x1+wl, y1+hl, output)
    x1 = random_ranges([(0, x1-wr), (x1+wl, W-wr)]) if restrict_x else torch.randint(0, W-wr, ())
    y1 = random_ranges([(0, y1-wr), (y1+hl, H-hr)]) if not restrict_x else torch.randint(0, H-hr, ())
    nested_ordinal_cutmix(images[center+1:], probabilities[center+1:], x1, y1, x1+wr, y1+hr, output)
    return output

probabilities = [0.1]*10
output = images[0].clone()
#nested_ordinal_cutmix(images, probabilities, 0, 0, images[0].shape[2], images[0].shape[1], output)
center = 5
output = jaime_ordinal_cutmix(images, probabilities, center)
for k in range(10):
    plt.subplot(3, 5, k+1)
    plt.imshow(images[k].permute(1, 2, 0))
plt.subplot(3, 5, 11)
plt.imshow(output.permute(1, 2, 0))
plt.suptitle('Nested cutmix ' + str([0.1]*10))
plt.show()