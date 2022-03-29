import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from os.path import join, dirname
import numpy as np
import random
import hashlib
import os
from torchvision.datasets import ImageFolder


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2 ** 31)


def _hparam(name, default_val, random_val_fn, random_seed=0):
    """Define a hyperparameter. random_val_fn takes a RandomState and
    returns a random hyperparameter value."""
    random_state = np.random.RandomState(seed_hash(random_seed, name))
    return (default_val, random_val_fn(random_state))


class Args():
    def __init__(self):
        self.gpu_id = 0
        self.batch_size = 5
        self.num_classes = 5
        self.num_steps = 900
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.min_scale = 0.8
        self.max_scale = 1.0
        self.random_horiz_flip = 0.5
        self.jitter = 0.4        
        self.image_size = 222

        ### for mixup (DG)
        self.num_domains = 3
        # self.mixup_alpha = _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, -1))
        self.mixup_alpha = 0.2


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def _dataset_info(txt_file):
    with open(txt_file, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels



### for ERM w/o domain information

class MyDataset(data.Dataset):
    def __init__(self, names, labels, img_transformer=None):
        self.names = names
        self.labels = labels

        self.N = len(self.names)
        self._image_transformer = img_transformer
    
    def get_image(self, index):
        img = Image.open(self.names[index]).convert('RGB')
        return self._image_transformer(img)
        
    def __getitem__(self, index):
        img = self.get_image(index)
        return img, int(self.labels[index])

    def __len__(self):
        return len(self.names)


class InfDataLoader():
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.data_iter = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            data = next(self.data_iter)
        return data
        
    def __len__(self):
        return len(self.dataloader)


def get_train_transformer(args):
    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
    img_tr = img_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)

def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)

def get_ERM_dataloader(args, phase):
    assert phase in ["train", "test"]
    names, labels = _dataset_info('data/ERM_' + phase + '.txt')

    if phase == "train":
        img_tr = get_train_transformer(args)
    else:
        img_tr = get_val_transformer(args)
    mydataset = MyDataset(names, labels, img_tr)
    do_shuffle = True if phase == "train" else False
    if phase == "train":
        loader = InfDataLoader(mydataset, batch_size=args.batch_size, shuffle=do_shuffle, num_workers=4)
    else:
        loader = data.DataLoader(mydataset, batch_size=args.batch_size, shuffle=do_shuffle, num_workers=4)
    return loader

def test(network, dataloader, device):
    network.eval()
    corrects = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            output = network.predict(images)
            _, predictions = output.max(dim=1)
            corrects += torch.sum(predictions == labels)
    accuracy = float(corrects) / len(dataloader.dataset)
    network.train()
    return accuracy




### for mixup w/ domain information

class MyDataset_DG(data.Dataset):
    def __init__(self, names, n_domains, labels, img_transformer=None):
        self.names = names
        self.labels = labels
        self.n_domains = n_domains

        self.N = len(self.names)
        self._image_transformer = img_transformer
        self.names_domains = [self.names[i*self.N//n_domains:(i+1)*self.N//n_domains] for i in range(n_domains)]
        self.labels_domain = [self.labels[i*self.N//n_domains:(i+1)*self.N//n_domains] for i in range(n_domains)]

    def get_image_domain(self, index, domain_index):
        img = Image.open(self.names_domains[domain_index][index]).convert('RGB')
        return self._image_transformer(img), self.labels_domain[domain_index][index]

    def __getitem__(self, index):
        input_set = []
        for i in range(self.n_domains):
            input_set.append(list(self.get_image_domain(index, i)))
        return input_set

    def __len__(self):
        return int(self.N / self.n_domains)


def get_DG_dataloader(args, phase):
    assert phase in ["train", "test"]
    names, labels = _dataset_info('data/DG_' + phase + '.txt')

    if phase == "train":
        img_tr = get_train_transformer(args)
    else:
        img_tr = get_val_transformer(args)
    mydataset = MyDataset_DG(names, 3, labels, img_tr) # n_domains = 3
    do_shuffle = True if phase == "train" else False
    if phase == "train":
        loader = InfDataLoader(mydataset, batch_size=args.batch_size, shuffle=do_shuffle, num_workers=8)
    else:
        loader = data.DataLoader(mydataset, batch_size=args.batch_size, shuffle=do_shuffle, num_workers=4)
    return loader



def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs
