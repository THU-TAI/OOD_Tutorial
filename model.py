import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torch.autograd as autograd
import copy
import helper

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Featurizer(nn.Module):
    def __init__(self):
        super(Featurizer, self).__init__()
        self.network = torchvision.models.resnet18(pretrained=False)
        self.output_dim = self.network.fc.in_features
        del self.network.fc
        self.network.fc = Identity()

    def forward(self, x):
        return self.network(x)

    def get_output_dim(self):
        return self.output_dim

class Classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(Classifier, self).__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.layer(x)


class ERM(nn.Module):
    def __init__(self, num_classes, args):
        super(ERM, self).__init__()
        self.featurizer = Featurizer()
        features_in = self.featurizer.get_output_dim()
        self.classifier = Classifier(features_in, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    def update(self, minibatches):
        all_x, all_y = minibatches
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)



class Mixup(ERM):
    """
    Mixup of minibatches from different domains (https://github.com/facebookresearch/DomainBed/blob/25f173caa689f20828629b2e42f90193f203fdfa/domainbed/algorithms.py#L410)
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, num_classes, args):
        super(Mixup, self).__init__(num_classes, args)
        self.args = args

    def update(self, minibatches, unlabeled=None):
        loss = 0

        for idx, minibatches_domain in enumerate(minibatches): # randomize the minibatch
            rand_idx = torch.randperm(len(minibatches_domain[0]))
            minibatches[idx] = [minibatches_domain[0][rand_idx], minibatches_domain[1][rand_idx]]

        for (xi, yi), (xj, yj) in helper.random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.args.mixup_alpha, self.args.mixup_alpha)
            xi, yi, xj, yj = xi.cuda(), yi.cuda(), xj.cuda(), yj.cuda()
            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            loss += lam * F.cross_entropy(predictions, yi)
            loss += (1 - lam) * F.cross_entropy(predictions, yj)

        loss /= len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
