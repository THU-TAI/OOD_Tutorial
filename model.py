import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torch.autograd as autograd
import copy

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Featurizer(nn.Module):
    def __init__(self):
        super(Featurizer, self).__init__()
        self.network = torchvision.models.resnet18(pretrained=True)
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
