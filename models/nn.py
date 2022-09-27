from .backbones import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class Networks(nn.Module):
    def __init__(self,
                 model_name: str,
                 n_dim: int = None,
                 n_class: int = None,
                 classify: bool = False
                 ):
        super(Networks, self).__init__()
        self.base = get_model(model_name)
        self.n_class = n_class
        self.n_dim = n_dim
        self.classify = classify
        self.embedding = None
        if n_dim is not None:
            self.embedding = nn.Linear(self.base.get_last_dimension(),
                                       n_dim)

        if classify:
            if n_dim is None:
                n_feats = self.base.get_last_dimension()
                self.batch_norm = nn.BatchNorm1d(n_feats)
                self.batch_norm.bias.requires_grad = False
                self.classifier = ngn.Linear(n_feats,
                                            n_class,
                                            bias=False)
            else:
                self.batch_norm = nn.BatchNorm1d(n_dim)
                self.batch_norm.bias.requires_grad = False
                self.classifier = nn.Linear(n_dim,
                                            n_class,
                                            bias=False)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        feats = self.base(x)
        if self.embedding is not None:
            feats = self.embedding(feats)
        scores = None
        if self.classify:
            feats_norm = self.batch_norm(feats)
            scores = self.classifier(feats_norm)
        feats = F.normalize(feats, p=2, dim=1)
        return scores, feats

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


# Debug
if __name__ == '__main__':
    model = Networks('res15', None, 35, True)
    print(model)