from .backbones import *
import torch
import torch.nn as nn


class ResnetModel(nn.Module):
    def __init__(self,
                 model_name: str,
                 n_labels: int,
                 ):
        super(ResnetModel, self).__init__()
        self.base = get_model(model_name, n_labels)
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        x = self.base(x)
        x = self.soft_max(x)
        return x, None

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

if __name__ == '__main__':
    out = torch.rand((128,35))
    out = out.max(dim=1, keepdim=True)
    probs = out[0].numpy().ravel()
    indices = out[1].numpy().ravel()
    print(probs, indices)