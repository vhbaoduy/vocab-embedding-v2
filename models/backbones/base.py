import torch
import torch.nn as nn


class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_dimension = None

    def get_last_dimension(self):
        return self.last_dimension
