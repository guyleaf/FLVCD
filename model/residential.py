import torch
import torch.nn as nn


class Residential(nn.Module):
    def forward(self, x, layer_output):
        return x + layer_output
