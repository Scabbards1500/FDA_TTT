from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
from utils.dice_score import dice_loss

class Testmodel(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model):
        super(Testmodel,self).__init__()
        self.model= model
        self.out = model.outc

    # def __init__(self):
    #     super(Testmodel,self).__init__()


    def forward(self, x):
        x = self.model(x)
        return x

