from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

from .Mbpa import ReplayMemory
import matplotlib.pyplot as plt

import numpy as np

torch.set_printoptions(precision=30)

buffer_size = 30


class Tent(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.memory = ReplayMemory(buffer_size)
        self.mse = nn.MSELoss()
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.gt = None

    def forward(self, x):
        if self.episodic:
            self.reset()
            print('Image-specific')

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer, self.memory, self.mse, self.gt)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.memory = ReplayMemory(buffer_size)
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    x_reshaped = x.view(-1, x.shape[2], x.shape[3])
    softmax_x = x_reshaped.softmax(0)
    log_softmax_x = x_reshaped.log_softmax(0)
    entropy = -(softmax_x * log_softmax_x).sum()/(x.shape[2]*x.shape[3])
    return entropy


@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, memory, mse, gt):
    outputs = model(x)
    features = model.inc(x)  # 模型的编码器 #这个地方从形状特征改成
    memory_size = memory.get_size()

    if memory_size > buffer_size:
        with torch.no_grad():
            retrieved_batches = memory.get_neighbours(features.cpu().numpy(), k=4)  # 4
            pseudo_past_logits = retrieved_batches.cuda()
            pseudo_current_logits = outputs
            pseudo_past_labels = nn.functional.softmax(pseudo_past_logits, dim=1)
            pseudo_current_labels = nn.functional.softmax(pseudo_current_logits / 2, dim=1)
            diff_loss = (F.kl_div(pseudo_past_labels.log(), pseudo_current_labels, None, None, 'none') + F.kl_div(
                pseudo_current_labels.log(), pseudo_past_labels, None, None, 'none')) / 2
            diff_loss = torch.sum(diff_loss, dim=1)
            diff_loss = diff_loss.cpu().numpy().tolist()
            sum_loss= sum(sum(sublist) for sublist in diff_loss[0])
            len_loss = len(diff_loss[0]) * len(diff_loss[0])
            diff_loss = sum_loss / len_loss

            for param_group in optimizer.param_groups:
                param_group['lr'] = diff_loss * param_group['lr']
                print("learningrate:", param_group['lr'])

    loss = softmax_entropy(outputs).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    outputs = model(x)

    with torch.no_grad():
        pseudo_logits = model(x)
        keys = model.inc(x)
        memory.push(keys.cpu().numpy(), pseudo_logits.cpu().numpy())

    return outputs


def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model