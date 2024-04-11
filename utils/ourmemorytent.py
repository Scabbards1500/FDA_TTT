from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

from .ourMb import ReplayMemory
import matplotlib.pyplot as plt

import numpy as np
from .Fourier_Tans import FDA_get_amp_pha_tensor, FDA_target_to_source,arc_add_amp


torch.set_printoptions(precision=5)
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
    outputs = model(x)  # 这个是变化前的
    #
    amp,pha = FDA_get_amp_pha_tensor(x) #获取x的振幅和相位
    style = amp

    memory_size = memory.get_size()
    pseudo_past_logits_input = None

    for param_group in optimizer.param_groups:
        param_group['lr']= 1e-6


    diff_loss = 0

    if memory_size > 4:
        with torch.no_grad():
            retrieved_batches = memory.get_neighbours(style.cpu().numpy(), k=4)  # 找最接近的几个风格
            pseudo_past_style = retrieved_batches.cuda() # 找的近似的(1,3,256,256)
            pseudo_past_logits_input = arc_add_amp(pseudo_past_style, style,pha,L=0.8) #更改过去风格后的本次图片
            # 计算pseudo_past_style和style之间的KL散度
            diff_loss = F.kl_div(pseudo_past_style.log(), style, reduction='none')

            # 将KL散度作为损失添加到总损失中
            diff_loss = torch.sum(diff_loss, dim=1) # 获取的loss
            diff_loss = diff_loss.cpu().numpy().tolist()
            sum_loss= sum(sum(sublist) for sublist in diff_loss[0])
            len_loss= len(diff_loss[0])
            diff_loss = (sum_loss / len_loss) * 1e-2



    if pseudo_past_logits_input!= None:
        outputs = model(pseudo_past_logits_input)
    else:
        outputs = model(x)

    loss = softmax_entropy(outputs).mean(0)
    loss += abs(diff_loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        amp,pha = FDA_get_amp_pha_tensor(x)
        memory.push(amp.cpu().numpy(), amp.cpu().numpy())


    #这里output要换成tensor
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