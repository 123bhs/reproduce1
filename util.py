import os
import random

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PATHOGENS'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32)


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def Spectral_Radius(eig):
    return ["{:.3f}".format(to_numpy(torch.max(torch.abs(torch.real(e)))).item()) for e in eig] if isinstance(eig, list) \
        else ["{:.3f}".format(to_numpy(max(torch.abs(torch.real(eig)))))]


def get_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params


def Cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, np.conjugate(vec2))
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)    
    return similarity


class EarlyStopping:
    """
    早停 + 学习率衰减（论文策略）：
    - 若连续 patience 个 epoch 验证损失未下降 → 学习率除以 5，并计一次“冷却”。
    - 当“冷却”次数超过 cold 次（即第 cold+1 次触发）→ 早停，训练结束。
    例如 patience=100, cold=3：第 1~3 次连续 100 epoch 不降只衰减 lr，第 4 次触发时停止训练。
    """
    def __init__(self, patience, cold, path='./checkpoint/Dynamic/dynamic.pth', use_mse=False):
        self.patience = patience
        self.cold = cold
        self.counter_p = 0
        self.counter_c = 0
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path
        self.use_mse = use_mse  # if True, track by MSE (align with test); else by Smooth L1

    def __call__(self, val_loss, model, optimizer, val_mse=None):
        track = val_mse if self.use_mse and val_mse is not None else val_loss
        if track < self.val_loss_min:
            self.val_loss_min = track
            self.save_checkpoint(model)
            self.counter_p = 0
        elif track >= self.val_loss_min:
            self.counter_p += 1
            if self.counter_p > self.patience:
                self.counter_c += 1
                if self.counter_c > self.cold:
                    self.early_stop = True
                optimizer.param_groups[0]['lr'] /= 5
                self.counter_p = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
