import torch
import torch.nn as nn
import torch.nn.functional as F


def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)


def adaptive_clip_grad(parameters, clip_factor=0.01, eps=1e-3, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is None:
            continue
        p_data = p.detach()
        g_data = p.grad.detach()
        max_norm = unitwise_norm(p_data, norm_type=norm_type).clamp_(min=eps).mul_(clip_factor)
        grad_norm = unitwise_norm(g_data, norm_type=norm_type)
        clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
        new_grad = torch.where(grad_norm < max_norm, g_data, clipped_grad)
        p.grad.detach().copy_(new_grad)


class KLLossSoft(torch.nn.modules.loss._Loss):
    def forward(self, output, target, T=1.0):
        # print('output:', output.shape, 'targe:', target.shape)
        output = output[0] if isinstance(output, tuple) else output
        target = target[0] if isinstance(target, tuple) else target
        output, target = output / T, target / T
        target_prob = F.softmax(target, dim=1)
        output_log_prob = F.log_softmax(output, dim=1)
        loss = - torch.sum(target_prob * output_log_prob, dim=1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class KLTokenMSELoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        kd_type: str = "last",
        reduction: str = "mean",
    ):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.kl_loss = KLLossSoft(reduction=reduction)
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.kd_type = kd_type

    def _kl_loss(self, output, target):
        return self.kl_loss(output, target)

    def _mse_loss(self, output, target):
        mse_loss = 0
        if self.kd_type == "last":
            if isinstance(output, torch.Tensor):
                _, N, _ = target.size()
                mse_loss = self.mse_loss(output[:, -N:], target)
            else:
                _, N, _ = target[-1].size()
                mse_loss = self.mse_loss(output[-1][:, -N:], target[-1])
        elif self.kd_type == "all":
            if isinstance(output, torch.Tensor):
                _, N, _ = target.size()
                mse_loss = self.mse_loss(output[:, -N:], target)
            else:
                assert len(output) == len(target)
                for i in range(len(output)):
                    _, N, _ = target[i].size()
                    mse_loss += self.mse_loss(output[i][:, -N:], target[i])
                mse_loss = mse_loss / len(output)
        else:
            raise NotImplementedError
        return mse_loss

    def forward(self, output, target):
        assert len(output) == len(target)
        # print('output:', output.shape, 'targe:', target.shape)
        kl_loss = self.kl_loss(output[0], target[0])
        mse_loss = self._mse_loss(output[1], target[1])
        loss = kl_loss + self.alpha * mse_loss
        # print(f"KL loss {kl_loss}, MSE loss {mse_loss}, total loss {loss}")

        return loss


# class KLLossSoft(torch.nn.Module):
#     def __init__(
#         self,
#         base_loss,
#         alpha: float = 0.5,
#         temperature: float = 1.0,
#         reduction: str = "mean"
#     ):
#         super().__init__()
#         self.base_loss = base_loss
#         self.alpha = alpha
#         self.temperature = temperature
#         self.reduction = reduction
# 
#     def _soft_loss(self, output, target):
#         output = output / self.temperature
#         target = target / self.temperature
#         target_prob = F.softmax(target, dim=1)
#         output_log_prob = F.log_softmax(output, dim=1)
#         loss = - torch.sum(target_prob * output_log_prob, dim=1)
#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         else:
#             return loss
# 
#     def _hard_loss(self, output, target):
#         return self.base_loss(output, target)
# 
# 
#     def forward(self, output, target):
#         if isinstance(target, tuple):
#             assert len(target) == 2
#             target, teacher_output = target[0], target[1]
#         else:
#             target, teacher_output = None, target
# 
#         if isinstance(output, tuple):
#             assert len(output) == 2
#             soft_loss = self._soft_loss(output[0], teacher_output)
#             if target is not None:
#                 hard_loss = self._hard_loss(output[1], target)
#                 loss = (1. - self.alpha) * hard_loss + self.alpha * soft_loss
#             else:
#                 loss = soft_loss
#         else:
#             soft_loss = self._soft_loss(output, teacher_output)
#             if target is not None:
#                 hard_loss = self._hard_loss(output, target)
#                 loss = (1. - self.alpha) * hard_loss + self.alpha * soft_loss
#             else:
#                 loss = soft_loss
# 
#         return loss
