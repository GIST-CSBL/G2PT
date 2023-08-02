import torch
from fast_soft_sort.pytorch_ops import soft_rank

class CCCLoss(torch.nn.Module):

    def __init__(self, eps=1e-6):
        super(CCCLoss, self).__init__()
        self.eps = eps

    def forward(self, y_true, y_hat):
        y_true_mean = torch.mean(y_true)
        y_hat_mean = torch.mean(y_hat)
        y_true_var = torch.var(y_true)
        y_hat_var = torch.var(y_hat)
        y_true_std = torch.std(y_true)
        y_hat_std = torch.std(y_hat)
        vx = y_true - torch.mean(y_true)
        vy = y_hat - torch.mean(y_hat)
        pcc = torch.sum(vx * vy) / (
                    torch.sqrt(torch.sum(vx ** 2) + self.eps) * torch.sqrt(torch.sum(vy ** 2) + self.eps))
        ccc = (2 * pcc * y_true_std * y_hat_std) / \
              (y_true_var + y_hat_var + (y_hat_mean - y_true_mean) ** 2)
        ccc = 1 - ccc
        return ccc

def corrcoef(target, pred):
    # np.corrcoef in torch from @mdo
    # https://forum.numer.ai/t/custom-loss-functions-for-xgboost-using-pytorch/960
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()


def spearman(
    target,
    pred,
    regularization="l2",
    regularization_strength=1.0,
):
    # fast_soft_sort uses 1-based indexing, divide by len to compute percentage of rank
    pred = soft_rank(
        pred,
        regularization=regularization,
        regularization_strength=regularization_strength,
    )
    return corrcoef(target, pred / pred.shape[-1])