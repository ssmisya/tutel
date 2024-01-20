import torch
from torch.distributions.normal import Normal

def multi_one_hot(indices, num_classes):
    one_hot = torch.zeros(*indices.shape[:-1], num_classes)
    one_hot.scatter_(1, indices, 1)
    return one_hot

def scatter_values_3d(indices, values, num_classes):
    output = torch.zeros(*indices.shape[:-1], num_classes,device=indices.device)
    output.scatter_(-1, indices, values)
    return output

def load_importance_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise):
    def load_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise):
        assert gate_noise > 0, "`gate_noise` must be > 0 for normalization in load_importance_loss()."
        normal = Normal(
            torch.tensor([0.0], device=scores_wo_noise.device),
            torch.tensor([gate_noise / num_global_experts], device=scores_wo_noise.device),
        )
        threshold = topk_logits[:, :, -1].view(-1, 1).float()
        diff = scores_wo_noise.view(-1, scores_wo_noise.shape[-1]).float() - threshold.float()
        prob = normal.cdf(diff)
        Load = prob.sum(0)
        l_load = Load.float().var() / (Load.float().mean() ** 2 + 1e-10)
        return l_load

    def importance_loss(scores_wo_noise):
        Impi = scores_wo_noise.view(-1, scores_wo_noise.shape[-1]).float().sum(0)
        l_imp = Impi.float().var() / (Impi.float().mean() ** 2 + 1e-10)

        return l_imp

    l_imp = importance_loss(scores_wo_noise)
    l_load = load_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise)
    return (l_imp + l_load) / 2.0