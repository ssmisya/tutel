import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN


class LinearExperts(nn.Module):
    """
    Modified from nn.Linear
    """

    __constants__ = ["bias", "in_features", "out_features", "num_experts"]

    def __init__(
        self, in_features, out_features, num_experts,hidden_size_per_expert,activation_fn=None, bias=True, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LinearExperts, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.hidden_size = hidden_size_per_expert
        self.activation_fn = activation_fn or (lambda x: F.relu(x))
        self.weight1 = nn.Parameter(
            torch.empty((num_experts, self.hidden_size, in_features), **factory_kwargs)
        )
        self.weight2 = nn.Parameter(
            torch.empty((num_experts, out_features, self.hidden_size), **factory_kwargs)
        )
        if bias:
            self.bias1 = nn.Parameter(
                torch.empty((num_experts, self.hidden_size), **factory_kwargs)
            )
            self.bias2 = nn.Parameter(
                torch.empty((num_experts, out_features), **factory_kwargs)
            )
        else:
            self.register_parameter("bias1", None)
            self.register_parameter("bias2", None)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight1[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight2[i], a=math.sqrt(5))
            if self.bias1 is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias1[i], -bound, bound)
            if self.bias2 is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias2[i], -bound, bound)

    def forward(self, input, i):
        x = input
        x = F.linear(x, self.weight1[i], self.bias1[i])
        x = self.activation_fn(x)
        x = F.linear(x, self.weight2[i], self.bias2[i])
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}, num_experts={}, bias={}".format(
            self.in_features, self.out_features, self.num_experts, self.bias is not None
        )



