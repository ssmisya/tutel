# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import copy
import os
import re
import time
import logging 
import collections
import importlib

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import ModuleList
import torch.nn.functional as F

from ..impls import communicate as C
from ..impls.fast_dispatch import fast_encode, fast_decode, extract_critical, get_dispatch_count
from ..impls.overlap import a2a_ffn_overlap_forward
from . import losses
from .moe.moe_expert import LinearExperts
from .moe.moe_router import MOERouter





class MOELayerCustom(torch.nn.Module):
    def __init__(
        self,
        gate_type,
        model_dim: int,
        experts=None,
        scan_expert_func=None,
        result_func=None,
        seeds=None,
        batch_prioritized_routing=False,
        normalize_gate=True,
        is_gshard_loss=True,
        **kwargs
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_global_experts = experts
        self.scan_expert_func = scan_expert_func
        self.result_func = result_func

        self.is_gshard_loss = is_gshard_loss
        self.gate_type = gate_type
        self.experts = experts
        self.seeds = seeds
        
        self.num_global_experts = experts['count_per_node']
        self.expert_hidden_size = experts.get('hidden_size', 2048)
        self.expert_activation_fn = experts.get('activation_fn', None)
        self.is_gshard_loss = is_gshard_loss
        
        self.l_aux = None

        self.router = MOERouter(gate_type=gate_type,
                                model_dim=model_dim,
                                num_experts=self.num_global_experts,
                                seeds=seeds,
                                is_gshard_loss=self.is_gshard_loss,
                                **kwargs)
        
        self.experts = LinearExperts(in_features=model_dim,
                                     out_features=model_dim,
                                     num_experts=self.num_global_experts,
                                     hidden_size_per_expert=self.expert_hidden_size,
                                     activation_fn=self.expert_activation_fn,
                                     bias=True)
        
    def forward(self, input: Tensor,):
        
        scores,indices,self.l_aux = self.router(input) #[E, B, L, 1]
        x = input.unsqueeze(0).repeat(self.num_global_experts, 1, 1, 1) * indices #[E, B, L, D]
        x = [self.experts(x[i], i)*scores[i] for i in range(self.num_global_experts)] # E * [B, L, D]
        
        x = sum(x)
        return x




moe_layer = MOELayerCustom