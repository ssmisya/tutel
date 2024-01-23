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

from ...impls import communicate as C
from ...impls.fast_dispatch import fast_encode, fast_decode, extract_critical, get_dispatch_count
from ...impls.overlap import a2a_ffn_overlap_forward
from .. import losses
from .utils.util import scatter_values_3d, multi_one_hot, load_importance_loss

class MOERouter(torch.nn.Module):
    def __init__(self, 
                 num_experts,
                 model_dim,
                 router_type='normal',
                 seeds = 0,
                 gate_type=None,
                 is_gshard_loss=True,
                 **kwargs,
                 ) -> None:
        super().__init__()
        # model variables
        self.num_global_experts = num_experts
        self.model_dim = model_dim
        self.k = gate_type.get('k', 1)
        self.gate_type = gate_type.get('type', 'top')
        self.router_type = router_type
        self.is_gshard_loss = is_gshard_loss

        # observation variables
        self.topk_indices = None
        
        # router settings
        assert self.router_type in ['normal', 'experts_choice','adaptive']
        
        # seed settings
        if seeds is not None and seeds[0] is not None:
            torch.manual_seed(seeds[0])
        gate_type.pop('type')
        # gate settings
        try:
            single_gate = importlib.import_module(f'....gates.{self.gate_type}' , __name__)
        except ModuleNotFoundError:
            raise Exception("Unrecognized gate_type: %s" % self.gate_type)
        gate_module = single_gate.Gate(model_dim=self.model_dim, num_global_experts=self.num_global_experts, **gate_type)
        
        # gate noise settings
        if not hasattr(gate_module, 'gate_noise'):
            gate_module.gate_noise = gate_type.get('gate_noise', 0.0)
        if not hasattr(gate_module, 'capacity_factor'):
            gate_module.capacity_factor = gate_type.get('capacity_factor', float(os.environ.get('CAP_FACTOR', 1.0)))

        self.gate = gate_module
        
               
    def forward(self, 
                input: Tensor, ):
        logits = self.gate(input)
        logits_wo_noise = F.softmax(logits, dim=-1)
        
        if self.training and self.gate.gate_noise > 0:
            logits_w_noise = logits + self.gate.gate_noise * torch.randn_like(logits) / self.num_global_experts
        else:
            logits_w_noise = logits
        
        
        if self.is_gshard_loss:
            _loss_fn = lambda gates, topk_ids: losses.gshard_loss(gates, topk_ids)
        else:
            _loss_fn = lambda gates, topk_ids: load_importance_loss(
            logits_wo_noise, logits_w_noise.gather(index=topk_ids, dim=-1),
            self.num_global_experts, self.gate.gate_noise)
        
            
        if self.router_type == 'normal':
            return self.topk_routing(_loss_fn, logits_w_noise,)
        elif self.router_type == 'experts_choice':
            return self.expert_choice_routing(logits_w_noise,)
        else:
            raise NotImplementedError(f'Not implemented router type: {self.router_type}')
            
        
    def topk_routing(self,loss_fn,gate_logits):
        scores = F.softmax(gate_logits, dim=-1) # [B, L, E]
        top_k, top_k_original = min(self.k, self.num_global_experts), self.k # [B, L, E]
        topk_values,topk_indices = torch.topk(scores, top_k, dim=-1) # [B, L, k]
        self.topk_indices = topk_indices
        l_loss = loss_fn(scores, topk_indices) if loss_fn is not None else None
        indicator_topk_importance_score = scatter_values_3d(topk_indices, topk_values, self.num_global_experts) # [B, L, E]
        indicator_topk_importance_score = indicator_topk_importance_score.permute(2, 0, 1).unsqueeze(-1) # [E, B, L, 1]
        indicator_topk_indices = scatter_values_3d(topk_indices, 1,self.num_global_experts) # [B, L, E]
        indicator_topk_indices = indicator_topk_indices.permute(2, 0, 1).unsqueeze(-1) # [E, B, L, 1]

        return indicator_topk_importance_score,indicator_topk_indices, l_loss
    
    def encode(self, input: Tensor, topk_indices: Tensor,importance_score: Tensor):
        x = input
        output = [[]]*self.num_global_experts
        index = [[]]*self.num_global_experts
        scores = [[]]*self.num_global_experts
        for i in range(x.shape[0]): # B
            for j in range(x.shape[1]): # L
                for k in range(topk_indices.shape[-1]):
                    output[topk_indices[i][j][k].item()].append(x[i][j])
                    index[topk_indices[i][j][k].item()].append([i,j])
                    scores[topk_indices[i][j][k].item()].append(importance_score[k])
        output = [torch.stack(output[i]) for i in range(self.num_global_experts)] # E * [n, D]
        return output, index, scores
    
    def decode(self, input: list, index: list, scores: list,batch_size: int, seq_len: int,):
        '''
            input: E * [n, D]
            index: E * [n, 2]
            importance_score: [ B , L , k ]
        '''
        x = torch.zeros(batch_size, seq_len, self.model_dim).to(input[0][0].device)
        for i in range(self.num_global_experts):
            for j in range(len(index[i])):
                x[index[i][j][0]][index[i][j][1]] = input[i][j] * scores[i][j]
        return x
    
    def expert_choice_routing(self,gate_logits,):
        batch_size = gate_logits.shape[0]
        sequence_length = gate_logits.shape[1]
        n = batch_size * sequence_length
        k = (n * self.k) / self.num_global_experts
        logits = gate_logits.reshape(-1, self.num_global_experts)
        scores = F.softmax(gate_logits, dim=0) # [B*L, E]
        topk_values,topk_indices = torch.topk(scores, k, dim=0) # [k, E]
         
        indicator_topk_indices = torch.zeros_like(scores)
        indicator_topk_importance_score = torch.zeros_like(scores)
        
        indicator_topk_indices.scatter_(0,topk_indices,1).reshape(batch_size, sequence_length, self.num_global_experts).permute(2, 0, 1).unsqueeze(-1) # [E, B, L, 1]
        indicator_topk_importance_score.scatter_(0,topk_indices,topk_values).rashape(batch_size, sequence_length, self.num_global_experts).permute(2, 0, 1).unsqueeze(-1) # [E, B, L, 1]
        return indicator_topk_importance_score,indicator_topk_indices, 0
    
    