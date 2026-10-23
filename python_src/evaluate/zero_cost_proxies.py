"""
Zero-Cost Proxies (ZCP) Engine
==============================
Single Responsibility Principle (SRP):
Dedicated exclusively to computing architecture saliency proxies (SNIP, SynFlow)
prior to training, enabling early pruning or validation of candidate subgraphs.
Fully supports multi-agent and HD-map branches.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from evaluate.metrics import ade, min_ade_k


def snip_score(net: nn.Module, batch: Dict[str, torch.Tensor], device: str = "cpu") -> float:
    """
    Single-shot Network Pruning (SNIP) Zero-Cost Proxy.
    Quantifies the saliency of parameters via gradient magnitude before full training:
      S = sum | (dL / d_theta) * theta |
    Higher score indicates higher initial gradient information flow.
    """
    net.zero_grad()
    past = batch["past"].to(device)
    future = batch["future"].to(device)
    target = future[:, :, 0, :] if future.dim() == 4 else future

    agents_past = batch.get("agents_past", None)
    if agents_past is not None and hasattr(agents_past, "to"):
        agents_past = agents_past.to(device)

    map_lanes = batch.get("map_lanes", None)
    if map_lanes is not None and hasattr(map_lanes, "to"):
        map_lanes = map_lanes.to(device)

    # Equal architectural weight distribution
    alphas = (
        torch.ones(5, device=device) / 5.0,
        torch.ones(5, device=device) / 5.0,
        torch.ones(4, device=device) / 4.0,
    )
    output = net(past, alphas, agents_past=agents_past, map_lanes=map_lanes)
    pred = output[0] if isinstance(output, tuple) else output

    loss = ade(pred, target)
    loss.backward()

    score = 0.0
    for p in net.parameters():
        if p.grad is not None:
            score += torch.sum(torch.abs(p.grad * p)).item()

    net.zero_grad()
    return score


def synflow_score(net: nn.Module, device: str = "cpu") -> float:
    """
    Synaptic Saliency Flow (SynFlow) Zero-Cost Proxy.
    Data-agnostic metric that measures gradient flow through deep networks
    without requiring ground-truth labels.
    """
    net.zero_grad()
    dummy_input = torch.ones((1, 20, 2), device=device, requires_grad=True)
    dummy_agents = torch.ones((1, 20, 16, 2), device=device, requires_grad=True)
    dummy_lanes = torch.ones((1, 20, 10, 2), device=device, requires_grad=True)

    alphas = (
        torch.ones(5, device=device) / 5.0,
        torch.ones(5, device=device) / 5.0,
        torch.ones(4, device=device) / 4.0,
    )
    output = net(dummy_input, alphas, agents_past=dummy_agents, map_lanes=dummy_lanes)
    if isinstance(output, tuple):
        output = output[0]

    torch.sum(output).backward()

    score = 0.0
    for p in net.parameters():
        if p.grad is not None:
            score += torch.sum(torch.abs(p.grad * p)).item()

    net.zero_grad()
    return score
