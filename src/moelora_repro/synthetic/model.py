"""MoE-LoRA with ordinary forward values and two backward rules."""
import math

import torch
from torch import nn


MODES = ("riemannian", "moe-riemannian")


def weight_expert(g, expert_output, mode):
    if mode == "riemannian":
        # Backward contributes g; the forward mixture contributes another g to
        # the induced output update (fixed gate, first order): the g^2 issue.
        return g * expert_output
    if mode == "moe-riemannian":
        # detach changes only the expert Jacobian: d weighted/d expert = sqrt(g).
        # The gate derivative remains expert_output. This is a surrogate
        # backward, not the mathematical derivative of the unchanged forward.
        sqrt_g_const = torch.sqrt(g).detach()
        expert_const = expert_output.detach()
        return sqrt_g_const * expert_output + (g - sqrt_g_const) * expert_const
    raise ValueError(f"Unknown mode: {mode}")


class MoELoRA(nn.Module):
    def __init__(self, hidden_dim=32, output_dim=8, num_experts=8, top_k=2,
                 rank=4, mode="riemannian", temperature=1.0):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("Require 1 <= top_k <= num_experts")
        if min(hidden_dim, output_dim, rank) < 1 or temperature <= 0:
            raise ValueError("Dimensions, rank and temperature must be positive")
        if mode not in MODES:
            raise ValueError(mode)
        self.top_k, self.mode, self.temperature = top_k, mode, temperature
        self.base = nn.Linear(hidden_dim, output_dim, bias=False)
        self.base.requires_grad_(False)
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.1 / math.sqrt(hidden_dim))
        self.A = nn.Parameter(torch.randn(num_experts, rank, hidden_dim)
                              / math.sqrt(hidden_dim))
        # Standard zero-B initialization: the adapter initially changes nothing.
        self.B = nn.Parameter(torch.zeros(num_experts, output_dim, rank))

    def expert_outputs(self, x):
        """Return every raw E_i(x)=B_i A_i x, before gates are applied."""
        latent = torch.einsum("...d,erd->...er", x, self.A)
        return torch.einsum("...er,eor->...eo", latent, self.B)

    def forward(self, x):
        # Routing per sample/token: full softmax, hard Top-K, renormalization.
        # Softmax on selected logits is algebraically the renormalized gate,
        # but stays stable at low temperatures and makes Top-1 exactly one.
        logits = self.router(x) / self.temperature
        probabilities = logits.softmax(dim=-1)
        ids = probabilities.topk(self.top_k, dim=-1).indices
        selected = logits.gather(-1, ids).softmax(dim=-1)
        gates = torch.zeros_like(probabilities).scatter(-1, ids, selected)
        mask = torch.zeros_like(probabilities, dtype=torch.bool).scatter(-1, ids, True)
        # LoRA equation E_i(x) = B_i A_i x, y = Wx + sum_i g_i E_i(x).
        # Compute all tiny experts for readability; unselected contributions
        # and their gradients are zero. This is not a sparse-compute benchmark.
        experts = self.expert_outputs(x)
        output = self.base(x) + weight_expert(gates.unsqueeze(-1), experts, self.mode).sum(-2)
        return output, {"gates": gates, "probabilities": probabilities, "mask": mask}
