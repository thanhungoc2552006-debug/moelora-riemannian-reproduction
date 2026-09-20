"""Damped per-expert Riemannian SGD; no router parameters here."""
import torch

from moelora_core import precondition_lora_pair


def pair_norm(a, b):
    return (a.square().sum(dim=(-2, -1)) + b.square().sum(dim=(-2, -1))).sqrt()


def precondition(A, B, grad_A, grad_B, damping):
    """Backward-compatible alias for the shared Riemannian primitive."""
    return precondition_lora_pair(A, B, grad_A, grad_B, damping)


class RiemannianSGD(torch.optim.Optimizer):
    def __init__(self, A, B, lr=0.1, damping=0.01):
        if lr <= 0 or damping <= 0:
            raise ValueError("lr and damping must be positive")
        if A.ndim != 3 or B.ndim != 3 or A.shape[0] != B.shape[0] or A.shape[1] != B.shape[2]:
            raise ValueError("Expected A[E,R,D] and B[E,O,R]")
        super().__init__([{"params": [A, B]}], dict(lr=lr, damping=damping))
        self.last_stats = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            A, B = group["params"]
            if A.grad is None or B.grad is None:
                raise RuntimeError("Both expert gradients must exist before step()")
            # Both directions use the SAME pre-update A and B.
            dA, dB = precondition(A, B, A.grad, B.grad, group["damping"])
            if not (torch.isfinite(dA).all() and torch.isfinite(dB).all()):
                raise FloatingPointError("Nonfinite preconditioned gradient")
            self.last_stats = {"raw": pair_norm(A.grad, B.grad).detach(),
                               "preconditioned": pair_norm(dA, dB).detach()}
            A.add_(dA, alpha=-group["lr"])
            B.add_(dB, alpha=-group["lr"])
        return loss
