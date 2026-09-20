
import torch

from moelora_core import precondition_lora_pair
from real_task.moe_lora import MoELoRALinear


class RiemannianSGD:
    """
    Riemannian SGD for LoRA expert pairs (A, B).

    A: [rank, in_dim]
    B: [out_dim, rank]

    grad_A <- (B^T B + reg I)^(-1) grad_A
    grad_B <- grad_B (A A^T + reg I)^(-1)
    """

    def __init__(self, model, lr=3e-5, reg=1e-6):
        self.lr = lr
        self.reg = reg
        self.pairs = []

        for module in model.modules():
            if isinstance(module, MoELoRALinear):
                for A, B in zip(module.lora_A, module.lora_B):
                    self.pairs.append((A.weight, B.weight))

    @torch.no_grad()
    def step(self):
        for A, B in self.pairs:
            if A.grad is None or B.grad is None:
                continue

            grad_A = A.grad
            grad_B = B.grad

            grad_A_scaled, grad_B_scaled = precondition_lora_pair(
                A,
                B,
                grad_A,
                grad_B,
                self.reg,
            )

            A.add_(grad_A_scaled, alpha=-self.lr)
            B.add_(grad_B_scaled, alpha=-self.lr)

    def zero_grad(self):
        for A, B in self.pairs:
            A.grad = None
            B.grad = None
