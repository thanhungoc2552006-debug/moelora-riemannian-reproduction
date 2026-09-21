
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELoRALinear(nn.Module):
    """
    Frozen Linear layer + token-level Mixture of LoRA Experts.

    Forward:
        y = Wx + sum_i g_i B_i A_i x

    mode="riemannian":
        normal gate factor g in backward

    mode="moe-riemannian":
        same forward value, but expert backward factor becomes sqrt(g)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 4,
        num_experts: int = 20,
        top_k: int = 10,
        alpha: float = 8.0,
        mode: str = "riemannian",
    ):
        super().__init__()

        assert top_k <= num_experts
        assert mode in {"riemannian", "moe-riemannian"}

        self.base_layer = base_layer
        self.rank = rank
        self.num_experts = num_experts
        self.top_k = top_k
        self.scaling = alpha / rank
        self.mode = mode

        # Freeze pretrained Linear
        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_dim = base_layer.in_features
        out_dim = base_layer.out_features
        dtype = torch.float32
        device = base_layer.weight.device

        self.lora_A = nn.ModuleList()
        self.lora_B = nn.ModuleList()

        for _ in range(num_experts):
            A = nn.Linear(in_dim, rank, bias=False, device=device, dtype=dtype)
            B = nn.Linear(rank, out_dim, bias=False, device=device, dtype=dtype)

            nn.init.kaiming_uniform_(A.weight, a=5 ** 0.5)
            nn.init.zeros_(B.weight)

            self.lora_A.append(A)
            self.lora_B.append(B)

        # Token-level router
        self.gate = nn.Linear(
            in_dim,
            num_experts,
            bias=False,
            device=device,
            dtype=dtype,
        )

        # Keep for later diagnostics
        self.last_gate = None

    def forward(self, x):
        result = self.base_layer(x)

        # MoE-LoRA + router compute in FP32
        x_lora = x.to(self.lora_A[0].weight.dtype)

        # [B, S, E]
        g = F.softmax(self.gate(x_lora), dim=-1)

        # Top-K routing
        _, ids = torch.topk(g, self.top_k, dim=-1)

        mask = F.one_hot(
            ids,
            num_classes=self.num_experts
        ).sum(dim=-2)

        g = g * mask
        g = g / g.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        self.last_gate = g.detach()

        for i in range(self.num_experts):
            expert_output = self.lora_B[i](
                self.lora_A[i](x_lora)
            ) * self.scaling

            gate_i = g[..., i].unsqueeze(-1)

            if self.mode == "moe-riemannian":
                sqrt_g = torch.sqrt(gate_i).detach()
                expert_const = expert_output.detach()

                weighted = (
                    sqrt_g * expert_output
                    + (gate_i - sqrt_g) * expert_const
                )
            else:
                weighted = gate_i * expert_output

            # Llama backbone remains BF16
            result = result + weighted.to(result.dtype)

        return result


def inject_moe_lora(
    model,
    rank=4,
    num_experts=20,
    top_k=10,
    alpha=8.0,
    mode="riemannian",
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
):
    """
    Replace target Linear modules by MoELoRALinear.
    """

    # Freeze whole pretrained model first
    for p in model.parameters():
        p.requires_grad = False

    replaced = []

    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):

            if (
                child_name in target_modules
                and isinstance(child, nn.Linear)
            ):
                new_layer = MoELoRALinear(
                    child,
                    rank=rank,
                    num_experts=num_experts,
                    top_k=top_k,
                    alpha=alpha,
                    mode=mode,
                )

                setattr(module, child_name, new_layer)

                replaced.append(
                    f"{module_name}.{child_name}"
                )

    return replaced
