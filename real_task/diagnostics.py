
import math
import torch

from moelora_core import precondition_lora_pair
from real_task.moe_lora import MoELoRALinear


EPS = 1e-12


def _pairwise_ba_cosine(module):
    """
    Mean cosine similarity between effective LoRA updates:
        W_i = B_i A_i

    Computed without explicitly constructing the huge B_i A_i matrix.
    """

    sims = []

    E = module.num_experts

    for i in range(E):
        Ai = module.lora_A[i].weight.float()
        Bi = module.lora_B[i].weight.float()

        norm_i_sq = torch.sum(
            (Bi.T @ Bi) * (Ai @ Ai.T)
        )

        for j in range(i + 1, E):
            Aj = module.lora_A[j].weight.float()
            Bj = module.lora_B[j].weight.float()

            norm_j_sq = torch.sum(
                (Bj.T @ Bj) * (Aj @ Aj.T)
            )

            inner = torch.sum(
                (Bi.T @ Bj) * (Ai @ Aj.T)
            )

            denom = torch.sqrt(
                norm_i_sq.clamp_min(EPS)
                * norm_j_sq.clamp_min(EPS)
            )

            sims.append((inner / denom).item())

    return sum(sims) / len(sims)


def _output_cosine(module, x):
    """
    Mean cosine similarity between expert outputs:
        E_i(x) = B_i A_i x

    Also avoids materializing all expert outputs simultaneously.
    """

    x = x.float().reshape(-1, x.shape[-1])

    zs = [
        module.lora_A[i](x)
        for i in range(module.num_experts)
    ]

    sims = []

    for i in range(module.num_experts):
        zi = zs[i]
        Bi = module.lora_B[i].weight.float()

        norm_i_sq = torch.sum(
            (Bi.T @ Bi) * (zi.T @ zi)
        )

        for j in range(i + 1, module.num_experts):
            zj = zs[j]
            Bj = module.lora_B[j].weight.float()

            norm_j_sq = torch.sum(
                (Bj.T @ Bj) * (zj.T @ zj)
            )

            inner = torch.sum(
                (Bi.T @ Bj) * (zi.T @ zj)
            )

            denom = torch.sqrt(
                norm_i_sq.clamp_min(EPS)
                * norm_j_sq.clamp_min(EPS)
            )

            sims.append((inner / denom).item())

    return sum(sims) / len(sims)


@torch.no_grad()
def compute_diagnostics(model, batch):
    """
    Diagnostics are measured on all q_proj MoE-LoRA layers.

    Returns:
        gate_entropy
        normalized_gate_entropy
        ba_cosine
        output_cosine
    """

    was_training = model.training
    model.eval()

    q_layers = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("q_proj")
        and isinstance(module, MoELoRALinear)
    ]

    output_sims = []
    hooks = []

    # Compute output similarity from each q_proj input.
    def make_hook(module):
        def hook(mod, inputs):
            x = inputs[0].detach()
            output_sims.append(
                _output_cosine(module, x)
            )
        return hook

    for _, module in q_layers:
        hooks.append(
            module.register_forward_pre_hook(
                make_hook(module)
            )
        )

    device = next(model.parameters()).device

    batch = {
        k: v.to(device)
        for k, v in batch.items()
    }

    # One fixed validation forward.
    model(**batch)

    for h in hooks:
        h.remove()

    entropies = []
    normalized_entropies = []
    ba_sims = []

    for _, module in q_layers:

        g = module.last_gate.float()

        entropy = -(
            g * torch.log(g.clamp_min(EPS))
        ).sum(dim=-1).mean()

        entropies.append(entropy.item())

        if module.top_k > 1:
            normalized_entropies.append(
                entropy.item() / math.log(module.top_k)
            )
        else:
            normalized_entropies.append(0.0)

        ba_sims.append(
            _pairwise_ba_cosine(module)
        )

    if was_training:
        model.train()

    return {
        "gate_entropy": sum(entropies) / len(entropies),
        "gate_entropy_norm": (
            sum(normalized_entropies)
            / len(normalized_entropies)
        ),
        "ba_cosine": sum(ba_sims) / len(ba_sims),
        "output_cosine": (
            sum(output_sims) / len(output_sims)
        ),
    }


@torch.no_grad()
def _pairwise_update_cosine(module, reg=1e-6):
    """
    Mean cosine similarity between Riemannian update directions
    of effective LoRA matrices W_i = B_i A_i.

    First-order:
        dW_i = dB_i A_i + B_i dA_i

    Must be computed after backward() and before optimizer.step().
    """

    factors = []
    eps = 1e-12

    for A_layer, B_layer in zip(module.lora_A, module.lora_B):

        if A_layer.weight.grad is None or B_layer.weight.grad is None:
            continue

        A = A_layer.weight.detach().float()
        B = B_layer.weight.detach().float()

        grad_A = A_layer.weight.grad.detach().float()
        grad_B = B_layer.weight.grad.detach().float()

        # Same shared Riemannian preconditioning as the optimizer.
        dA, dB = precondition_lora_pair(
            A,
            B,
            grad_A,
            grad_B,
            reg,
        )

        # dW = dB A + B dA
        #    = [dB, B] [A; dA]
        U = torch.cat([dB, B], dim=1)
        V = torch.cat([A, dA], dim=0)

        factors.append((U, V))

    if len(factors) < 2:
        return float("nan")

    norms = []

    for U, V in factors:

        norm_sq = torch.sum(
            (U.T @ U) * (V @ V.T).T
        )

        norms.append(
            torch.sqrt(
                norm_sq.clamp_min(eps)
            )
        )

    similarities = []

    for i in range(len(factors)):

        Ui, Vi = factors[i]

        for j in range(i + 1, len(factors)):

            Uj, Vj = factors[j]

            inner = torch.sum(
                (Ui.T @ Uj)
                * (Vi @ Vj.T)
            )

            denom = (
                norms[i] * norms[j]
            ).clamp_min(eps)

            similarities.append(
                (inner / denom).item()
            )

    return (
        sum(similarities)
        / len(similarities)
    )


@torch.no_grad()
def compute_update_cosine(model, reg=1e-6):
    """
    Average expert-update cosine across q_proj MoE-LoRA layers.

    Call:
        loss.backward()
        compute_update_cosine(...)
        optimizer.step()
    """

    values = []

    for name, module in model.named_modules():

        if (
            name.endswith("q_proj")
            and isinstance(module, MoELoRALinear)
        ):

            value = _pairwise_update_cosine(
                module,
                reg=reg,
            )

            if not math.isnan(value):
                values.append(value)

    if not values:
        return float("nan")

    return sum(values) / len(values)
