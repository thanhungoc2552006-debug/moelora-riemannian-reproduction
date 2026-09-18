
import math
import torch


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
        and module.__class__.__name__ == "MoELoRALinear"
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
