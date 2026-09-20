"""Shared mathematical primitives for the synthetic and LLM reproductions.

Keeping these functions in one place prevents the two experiment tracks from
silently drifting apart while leaving their model wrappers independent.
"""
import torch

RSGD = "riemannian"
GRSGD = "moe-riemannian"
MODES = (RSGD, GRSGD)


def weight_expert(gate, expert_output, mode):
    """Apply routing while optionally rescaling only the expert backward path."""
    if mode == RSGD:
        return gate * expert_output
    if mode == GRSGD:
        sqrt_gate_const = torch.sqrt(gate).detach()
        expert_const = expert_output.detach()
        return (
            sqrt_gate_const * expert_output
            + (gate - sqrt_gate_const) * expert_const
        )
    raise ValueError(f"Unknown optimization mode: {mode!r}")


def precondition_lora_pair(A, B, grad_A, grad_B, damping):
    """Return damped Riemannian directions for one or batched LoRA pair.

    Shapes may be [R, D]/[O, R] or batched [E, R, D]/[E, O, R].
    Both directions are computed from the same pre-update A and B.
    """
    if damping <= 0:
        raise ValueError("damping must be positive")

    rank = A.shape[-2]
    eye = torch.eye(rank, device=A.device, dtype=A.dtype)
    gram_B = B.transpose(-1, -2) @ B + damping * eye
    gram_A = A @ A.transpose(-1, -2) + damping * eye

    dA = torch.linalg.solve(gram_B, grad_A)
    dB = torch.linalg.solve(
        gram_A, grad_B.transpose(-1, -2)
    ).transpose(-1, -2)
    return dA, dB
