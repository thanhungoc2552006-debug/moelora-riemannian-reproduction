"""Regression tests for primitives shared by synthetic and LLM experiments."""
import unittest

import torch

from moelora_core import GRSGD, RSGD, precondition_lora_pair, weight_expert


class SharedCoreTests(unittest.TestCase):
    def test_gate_rescaling_preserves_forward_and_changes_expert_jacobian(self):
        gate = torch.tensor([0.04, 0.25, 1.0], dtype=torch.float64)
        expert = torch.tensor([0.3, -0.7, 0.2], dtype=torch.float64)

        outputs = []
        gradients = []
        for mode in (RSGD, GRSGD):
            x = expert.clone().requires_grad_()
            y = weight_expert(gate, x, mode)
            outputs.append(y.detach())
            y.sum().backward()
            gradients.append(x.grad.detach())

        torch.testing.assert_close(outputs[0], outputs[1])
        torch.testing.assert_close(gradients[0], gate)
        torch.testing.assert_close(gradients[1], gate.sqrt())

    def test_preconditioner_matches_single_and_batched_calls(self):
        generator = torch.Generator().manual_seed(123)
        A = torch.randn(3, 2, 5, generator=generator, dtype=torch.float64)
        B = torch.randn(3, 4, 2, generator=generator, dtype=torch.float64)
        grad_A = torch.randn(A.shape, generator=generator, dtype=torch.float64)
        grad_B = torch.randn(B.shape, generator=generator, dtype=torch.float64)

        batch_dA, batch_dB = precondition_lora_pair(
            A, B, grad_A, grad_B, damping=1e-3
        )

        for i in range(A.shape[0]):
            dA, dB = precondition_lora_pair(
                A[i], B[i], grad_A[i], grad_B[i], damping=1e-3
            )
            torch.testing.assert_close(batch_dA[i], dA)
            torch.testing.assert_close(batch_dB[i], dB)


if __name__ == "__main__":
    unittest.main()
