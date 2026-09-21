"""Test local Jacobians, routing invariants, and actual paired SGD updates."""
import copy
import unittest

import torch

from moelora_synthetic.model import MoELoRA, MODES, weight_expert
from moelora_synthetic.optimizer import RiemannianSGD, precondition
from moelora_synthetic.train import (mean_output_pairwise_cosine, mean_vector_pairwise_cosine,
                   similarity_diagnostics)


class MechanismTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_forward_and_exact_expert_and_gate_gradients(self):
        gate_values = torch.tensor([0., 0.01, 0.25, 0.64, 1.], dtype=torch.float64)
        outputs, gradients = [], []
        for mode in MODES:
            g = gate_values.clone().requires_grad_()
            expert = torch.randn(5, dtype=torch.float64, generator=torch.Generator().manual_seed(1)).requires_grad_()
            upstream = torch.arange(1, 6, dtype=torch.float64)
            out = weight_expert(g, expert, mode)
            (out * upstream).sum().backward()
            expected = g.detach() if mode == MODES[0] else g.detach().sqrt()
            torch.testing.assert_close(expert.grad, expected * upstream)
            torch.testing.assert_close(g.grad, expert.detach() * upstream)
            self.assertTrue(torch.isfinite(g.grad).all())
            outputs.append(out.detach())
            gradients.append(expert.grad)
        torch.testing.assert_close(*outputs)
        self.assertFalse(torch.allclose(*gradients))

    def test_factor_gradients_and_router_preserved_at_same_state(self):
        normal = MoELoRA(5, 3, 4, 4, 2).double()
        with torch.no_grad():
            normal.B.normal_()
        scaled = copy.deepcopy(normal)
        scaled.mode = MODES[1]
        # One sample avoids interpreting a ratio of sums as a per-token ratio.
        x = torch.randn(1, 5, dtype=torch.float64)
        outputs = []
        for model in (normal, scaled):
            out, routing = model(x)
            outputs.append(out)
            out.sum().backward()
        torch.testing.assert_close(*outputs)
        g = routing["gates"].detach().flatten()
        for name in ("A", "B"):
            torch.testing.assert_close(getattr(scaled, name).grad,
                                       getattr(normal, name).grad / g.sqrt()[:, None, None])
        torch.testing.assert_close(normal.router.weight.grad, scaled.router.weight.grad)
        self.assertIsNone(normal.base.weight.grad)

    def test_routing_tokens_and_inactive_experts(self):
        for mode in MODES:
            model = MoELoRA(5, 3, 4, 2, 2, mode).double()
            with torch.no_grad():
                model.B.normal_()
            out, info = model(torch.randn(1, 1, 5, dtype=torch.float64))
            self.assertEqual(tuple(out.shape), (1, 1, 3))
            self.assertTrue((info["mask"].sum(-1) == 2).all())
            torch.testing.assert_close(info["gates"].sum(-1), torch.ones(1, 1, dtype=torch.float64))
            reference = info["probabilities"] * info["mask"]
            reference = reference / reference.sum(-1, keepdim=True)
            torch.testing.assert_close(info["gates"], reference)
            out.sum().backward()
            inactive = ~info["mask"].flatten()
            self.assertEqual(model.A.grad[inactive].abs().sum().item(), 0)
            self.assertEqual(model.B.grad[inactive].abs().sum().item(), 0)

    def test_top_one_identical_training_and_zero_router_gradient(self):
        normal = MoELoRA(5, 3, 4, 1, 2).double()
        scaled = copy.deepcopy(normal)
        scaled.mode = MODES[1]
        models = (normal, scaled)
        opts = [RiemannianSGD(m.A, m.B) for m in models]
        routers = [torch.optim.SGD(m.router.parameters(), lr=0.01) for m in models]
        for _ in range(4):
            x, target = torch.randn(8, 5, dtype=torch.float64), torch.randn(8, 3, dtype=torch.float64)
            for model, opt, router in zip(models, opts, routers):
                opt.zero_grad()
                router.zero_grad()
                out, info = model(x)
                torch.testing.assert_close(info["gates"][info["mask"]], torch.ones(8, dtype=torch.float64))
                (out-target).square().mean().backward()
                self.assertEqual(model.router.weight.grad.abs().sum().item(), 0)
                opt.step()
                router.step()
        for a, b in zip(normal.parameters(), scaled.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_solve_and_simultaneous_update_match_reference(self):
        A = torch.nn.Parameter(torch.randn(3, 2, 5, dtype=torch.float64))
        B = torch.nn.Parameter(torch.randn(3, 4, 2, dtype=torch.float64))
        A.grad, B.grad = torch.randn_like(A), torch.randn_like(B)
        old_A, old_B = A.detach().clone(), B.detach().clone()
        eye, damping, lr = torch.eye(2, dtype=torch.float64), 0.07, 0.03
        # Explicit inverse is used ONLY as an independent tiny test oracle.
        expected_A = torch.stack([torch.linalg.inv(b.T @ b + damping*eye) @ da for b,da in zip(B, A.grad)])
        expected_B = torch.stack([db @ torch.linalg.inv(a @ a.T + damping*eye) for a,db in zip(A, B.grad)])
        opt = RiemannianSGD(A, B, lr, damping)
        opt.step()
        torch.testing.assert_close(A, old_A-lr*expected_A)
        torch.testing.assert_close(B, old_B-lr*expected_B)
        torch.testing.assert_close(opt.last_stats["preconditioned"],
                                   (expected_A.square().sum((1,2))+expected_B.square().sum((1,2))).sqrt())

    def test_zero_b_initialization_is_finite(self):
        model = MoELoRA(5, 3, 4, 2, 2)
        model(torch.randn(8, 5))[0].square().mean().backward()
        self.assertEqual(model.A.grad.abs().sum().item(), 0)
        opt = RiemannianSGD(model.A, model.B)
        opt.step()
        self.assertTrue(torch.isfinite(model.A).all() and torch.isfinite(model.B).all())
        self.assertGreater(model.B.abs().sum().item(), 0)

    def test_actual_forward_change_has_g_squared_or_g_three_halves(self):
        # Finite parameter update of the REAL forward, with gates fixed.
        # This distinguishes the surrogate Jacobian from a true derivative.
        g, damping, lr = 0.04, 0.01, 1e-6
        geometry = 0.9**2 / (0.9**2+damping) + 0.7**2 / (0.7**2+damping)
        for mode, power in zip(MODES, (2.0, 1.5)):
            A = torch.nn.Parameter(torch.tensor([[[0.7]]], dtype=torch.float64))
            B = torch.nn.Parameter(torch.tensor([[[0.9]]], dtype=torch.float64))
            old_output = (g * B @ A).item()
            weight_expert(torch.tensor(g, dtype=torch.float64), B @ A, mode).sum().backward()
            RiemannianSGD(A, B, lr, damping).step()
            observed = (old_output - (g * B @ A).item()) / lr
            self.assertAlmostEqual(observed / (g**power * geometry), 1.0, places=5)

    def test_pairwise_similarity_helpers(self):
        vectors = torch.tensor([[1., 0.], [1., 0.], [-1., 0.]])
        mean, pairs = mean_vector_pairwise_cosine(vectors)
        self.assertEqual(pairs, 3)
        self.assertAlmostEqual(mean, -1 / 3)

        outputs = torch.tensor([[[1., 0.], [1., 0.], [1., 0.]],
                                [[0., 1.], [0., 1.], [0., 1.]]])
        mean, values = mean_output_pairwise_cosine(outputs)
        self.assertEqual(values, 6)
        self.assertAlmostEqual(mean, 1.0)

    def test_similarity_diagnostic_does_not_touch_training_grads(self):
        model = MoELoRA(5, 3, 4, 2, 2)
        with torch.no_grad():
            model.B.normal_()
        model.A.grad = torch.full_like(model.A, 7)
        model.B.grad = torch.full_like(model.B, 9)
        result = similarity_diagnostics(model, torch.randn(32, 5),
                                        torch.randn(32, 3))
        self.assertEqual(result["update_valid_pairs"], 6)
        self.assertEqual(result["gradient_valid_pairs"], 6)
        torch.testing.assert_close(model.A.grad, torch.full_like(model.A, 7))
        torch.testing.assert_close(model.B.grad, torch.full_like(model.B, 9))

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            MoELoRA(top_k=9)
        with self.assertRaises(ValueError):
            MoELoRA(temperature=0)
        model = MoELoRA()
        with self.assertRaises(ValueError):
            RiemannianSGD(model.A, model.B, damping=0)


if __name__ == "__main__":
    unittest.main()
