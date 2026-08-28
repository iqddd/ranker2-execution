from __future__ import annotations

import unittest

import torch
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

from ranker.prodigy_guard import (
    neutral_schedulefree_evaluation,
    neutral_schedulefree_multi_evaluation,
)


class NeutralScheduleFreeEvaluationTests(unittest.TestCase):
    def _optimizer(self) -> tuple[torch.nn.Module, ProdigyPlusScheduleFree]:
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 5),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(5, 1),
        )
        groups = [
            {"params": list(model[0].parameters())},
            {"params": list(model[2].parameters())},
        ]
        optimizer = ProdigyPlusScheduleFree(groups, lr=1.0)
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(3, 4)).square().mean().backward()
        optimizer.step()
        return model, optimizer

    def test_restores_parameters_rng_modes_and_optimizer_mode_exactly(self) -> None:
        torch.manual_seed(1234)
        model, optimizer = self._optimizer()
        parameters = list(model.parameters())
        expected_parameters = [parameter.detach().clone() for parameter in parameters]
        expected_rng = torch.random.get_rng_state().clone()
        expected_modes = [module.training for module in model.modules()]

        with neutral_schedulefree_evaluation(optimizer, parameters, model=model):
            self.assertTrue(all(not group["train_mode"] for group in optimizer.param_groups))
            model.eval()
            torch.rand(20)

        self.assertTrue(
            all(torch.equal(parameter, expected) for parameter, expected in zip(parameters, expected_parameters))
        )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
        self.assertEqual([module.training for module in model.modules()], expected_modes)
        self.assertTrue(all(group["train_mode"] for group in optimizer.param_groups))

    def test_restores_state_when_validation_raises(self) -> None:
        torch.manual_seed(4321)
        model, optimizer = self._optimizer()
        parameters = list(model.parameters())
        expected_parameters = [parameter.detach().clone() for parameter in parameters]

        with (
            self.assertRaisesRegex(RuntimeError, "validation failure"),
            neutral_schedulefree_evaluation(optimizer, parameters, model=model),
        ):
            model.eval()
            torch.rand(3)
            raise RuntimeError("validation failure")

        self.assertTrue(
            all(torch.equal(parameter, expected) for parameter, expected in zip(parameters, expected_parameters))
        )
        self.assertTrue(all(group["train_mode"] for group in optimizer.param_groups))

    def test_multi_optimizer_restore_is_exact(self) -> None:
        left = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Dropout(0.25))
        right = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Dropout(0.25))
        left_optimizer = ProdigyPlusScheduleFree(
            [{"params": list(left.parameters())}], lr=1.0
        )
        right_optimizer = ProdigyPlusScheduleFree(
            [{"params": list(right.parameters())}], lr=1.0
        )
        for model, optimizer in (
            (left, left_optimizer),
            (right, right_optimizer),
        ):
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(3, 4)).square().mean().backward()
            optimizer.step()
        optimizers = [left_optimizer, right_optimizer]
        groups = [list(left.parameters()), list(right.parameters())]
        before = [parameter.detach().clone() for group in groups for parameter in group]

        with neutral_schedulefree_multi_evaluation(
            optimizers, groups, model=torch.nn.ModuleList([left, right])
        ):
            self.assertTrue(
                all(
                    not optimizer.param_groups[0]["train_mode"]
                    for optimizer in optimizers
                )
            )

        after = [parameter.detach() for group in groups for parameter in group]
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(before, after, strict=True))
        )
        self.assertTrue(
            all(optimizer.param_groups[0]["train_mode"] for optimizer in optimizers)
        )


if __name__ == "__main__":
    unittest.main()
