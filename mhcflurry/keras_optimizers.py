# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch optimizers implementing the historical Keras update equations."""

import math

import torch


class KerasRMSprop(torch.optim.Optimizer):
    """RMSprop with Keras' ``sqrt(velocity + epsilon)`` denominator."""

    def __init__(self, params, lr=0.001, rho=0.9, epsilon=1e-7):
        if lr < 0.0:
            raise ValueError("learning rate must be non-negative")
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must be in [0, 1)")
        if epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")
        super().__init__(
            params,
            dict(lr=lr, rho=rho, epsilon=epsilon),
        )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one Keras-compatible RMSprop update."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("KerasRMSprop does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["velocity"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )
                velocity = state["velocity"]
                velocity.mul_(group["rho"]).addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - group["rho"],
                )
                denominator = velocity.add(group["epsilon"]).sqrt_()
                parameter.addcdiv_(gradient, denominator, value=-group["lr"])
        return loss


class KerasAdam(torch.optim.Optimizer):
    """Adam with Keras' non-adaptive epsilon placement."""

    def __init__(
        self,
        params,
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
    ):
        if lr < 0.0:
            raise ValueError("learning rate must be non-negative")
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError("beta_1 must be in [0, 1)")
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError("beta_2 must be in [0, 1)")
        if epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")
        super().__init__(
            params,
            dict(
                lr=lr,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one Keras-compatible Adam update."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("KerasAdam does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )
                    state["velocity"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )

                state["step"] += 1
                momentum = state["momentum"]
                velocity = state["velocity"]
                momentum.lerp_(gradient, 1.0 - beta_1)
                velocity.mul_(beta_2).addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - beta_2,
                )
                alpha = (
                    group["lr"]
                    * math.sqrt(1.0 - beta_2 ** state["step"])
                    / (1.0 - beta_1 ** state["step"])
                )
                denominator = velocity.sqrt().add_(group["epsilon"])
                parameter.addcdiv_(momentum, denominator, value=-alpha)
        return loss
