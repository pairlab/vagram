# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import pickle
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

import mbrl.util.math

from .model import Ensemble
from .util import EnsembleLinearLayer, truncated_normal_init


class VAMLMLP(Ensemble):
    """Implements an ensemble of multi-layer perceptrons each modeling a Gaussian distribution.

    This model corresponds to a Probabilistic Ensemble in the Chua et al.,
    NeurIPS 2018 paper (PETS) https://arxiv.org/pdf/1805.12114.pdf

    It predicts per output mean and log variance, and its weights are updated using a Gaussian
    negative log likelihood loss. The log variance is bounded between learned ``min_log_var``
    and ``max_log_var`` parameters, trained as explained in Appendix A.1 of the paper.

    This class can also be used to build an ensemble of GaussianMLP models, by setting
    ``ensemble_size > 1`` in the constructor. Then, a single forward pass can be used to evaluate
    multiple independent MLPs at the same time. When this mode is active, the constructor will
    set ``self.num_members = ensemble_size``.

    For the ensemble variant, uncertainty propagation methods are available that can be used
    to aggregate the outputs of the different models in the ensemble.
    Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

    The default value of ``None`` indicates that no uncertainty propagation, and the forward
    method returns all outputs of all models.

    Args:
        in_size (int): size of model input.
        out_size (int): size of model output.
        device (str or torch.device): the device to use for the model.
        num_layers (int): the number of layers in the model
                          (e.g., if ``num_layers == 3``, then model graph looks like
                          input -h1-> -h2-> -l3-> output).
        ensemble_size (int): the number of members in the ensemble. Defaults to 1.
        hid_size (int): the size of the hidden layers (e.g., size of h1 and h2 in the graph above).
        use_silu (bool): if ``True``, hidden layers will use SiLU activations, otherwise
                         ReLU activations will be used. Defaults to ``False``.
        deterministic (bool): if ``True``, the model will be trained using MSE loss and no
            logvar prediction will be done. Defaults to ``False``.
        propagation_method (str, optional): the uncertainty propagation method to use (see
            above). Defaults to ``None``            # quantile clipping
            norms = torch.sqrt(torch.sum(g ** 2, -1))
            quantile_bound = np.quantile(norms.detach().cpu().numpy(), 0.95)
            # absolute bound on gradients
            norms = norms.unsqueeze(1)
            if self.bound_clipping:
                g = torch.where(norms < quantile_bound, g, (quantile_bound/norms) * g)

    """

    # TODO integrate this with the checkpoint in the next version
    _ELITE_FNAME = "elite_models.pkl"

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        use_silu: bool = False,
        propagation_method: Optional[str] = None,
        bound_clipping: bool = False,
        bound_clipping_quantile=0.95,
        use_vaml: bool = False,
        use_scaling: bool = True,
        use_all_vf: bool = True,
        add_mse: bool = False
    ):
        super().__init__(
            ensemble_size, device, propagation_method, deterministic=True
        )

        self.in_size = in_size
        self.out_size = out_size

        activation_cls = nn.SiLU if use_silu else nn.ReLU

        def create_linear_layer(l_in, l_out):
            return EnsembleLinearLayer(ensemble_size, l_in, l_out)

        hidden_layers = [
            nn.Sequential(create_linear_layer(in_size, hid_size), activation_cls())
        ]
        for i in range(num_layers - 1):
            hidden_layers.append(
                nn.Sequential(
                    create_linear_layer(hid_size, hid_size),
                    activation_cls(),
                )
            )
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.mean_and_logvar = create_linear_layer(hid_size, out_size)

        self.apply(truncated_normal_init)
        self.to(self.device)

        self.elite_models: List[int] = None

        self._propagation_indices: torch.Tensor = None

        # VAML settings
        self.bound_clipping = bound_clipping
        self.bound_clipping_quantile = bound_clipping_quantile
        self.use_vaml = use_vaml
        self.use_scaling = use_scaling
        self.add_mse = add_mse
        self.use_all_vf = use_all_vf

    def _maybe_toggle_layers_use_only_elite(self, only_elite: bool):
        if self.elite_models is None:
            return
        if self.num_members > 1 and only_elite:
            for layer in self.hidden_layers:
                # each layer is (linear layer, activation_func)
                layer[0].set_elite(self.elite_models)
                layer[0].toggle_use_only_elite()
            self.mean_and_logvar.set_elite(self.elite_models)
            self.mean_and_logvar.toggle_use_only_elite()

    def _default_forward(
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._maybe_toggle_layers_use_only_elite(only_elite)
        x = self.hidden_layers(x)
        mean_and_logvar = self.mean_and_logvar(x)
        self._maybe_toggle_layers_use_only_elite(only_elite)
        return mean_and_logvar, None

    def _forward_from_indices(
        self, x: torch.Tensor, model_shuffle_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, batch_size, _ = x.shape

        num_models = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        shuffled_x = x[:, model_shuffle_indices, ...].view(
            num_models, batch_size // num_models, -1
        )

        mean, logvar = self._default_forward(shuffled_x, only_elite=True)
        # note that mean and logvar are shuffled
        mean = mean.view(batch_size, -1)
        mean[model_shuffle_indices] = mean.clone()  # invert the shuffle

        if logvar is not None:
            logvar = logvar.view(batch_size, -1)
            logvar[model_shuffle_indices] = logvar.clone()  # invert the shuffle

        return mean, logvar

    def _forward_ensemble(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.propagation_method is None:
            mean, logvar = self._default_forward(x, only_elite=False)
            if self.num_members == 1:
                mean = mean[0]
                logvar = logvar[0] if logvar is not None else None
            return mean, logvar
        assert x.ndim == 2
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if x.shape[0] % model_len != 0:
            raise ValueError(
                f"GaussianMLP ensemble requires batch size to be a multiple of the "
                f"number of models. Current batch size is {x.shape[0]} for "
                f"{model_len} models."
            )
        x = x.unsqueeze(0)
        if self.propagation_method == "random_model":
            # passing generator causes segmentation fault
            # see https://github.com/pytorch/pytorch/issues/44714
            model_indices = torch.randperm(x.shape[1], device=self.device)
            return self._forward_from_indices(x, model_indices)
        if self.propagation_method == "fixed_model":
            return self._forward_from_indices(x, self._propagation_indices)
        if self.propagation_method == "expectation":
            mean, logvar = self._default_forward(x, only_elite=True)
            return mean.mean(dim=0), logvar.mean(dim=0)
        raise ValueError(f"Invalid propagation method {self.propagation_method}.")

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes mean and logvar predictions for the given input.

        When ``self.num_members > 1``, the model supports uncertainty propagation options
        that can be used to aggregate the outputs of the different models in the ensemble.
        Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

        If a set of elite models has been indicated (via :meth:`set_elite()`), then all
        propagation methods will operate with only on the elite set. This has no effect when
        ``propagation is None``, in which case the forward pass will return one output for
        each model.

        Args:
            x (tensor): the input to the model. When ``self.propagation is None``,
                the shape must be ``E x B x Id`` or ``B x Id``, where ``E``, ``B``
                and ``Id`` represent ensemble size, batch size, and input dimension,
                respectively. In this case, each model in the ensemble will get one slice
                from the first dimension (e.g., the i-th ensemble member gets ``x[i]``).

                For other values of ``self.propagation`` (and ``use_propagation=True``),
                the shape must be ``B x Id``.
            rng (torch.Generator, optional): random number generator to use for "random_model"
                propagation.
            use_propagation (bool): if ``False``, the propagation method will be ignored
                and the method will return outputs for all models. Defaults to ``True``.

        Returns:
            (tuple of two tensors): the predicted mean and log variance of the output. If
            ``propagation is not None``, the output will be 2-D (batch size, and output dimension).
            Otherwise, the outputs will have shape ``E x B x Od``, where ``Od`` represents
            output dimension.

        Note:
            For efficiency considerations, the propagation method used by this class is an
            approximate version of that described by Chua et al. In particular, instead of
            sampling models independently for each input in the batch, we ensure that each
            model gets exactly the same number of samples (which are assigned randomly
            with equal probability), resulting in a smaller batch size which we use for the forward
            pass. If this is a concern, consider using ``propagation=None``, and passing
            the output to :func:`mbrl.util.math.propagate`.

        """
        if use_propagation:
            return self._forward_ensemble(x, rng=rng)
        return self._default_forward(x)

    def _mse_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add model dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, _ = self.forward(model_in, use_propagation=False)
        return F.mse_loss(pred_mean, target, reduction="none")

    def _vaml_loss(self, model_in: torch.Tensor, target: torch.Tensor, idx: torch.Tensor, eval: bool = False) -> torch.Tensor:
        # assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, _ = self.forward(model_in, use_propagation=False)
        if target.shape[0] != self.num_members:
            target = target.repeat(self.num_members, 1, 1)
        
        target.requires_grad = True
        self._agent.critic.requires_grad = False
        self._agent.critic_target.requires_grad = False

        vf_pred = self.values(target[..., :-1])
        vaml_loss = 0.
        
        for i, vf in enumerate(vf_pred):
            if eval and torch.all(self.known_eval_gradients[idx]):
                g = self.eval_gradients[idx, i].unsqueeze(0)
            elif torch.all(self.known_gradients[idx]):
                g = self.gradients[idx, i].unsqueeze(0)
            else:
                if i == len(vf) - 1:
                    # annoying hack to prevent memory leak in the graph
                    vf.sum().backward(retain_graph=False)
                else:
                    vf.sum().backward(retain_graph=True)
                g = target.grad.clone().detach().squeeze()[..., :-1]
                if eval:
                    self.eval_gradients[idx, i] = g[0]
                else:
                    self.gradients[idx, i] = g[0]

            # quantile clipping
            if self.bound_clipping:
                norms = torch.sqrt(torch.sum(g ** 2, -1))
                quantile_bound = np.quantile(norms.detach().cpu().numpy(), self.bound_clipping_quantile)
                # absolute bound on gradients
                norms = norms.unsqueeze(-1)
                g = torch.where(norms < quantile_bound, g, (quantile_bound/norms) * g).detach()
            else:
                # hack to force copying the tensor
                g = g.clone().detach()
            if self.use_vaml:
                vaml_loss += (torch.sum(g * (pred_mean[..., :-1] - target[..., :-1]), -1, keepdim=True) ** 2)
            elif self.use_scaling:
                vaml_loss += ((g * (pred_mean[..., :-1] - target[..., :-1])) ** 2)
            else:
                raise NotImplementedError("Have to compute either Taylor VAML or scaled MSE")
            self._agent.critic.zero_grad()
            self._agent.critic_target.zero_grad()
            if target.grad is not None:
                target.grad[:] = 0.

        if eval:
            self.known_gradients[idx] = True        
        else:
            self.known_gradients[idx] = True        

        target.requires_grad = False
        self._agent.critic.requires_grad = True
        self._agent.critic_target.requires_grad = True
        
        vaml_loss /= len(vf_pred)

        # reward component
        vaml_loss += ((pred_mean[..., -1:] - target[..., -1:]) ** 2)
        
        return vaml_loss
    
    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        idx = None,
        eval = False
    ) -> torch.Tensor:
        """Computes Gaussian NLL loss.

        It also includes terms for ``max_logvar`` and ``min_logvar`` with small weights,
        with positive and negative signs, respectively.

        Args:
            model_in (tensor): input tensor. The shape must be ``E x B x Id``, or ``B x Id``
                where ``E``, ``B`` and ``Id`` represent ensemble size, batch size, and input
                dimension, respectively.
            target (tensor): target tensor. The shape must be ``E x B x Id``, or ``B x Od``
                where ``E``, ``B`` and ``Od`` represent ensemble size, batch size, and output
                dimension, respectively.

        Returns:
            (tensor): a loss tensor representing the Gaussian negative log-likelihood of
            the model over the given input/target. If the model is an ensemble, returns
            the average over all models.
        """
        loss = self._vaml_loss(model_in, target, idx=idx, eval=False)
        if self.add_mse:
            loss += self._mse_loss(model_in, target).mean(-1, keepdim=True)
        return loss.mean()

    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None, idx=None
    ) -> torch.Tensor:
        """Computes the squared error for the model over the given input/target.

        When model is not an ensemble, this is equivalent to
        `F.mse_loss(model(model_in, target), reduction="none")`. If the model is ensemble,
        then return is batched over the model dimension.

        Args:
            model_in (tensor): input tensor. The shape must be ``B x Id``, where `B`` and ``Id``
                batch size, and input dimension, respectively.
            target (tensor): target tensor. The shape must be ``B x Od``, where ``B`` and ``Od``
                represent batch size, and output dimension, respectively.

        Returns:
            (tensor): a tensor with the squared error per output dimension, batched over model.
        """
        # target = target.repeat((self.num_members, 1, 1))
        loss = self._vaml_loss(model_in, target, idx, eval=True)
        if self.add_mse:
            loss += self._mse_loss(model_in, target).mean(-1, keepdim=True)
        return loss.detach()


    def reset(  # type: ignore
        self, x: torch.Tensor, rng: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Initializes any internal dependent state when using the model for simulation.

        Initializes model indices for "fixed_model" propagation method
        a bootstrapped ensemble with TSinf propagation).

        Args:
            x (tensor): the input to the model.
            rng (random number generator): a rng to use for sampling the model
                indices.

        Returns:
            (tensor): forwards the same input.
        """
        assert rng is not None
        self._propagation_indices = self._sample_propagation_indices(x.shape[0], rng)
        return x

    def _sample_propagation_indices(
        self, batch_size: int, _rng: torch.Generator
    ) -> torch.Tensor:
        """Returns a random permutation of integers in [0, ``batch_size``)."""
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if batch_size % model_len != 0:
            raise ValueError(
                "To use GaussianMLP's ensemble propagation, the batch size must "
                "be a multiple of the number of models in the ensemble."
            )
        # rng causes segmentation fault, see https://github.com/pytorch/pytorch/issues/44714
        return torch.randperm(batch_size, device=self.device)

    def set_elite(self, elite_indices: Sequence[int]):
        if len(elite_indices) != self.num_members:
            self.elite_models = list(elite_indices)

    def save(self, path: Union[str, pathlib.Path]):
        """Saves the model to the given path."""
        super().save(path)
        path = pathlib.Path(path)
        elite_path = path / self._ELITE_FNAME
        if self.elite_models:
            warnings.warn(
                "Future versions of GaussianMLP will save elite models in the same "
                "checkpoint file as the model weights."
            )
            with open(elite_path, "wb") as f:
                pickle.dump(self.elite_models, f)

    def load(self, path: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        super().load(path)
        path = pathlib.Path(path)
        elite_path = path / self._ELITE_FNAME
        if pathlib.Path.is_file(elite_path):
            warnings.warn(
                "Future versions of GaussianMLP will load elite models from the same "
                "checkpoint file as the model weights."
            )
            with open(elite_path, "rb") as f:
                self.elite_models = pickle.load(f)
        else:
            warnings.warn("No elite model information found in model load directory.")

    def set_agent(self, agent):
        self._agent = agent
        self.gradients[:] = 0.
        self.eval_gradients[:] = 0.
        self.known_gradients[:] = 0.
        self.known_eval_gradients[:] = 0.

    def set_gradient_buffer(self, obs_shape, act_shape, cfg):
        dataset_size = (
            cfg.algorithm.get("dataset_size", None) if "algorithm" in cfg else None
        )
        if not dataset_size:
            dataset_size = cfg.overrides.num_steps

        if self.use_all_vf:
            self.gradients = torch.zeros((dataset_size, 4, obs_shape[0]), device=self.device)
            self.eval_gradients = torch.zeros((dataset_size, 4, obs_shape[0]), device=self.device)
        else:
            self.gradients = torch.zeros((dataset_size, 2, obs_shape[0]), device=self.device)
            self.eval_gradients = torch.zeros((dataset_size, 2, obs_shape[0]), device=self.device)

        self.known_gradients = torch.zeros((dataset_size, 1), dtype=torch.bool, device=self.device)
        self.known_eval_gradients = torch.zeros((dataset_size, 1), dtype=torch.bool, device=self.device)


    def values(self, input):
        self._agent.actor.requires_grad = False
        self._agent.critic.requires_grad = False
        self._agent.critic_target.requires_grad = False
        dist = self._agent.actor(input)
        next_action = dist.mean

        values = self._agent.critic(input, next_action.detach())

        values_target = self._agent.critic_target(input, next_action.detach())
        if self.use_all_vf:
            all_values = torch.stack([*values, *values_target], 0)
        else:
            all_values = torch.stack([*values_target], 0)
        return all_values.squeeze(1)


class ValueWeightedModel(VAMLMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _vaml_loss(self, model_in: torch.Tensor, target: torch.Tensor, idx: torch.Tensor, eval: bool = False) -> torch.Tensor:
        # assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, _ = self.forward(model_in, use_propagation=False)
        if target.shape[0] != self.num_members:
            target = target.repeat(self.num_members, 1, 1)
        
        vf_pred = self.values(target[..., :-1])
        vf_weighing = torch.abs(vf_pred).mean(0) / torch.sum(torch.abs(vf_pred))
        vf_weighing = vf_weighing.detach()
        vaml_loss = 0.
        
        vaml_loss += vf_weighing * ((pred_mean[..., :-1] - target[..., :-1]) ** 2)

        # reward component
        vaml_loss += ((pred_mean[..., -1:] - target[..., -1:]) ** 2)
        
        return vaml_loss
