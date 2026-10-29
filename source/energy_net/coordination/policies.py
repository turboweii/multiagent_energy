"""PPO contract policies with a shared learned scorer for execution options.

Each user's model has its own parameters. Within a model, the same network
scores every contract, so changing a contract's price changes the inputs used
to score that contract rather than relying on a separate action-index head.
Costs remain observations: no argmin rule or fixed economic score is used.
"""
from __future__ import annotations

from gymnasium import spaces
from torch import nn
import torch
from stable_baselines3.common.policies import ActorCriticPolicy


def _mlp(input_size, hidden_sizes, output_size=None):
    layers = []
    previous = input_size
    for size in hidden_sizes:
        layers.extend([nn.Linear(previous, size), nn.Tanh()])
        previous = size
    if output_size is not None:
        layers.append(nn.Linear(previous, output_size))
    return nn.Sequential(*layers)


class ContractExtractor(nn.Module):
    """Ten equivariant actor scores and a separate day-context critic latent."""

    def __init__(self, features_dim=92, contract_hidden_sizes=(32, 32),
                 value_hidden_sizes=(64, 32), economic_only=False,
                 participation_constraint=False):
        super().__init__()
        if features_dim != 92:
            raise ValueError("Contract policies expect the 92-feature task observation.")
        if not value_hidden_sizes:
            raise ValueError("Provide at least one critic hidden layer.")
        self.latent_dim_pi = 10
        self.latent_dim_vf = value_hidden_sizes[-1]
        self.economic_only = bool(economic_only)
        self.participation_constraint = bool(participation_constraint)
        # 7 option features, bill+own loss, 6 private task features, remaining
        # task count. The last two inputs are shared across all ten contracts.
        self.option_score = _mlp(3 if self.economic_only else 15, contract_hidden_sizes, 1)
        self.value_encoder = _mlp(features_dim, value_hidden_sizes)

    def forward_actor(self, features):
        options = features[:, 6:76].reshape(-1, 10, 7)
        generalized_cost = options[:, :, 2:3] + options[:, :, 3:4]
        if self.economic_only:
            actor_input = torch.cat([options[:, :, 2:4], generalized_cost], dim=-1)
        else:
            context = torch.cat([features[:, :6], features[:, -1:]], dim=-1)
            context = context[:, None, :].expand(-1, 10, -1)
            actor_input = torch.cat([options, generalized_cost, context], dim=-1)
        scores = self.option_score(actor_input).squeeze(-1)
        available = (options[:, :, 0] > 0.5) & (options[:, :, 1] > 0.5)
        if self.participation_constraint:
            # A private reservation utility: the owner can reject a contract
            # whose known bill + own inconvenience exceeds its original one.
            # This retains all acceptable alternatives; it does not pick argmin.
            costs = generalized_cost.squeeze(-1)
            available = available & (costs <= costs[:, :1])
        # Empty terminal observations may be passed through a policy by generic
        # tooling. The market itself always retains an available option 0.
        fallback = torch.zeros_like(available)
        fallback[:, 0] = True
        available = torch.where(available.any(dim=-1, keepdim=True), available, fallback)
        return scores.masked_fill(~available, torch.finfo(scores.dtype).min)

    def forward_critic(self, features):
        return self.value_encoder(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class ContractActorCriticPolicy(ActorCriticPolicy):
    """Standard SB3 PPO policy using contract features instead of action IDs.

    Use ``PPO(ContractActorCriticPolicy, env, ...)``. Optional policy kwargs are
    ``contract_hidden_sizes``, ``value_hidden_sizes`` and ``economic_only``.
    Economic-only actors use bill, own loss and their sum, while their critic
    still sees the complete observation. The default retains the full-feature
    actor for compatibility with previously saved policies. Standard PPO
    save/load preserves these settings and every learned parameter.
    """

    def __init__(self, *args, contract_hidden_sizes=(32, 32),
                 value_hidden_sizes=(64, 32), economic_only=False,
                 participation_constraint=False, **kwargs):
        self.contract_hidden_sizes = tuple(contract_hidden_sizes)
        self.value_hidden_sizes = tuple(value_hidden_sizes)
        self.economic_only = bool(economic_only)
        self.participation_constraint = bool(participation_constraint)
        kwargs.setdefault("ortho_init", False)
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_space, spaces.Discrete) or self.action_space.n != 10:
            raise ValueError("Contract policies require Discrete(10) actions.")

    def _build_mlp_extractor(self):
        self.mlp_extractor = ContractExtractor(
            self.features_dim, self.contract_hidden_sizes, self.value_hidden_sizes,
            economic_only=self.economic_only,
            participation_constraint=self.participation_constraint,
        )

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # A dense 10-to-10 head would reintroduce action-index parameters and
        # destroy the actor's option-permutation property. Scores are logits.
        self.action_net = nn.Identity()
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self):
        values = super()._get_constructor_parameters()
        values.update(contract_hidden_sizes=self.contract_hidden_sizes,
                      value_hidden_sizes=self.value_hidden_sizes,
                      economic_only=self.economic_only,
                      participation_constraint=self.participation_constraint)
        return values
