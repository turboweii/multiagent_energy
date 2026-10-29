"""Learned binary responses to one posted offer and an outside contract.

The old economic option scorer is retained exactly. A zero-output context
residual allows subsequent PPO updates to learn how a current rejection can
affect future offers, using the user's own history in the 92-field state.
No fixed utility score or best-response action replaces the policy.
"""
from __future__ import annotations

from copy import deepcopy

from gymnasium import spaces
from torch import nn
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from .policies import ContractActorCriticPolicy, ContractExtractor, _mlp


class BinaryContractExtractor(ContractExtractor):
    """The legacy economic scorer plus a learnable private-history residual."""

    def __init__(self, features_dim=92, contract_hidden_sizes=(32,32),
                 value_hidden_sizes=(64,32), context_hidden_sizes=(32,16)):
        super().__init__(features_dim, contract_hidden_sizes, value_hidden_sizes,
                         economic_only=True, participation_constraint=True)
        self.latent_dim_pi = 2
        self.context_residual = _mlp(features_dim, context_hidden_sizes, 2)
        self.zero_context_output()

    def zero_context_output(self):
        output = self.context_residual[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward_actor(self, features):
        all_options = features[:,6:76].reshape(-1,10,7)
        all_costs = all_options[:,:,2:3] + all_options[:,:,3:4]
        economic_inputs = torch.cat((all_options[:,:,2:4],all_costs),dim=-1)
        # Retain the old scorer's ten-row numerical layout, then discard
        # unavailable padding. Zero-residual warm logits exactly match the
        # corresponding legacy logits, including the GEMM reduction path.
        scores = self.option_score(economic_inputs).squeeze(-1)[:,:2]
        scores = scores + self.context_residual(features)
        options, generalized_cost = all_options[:,:2], all_costs[:,:2]
        available = (options[:,:,0]>.5) & (options[:,:,1]>.5)
        costs = generalized_cost.squeeze(-1)
        available = available & (costs<=costs[:,:1])
        # The accepted outside contract is always retained, including the
        # generic empty terminal observation passed by SB3 tooling.
        available = available.clone()
        available[:,0] = True
        return scores.masked_fill(~available,torch.finfo(scores.dtype).min)


class BinaryContractActorCriticPolicy(ActorCriticPolicy):
    """Discrete(2) PPO: reject to outside or accept the unique posted offer.

    Observations keep the existing 92-field layout: six private task fields,
    ten seven-field contract rows (only rows zero/one may be used), then the
    user's sixteen history fields. The critic retains the legacy encoder.
    """

    def __init__(self,*args,contract_hidden_sizes=(32,32),value_hidden_sizes=(64,32),
                 context_hidden_sizes=(32,16),economic_only=True,
                 participation_constraint=True,**kwargs):
        if not economic_only or not participation_constraint:
            raise ValueError("Binary posted offers require the economic scorer and private IR mask")
        self.contract_hidden_sizes = tuple(contract_hidden_sizes)
        self.value_hidden_sizes = tuple(value_hidden_sizes)
        self.context_hidden_sizes = tuple(context_hidden_sizes)
        self.economic_only = True
        self.participation_constraint = True
        kwargs.setdefault("ortho_init",False)
        super().__init__(*args,**kwargs)

    def _build_mlp_extractor(self):
        if not isinstance(self.action_space,spaces.Discrete) or self.action_space.n!=2:
            raise ValueError("Binary contract policies require Discrete(2) actions")
        self.mlp_extractor = BinaryContractExtractor(
            self.features_dim,self.contract_hidden_sizes,self.value_hidden_sizes,
            self.context_hidden_sizes)

    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
        # Explicit ortho_init=True must not accidentally overwrite the
        # required zero-output residual during ActorCriticPolicy._build.
        self.mlp_extractor.zero_context_output()
        self.optimizer = self.optimizer_class(
            self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        values = super()._get_constructor_parameters()
        values.update(contract_hidden_sizes=self.contract_hidden_sizes,
                      value_hidden_sizes=self.value_hidden_sizes,
                      context_hidden_sizes=self.context_hidden_sizes,
                      economic_only=True,participation_constraint=True)
        return values


def _policy(model_or_policy):
    return getattr(model_or_policy,"policy",model_or_policy)


def copy_legacy_user_weights(target,source):
    """Copy every named scorer/critic tensor after validating the whole match.

    No mismatched layer is skipped, reshaped or reinitialized. The target
    residual must still have zero output and remains untouched. Optimizer
    parameters retain the target settings but get a newly constructed,
    empty optimizer; old Adam moments are never inherited. Returns a small
    serializable report, also attached as ``warm_start_metadata``.
    """
    target_policy,source_policy = _policy(target),_policy(source)
    if not isinstance(target_policy,BinaryContractActorCriticPolicy):
        raise TypeError("Target must be BinaryContractActorCriticPolicy or its PPO")
    if not isinstance(source_policy,ContractActorCriticPolicy) or not source_policy.economic_only:
        raise TypeError("Source must be a legacy economic-only ContractActorCriticPolicy or its PPO")
    prefixes = ("mlp_extractor.option_score.","mlp_extractor.value_encoder.","value_net.")
    target_state,source_state = target_policy.state_dict(),source_policy.state_dict()
    names = []
    for prefix in prefixes:
        target_names = {name for name in target_state if name.startswith(prefix)}
        source_names = {name for name in source_state if name.startswith(prefix)}
        if not source_names or target_names!=source_names:
            raise ValueError(f"Legacy/target tensor names differ in {prefix}")
        for name in sorted(source_names):
            if target_state[name].shape!=source_state[name].shape:
                raise ValueError(f"Legacy/target tensor shape differs: {name}")
            if target_state[name].dtype!=source_state[name].dtype:
                raise ValueError(f"Legacy/target tensor dtype differs: {name}")
            names.append(name)
    residual_output = target_policy.mlp_extractor.context_residual[-1]
    if (torch.count_nonzero(residual_output.weight).item()
            or torch.count_nonzero(residual_output.bias).item()):
        raise ValueError("Warm loading requires an untouched zero-output context residual")
    fresh_optimizer = target_policy.optimizer_class(
        target_policy.parameters(),**dict(target_policy.optimizer.defaults))
    with torch.no_grad():
        for name in names:
            target_state[name].copy_(source_state[name])
    target_policy.optimizer = fresh_optimizer
    report = dict(
        source_policy=type(source_policy).__name__,target_policy=type(target_policy).__name__,
        source_num_timesteps=getattr(source,"num_timesteps",None),
        source_num_updates=getattr(source,"_n_updates",None),
        copied_tensor_names=names,copied_tensor_count=len(names),
        contract_hidden_sizes=list(target_policy.contract_hidden_sizes),
        value_hidden_sizes=list(target_policy.value_hidden_sizes),
        context_hidden_sizes=list(target_policy.context_hidden_sizes),
        context_residual="unchanged, zero output at warm start",
        optimizer="fresh target optimizer; no legacy state/moments copied",
        optimizer_state_entries=len(fresh_optimizer.state),
    )
    target_policy.warm_start_metadata = deepcopy(report)
    if target is not target_policy:
        target.warm_start_metadata = deepcopy(report)
    return report


def warm_load_binary_user(source_PPO,newenv,seed=0,**overrides):
    """Create a fresh PPO around copied legacy scorer/critic parameters.

    The old PPO remains unchanged. Training settings otherwise follow the
    source, while gamma/GAE default to one for finite strategic offer episodes.
    Callers can supply explicit PPO overrides, including ``policy_kwargs``;
    incompatible shared architectures fail in the strict copy helper.
    Counters and optimizer history start fresh and the transfer is recorded.
    """
    if not isinstance(source_PPO,PPO):
        raise TypeError("warm_load_binary_user expects a source PPO")
    source = source_PPO.policy
    if not isinstance(source,ContractActorCriticPolicy) or not source.economic_only:
        raise TypeError("Warm loading needs an economic-only legacy contract user")
    policy_kwargs = dict(source_PPO.policy_kwargs)
    policy_kwargs.update(contract_hidden_sizes=source.contract_hidden_sizes,
                         value_hidden_sizes=source.value_hidden_sizes,
                         economic_only=True,participation_constraint=True)
    policy_kwargs.update(dict(overrides.pop("policy_kwargs",{})))
    kwargs = dict(learning_rate=source_PPO.learning_rate,n_steps=source_PPO.n_steps,
        batch_size=source_PPO.batch_size,n_epochs=source_PPO.n_epochs,gamma=1.,gae_lambda=1.,
        clip_range=source_PPO.clip_range,clip_range_vf=source_PPO.clip_range_vf,
        normalize_advantage=source_PPO.normalize_advantage,ent_coef=source_PPO.ent_coef,
        vf_coef=source_PPO.vf_coef,max_grad_norm=source_PPO.max_grad_norm,
        use_sde=source_PPO.use_sde,sde_sample_freq=source_PPO.sde_sample_freq,
        target_kl=source_PPO.target_kl,device=source_PPO.device,verbose=source_PPO.verbose,
        policy_kwargs=policy_kwargs)
    kwargs.update(overrides)
    target = PPO(BinaryContractActorCriticPolicy,newenv,seed=seed,**kwargs)
    report = copy_legacy_user_weights(target,source_PPO)
    report.update(algorithm_seed=int(seed),gamma=float(target.gamma),gae_lambda=float(target.gae_lambda),
                  target_num_timesteps=int(target.num_timesteps),target_num_updates=int(target._n_updates))
    target.warm_start_metadata = deepcopy(report)
    target.policy.warm_start_metadata = deepcopy(report)
    return target
