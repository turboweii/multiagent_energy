"""Binary strategic users retain their learned legacy economic comparison."""
from copy import deepcopy

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from torch import nn
from stable_baselines3 import PPO

from energy_net.coordination.policies import ContractActorCriticPolicy
from energy_net.coordination.posted_offer_policy import (
    BinaryContractActorCriticPolicy,copy_legacy_user_weights,warm_load_binary_user,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observation(bill=-.2,loss=.1):
    values=np.zeros(92,dtype=np.float32)
    rows=values[6:76].reshape(10,7)
    rows[0,:2]=1;rows[1,:2]=1
    rows[1,2]=bill;rows[1,3]=loss
    return values


def legacy_policy(**kwargs):
    return ContractActorCriticPolicy(spaces.Box(-np.inf,np.inf,(92,),dtype=np.float32),
        spaces.Discrete(10),lambda _:3e-4,economic_only=True,participation_constraint=True,**kwargs)


def binary_policy(**kwargs):
    return BinaryContractActorCriticPolicy(spaces.Box(-np.inf,np.inf,(92,),dtype=np.float32),
        spaces.Discrete(2),lambda _:3e-4,**kwargs)


def probabilities(policy,values):
    with torch.no_grad():
        tensor=policy.obs_to_tensor(values)[0]
        return policy.get_distribution(tensor).distribution.probs


@pytest.mark.parametrize('ortho_init',[False,True])
def test_warm_actor_and_critic_are_exact_legacy_tensors_with_zero_residual(ortho_init):
    source,target=legacy_policy(),binary_policy(ortho_init=ortho_init)
    residual_before={name:value.clone() for name,value in target.mlp_extractor.context_residual.state_dict().items()}
    report=copy_legacy_user_weights(target,source)
    assert report['copied_tensor_count']==12 and report['optimizer_state_entries']==0
    for name in report['copied_tensor_names']:
        torch.testing.assert_close(target.state_dict()[name],source.state_dict()[name],rtol=0,atol=0)
        assert target.state_dict()[name].data_ptr()!=source.state_dict()[name].data_ptr()
    for name,value in residual_before.items():
        torch.testing.assert_close(value,target.mlp_extractor.context_residual.state_dict()[name],rtol=0,atol=0)
    values=torch.as_tensor(np.stack([observation(),observation(-.4,.2)]))
    with torch.no_grad():
        torch.testing.assert_close(target.mlp_extractor.context_residual(values),torch.zeros((2,2)))
        torch.testing.assert_close(target.mlp_extractor.forward_actor(values),
                                   source.mlp_extractor.forward_actor(values)[:,:2],rtol=0,atol=0)
        torch.testing.assert_close(target.predict_values(values),source.predict_values(values),rtol=0,atol=0)
        torch.testing.assert_close(probabilities(target,values.numpy()),probabilities(source,values.numpy())[:,:2])


@pytest.mark.parametrize('case',['unavailable','infeasible','private_ir'])
def test_unavailable_infeasible_or_private_negative_offer_is_masked(case):
    policy=binary_policy();values=observation()
    rows=values[6:76].reshape(10,7)
    if case=='unavailable':rows[1,0]=0
    elif case=='infeasible':rows[1,1]=0
    else:rows[1,2:4]=(.2,.1)
    np.testing.assert_array_equal(probabilities(policy,values).numpy(),[[1,0]])


def test_outside_always_available_and_ir_compares_to_the_actual_outside_row():
    policy=binary_policy();values=observation(.2,.1)
    rows=values[6:76].reshape(10,7)
    rows[0,:2]=0;rows[0,2:4]=(.3,.1)
    assert torch.all(probabilities(policy,values)>0)
    np.testing.assert_array_equal(probabilities(policy,np.zeros(92,dtype=np.float32)).numpy(),[[1,0]])


def test_no_fixed_utility_or_argmin_term_is_added_to_the_learned_scores():
    policy=binary_policy()
    with torch.no_grad():
        for parameter in policy.mlp_extractor.option_score.parameters():parameter.zero_()
    # The offer has strictly greater immediate utility, yet zero learned
    # scores give a stochastic 50/50 policy, not a hard economic action.
    np.testing.assert_array_equal(probabilities(policy,observation(-1.,.1)).numpy(),[[.5,.5]])


def test_private_history_can_change_acceptance_without_changing_the_legacy_scorer():
    policy=binary_policy();original=observation();changed=original.copy();changed[76]=1
    torch.testing.assert_close(probabilities(policy,original),probabilities(policy,changed))
    residual=policy.mlp_extractor.context_residual
    linears=[layer for layer in residual if isinstance(layer,nn.Linear)]
    with torch.no_grad():
        for layer in linears:layer.weight.zero_();layer.bias.zero_()
        linears[0].weight[0,76]=1
        linears[1].weight[0,0]=1
        linears[2].weight[1,0]=1
    assert probabilities(policy,changed)[0,1]>probabilities(policy,original)[0,1]


def test_strict_transfer_rejects_shape_mismatch_before_changing_any_weight():
    target=binary_policy();source=legacy_policy(contract_hidden_sizes=(16,16))
    before={name:value.clone() for name,value in target.state_dict().items()}
    optimizer=target.optimizer
    with pytest.raises(ValueError,match='shape differs'):
        copy_legacy_user_weights(target,source)
    for name,value in before.items():torch.testing.assert_close(value,target.state_dict()[name],rtol=0,atol=0)
    assert target.optimizer is optimizer


def test_transfer_does_not_silently_erase_an_already_learned_residual():
    target=binary_policy();source=legacy_policy()
    with torch.no_grad():target.mlp_extractor.context_residual[-1].bias[0]=1
    with pytest.raises(ValueError,match='untouched zero-output'):
        copy_legacy_user_weights(target,source)


class TwoOfferEnv(gym.Env):
    """Rejecting a positive first offer preserves a better subsequent offer."""
    def __init__(self,actions):
        self.action_space=spaces.Discrete(actions)
        self.observation_space=spaces.Box(-np.inf,np.inf,(92,),dtype=np.float32)
        self.stage=0

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.stage=0
        return observation(-1,0),{}

    def step(self,action):
        if self.stage==0 and int(action)==0:
            self.stage=1;values=observation(-2,0);values[76]=1
            return values,0.,False,False,{}
        reward=float((1 if self.stage==0 else 2) if int(action)==1 else 0)
        return np.zeros(92,dtype=np.float32),reward,True,False,{}


def test_real_ppo_warmstart_fresh_optimizer_updates_and_save_load(tmp_path):
    source=PPO(ContractActorCriticPolicy,TwoOfferEnv(10),seed=3,n_steps=8,batch_size=8,n_epochs=2,
        gamma=0.,policy_kwargs=dict(economic_only=True,participation_constraint=True),device='cpu')
    source.learn(16)
    source_before={name:value.clone() for name,value in source.policy.state_dict().items()}
    source_optimizer=deepcopy(source.policy.optimizer.state_dict())
    assert source_optimizer['state']
    # The old bundle may be frozen; copied parameters must remain trainable.
    for parameter in source.policy.parameters():parameter.requires_grad_(False)
    target=warm_load_binary_user(source,TwoOfferEnv(2),seed=0,
        policy_kwargs=dict(context_hidden_sizes=(24,12)))
    assert target.gamma==target.rollout_buffer.gamma==target.gae_lambda==1
    assert target.num_timesteps==target._n_updates==0
    assert not target.policy.optimizer.state and target.warm_start_metadata['optimizer_state_entries']==0
    assert all(parameter.requires_grad for parameter in target.policy.parameters())
    assert target.policy.context_hidden_sizes==(24,12)
    optimizer_parameters={id(parameter) for group in target.policy.optimizer.param_groups for parameter in group['params']}
    assert optimizer_parameters=={id(parameter) for parameter in target.policy.parameters()}
    for name in target.warm_start_metadata['copied_tensor_names']:
        torch.testing.assert_close(target.policy.state_dict()[name],source_before[name],rtol=0,atol=0)
    initial_score=[parameter.clone() for parameter in target.policy.mlp_extractor.option_score.parameters()]
    target.learn(64)
    assert torch.count_nonzero(target.policy.mlp_extractor.context_residual[-1].weight)>0
    assert any(not torch.equal(a,b) for a,b in zip(initial_score,target.policy.mlp_extractor.option_score.parameters()))
    for name,value in source_before.items():torch.testing.assert_close(value,source.policy.state_dict()[name],rtol=0,atol=0)
    assert source.policy.optimizer.state_dict()['param_groups']==source_optimizer['param_groups']
    values=np.stack([observation(),observation(-.4,.1)])
    expected=probabilities(target.policy,values)
    target.save(tmp_path/'binary')
    restored=PPO.load(tmp_path/'binary',device='cpu')
    assert isinstance(restored.policy,BinaryContractActorCriticPolicy)
    assert restored.policy.context_hidden_sizes==(24,12)
    assert restored.warm_start_metadata==target.warm_start_metadata
    torch.testing.assert_close(probabilities(restored.policy,values),expected,rtol=0,atol=0)
    target.policy.save(tmp_path/'binary-policy')
    policy=BinaryContractActorCriticPolicy.load(tmp_path/'binary-policy',device='cpu')
    assert policy.context_hidden_sizes==(24,12)
    torch.testing.assert_close(probabilities(policy,values),expected,rtol=0,atol=0)


def test_only_economic_legacy_and_binary_target_architectures_are_accepted():
    ordinary=ContractActorCriticPolicy(spaces.Box(-np.inf,np.inf,(92,),dtype=np.float32),
        spaces.Discrete(10),lambda _:3e-4,economic_only=False)
    with pytest.raises(TypeError,match='economic-only'):copy_legacy_user_weights(binary_policy(),ordinary)
    with pytest.raises(ValueError,match='private IR'):binary_policy(participation_constraint=False)
    with pytest.raises(ValueError,match='Discrete'):
        BinaryContractActorCriticPolicy(spaces.Box(-np.inf,np.inf,(92,),dtype=np.float32),
            spaces.Discrete(10),lambda _:3e-4)
