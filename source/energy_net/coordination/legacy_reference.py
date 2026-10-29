"""Load the implementable legacy policy selected before amendment experiments."""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .flex_market import FlexMarketConfig
from .flex_tariffs import FlexTariffObserver
from .gpu_tariffs import FixedGpuTariffCatalogue
from .learning import PolicyBundle


@dataclass(frozen=True)
class FixedLegacyMenu:
    """A training-selected constant quote; it never inspects today's outcomes."""
    menu_index: int
    n_actions: int = 77

    def __post_init__(self):
        if int(self.menu_index) != self.menu_index or not 0 <= self.menu_index < self.n_actions:
            raise ValueError("A fixed legacy quote needs a valid catalogue index")

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        shape = np.asarray(observation).shape
        action = (np.full(shape[0], self.menu_index, dtype=np.int64)
                  if len(shape) == 3 else np.asarray(self.menu_index, dtype=np.int64))
        return action, state


def load_legacy_bundle(run_dir):
    """Load one frozen old policy and its three frozen user models.

    ``selection.json`` records the winner chosen using training/validation
    cash returns. Test days do not enter this loader or the selection rule.
    ``legacy_config`` and ``legacy_selection`` are attached to the returned
    bundle so callers can construct the matching common-cap old market.
    """
    run_dir = Path(run_dir)
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    config_values = dict(selection["config"])
    # JSON serializes tuple-valued physical parameters as lists. Restore their
    # immutable schema before recreating the frozen market configuration.
    for field in fields(FlexMarketConfig):
        if isinstance(field.default, tuple) and isinstance(config_values.get(field.name), list):
            config_values[field.name] = tuple(config_values[field.name])
    config = FlexMarketConfig(**config_values)
    if config.sharing_enabled or config.carbon_eligibility or config.internal_carbon_value_yuan_per_t != 0:
        raise ValueError("The outside reference must use legacy cash settlement and the common physical cap")
    catalogue = FixedGpuTariffCatalogue(config)
    users = {user: PPO.load(run_dir / "selected" / f"user_{user}.zip", device="cpu")
             for user in range(config.n_users)}
    kind = selection["policy_kind"]
    if kind == "ppo":
        platform = PPO.load(run_dir / selection.get("platform_model", "selected/platform.zip"), device="cpu")
        observer = FlexTariffObserver(catalogue)
    elif kind == "constant":
        platform = FixedLegacyMenu(int(selection["menu_index"]), catalogue.n_actions)
        observer = None
    else:
        raise ValueError(f"Unknown selected legacy policy kind: {kind}")
    for model in (platform, *users.values()):
        policy = getattr(model, "policy", None)
        if policy is not None:
            policy.set_training_mode(False)
            for parameter in policy.parameters():
                parameter.requires_grad_(False)
    bundle = PolicyBundle(platform=platform, users=users, platform_codec=catalogue,
                          platform_observer=observer, training=False)
    bundle.legacy_config = config
    bundle.legacy_selection = selection
    return bundle
