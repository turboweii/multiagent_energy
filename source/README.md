# EnergyNet

> A multi-agent reinforcement learning framework for smart grid simulation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

**EnergyNet** is a Python framework for simulating smart power grids with interacting reinforcement learning agents. It provides OpenAI Gym-compatible environments for modeling energy management scenarios, including battery storage optimization, grid operator pricing strategies, and multi-agent coordination.

The framework enables researchers and practitioners to experiment with energy arbitrage, demand response, and grid optimization using state-of-the-art RL algorithms. Each environment simulates realistic grid dynamics with configurable consumption patterns, pricing strategies, and storage constraints.

EnergyNet is designed for flexibility and extensibility, making it suitable for both academic research and practical experimentation in smart grid management.

---

## Key Features

- **Three OpenAI Gym Environments**
  - `PCSEnv`: Battery storage management with price arbitrage
  - `ISOEnv`: Grid operator pricing and dispatch optimization
  - `AlternatingISOEnv`: Multi-agent training with ISO-PCS interaction

- **Realistic Grid Simulation**
  - Configurable consumption patterns (constant, sinusoidal, data-driven)
  - Dynamic pricing strategies
  - Battery storage dynamics with charge/discharge constraints
  - Shortage penalties and demand-supply balancing

- **Multi-Agent Learning**
  - Alternating training framework for ISO and PCS agents
  - Coordinated or adversarial agent objectives
  - Flexible reward structures

- **Integration with Popular RL Libraries**
  - Compatible with Stable-Baselines3, SB3-Contrib
  - Supports PPO, SAC, TD3, and other algorithms
  - VecNormalize support for training stability

---

## Quick Start

### Electricity–carbon–compute coordination study

The added study models posted compute tariffs, three independent task owners,
network-constrained electricity clearing and a daily carbon cap. See
[the research framework](docs/coordination_framework.md) for objectives, units,
agent definitions and the distinction between rule-based comparisons and
complete-information benchmarks.

[The staged training workflow](docs/conditional_contract_training.md) adds
diverse-quote user pretraining, shared contract scoring, a 55-action platform
catalogue and sampled frozen opponents during joint learning.

[Conditional platform PPO](docs/platform_contextual_ppo.md) adds public
candidate-tariff features, centered inputs and shared LayerNorm/SiLU scores,
followed by four-agent joint training and an explicit private participation rule.

```bash
pip install -e '.[coordination,test]'
python scripts/train_coordination.py --seed 0 --cycles 8 --platform-days 32 --user-steps 128 --eval-days 8 --output results/ppo-seed0
python scripts/run_coordination.py --days 8 --checkpoints results/ppo-seed0/checkpoints --output results/comparison
```

### Installation

```bash
# Clone the repository
git clone https://github.com/CLAIR-LAB-TECHNION/energy-net.git
cd energy-net

# Install the package
pip install -e .

# For RL training capabilities
pip install -e .[rl]

# For development and testing
pip install -e .[test]
```

### Basic Usage

```python
from energy_net.gym_envs.pcs_env import PCSEnv
from stable_baselines3 import SAC

# Create environment
env = PCSEnv(
    test_data_file='tests/gym/data_for_tests/synthetic_household_consumption_test.csv',
    predictions_file='tests/gym/data_for_tests/consumption_predictions_without_features.csv',
    shortage_penalty=1.0
)

# Train an agent
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluate
obs, info = env.reset()
for _ in range(48):  # One day (48 half-hour steps)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

---

## Environments

### PCSEnv (Production-Consumption-Storage)

Simulates a household or facility managing:
- **Energy consumption**: Time-varying demand that must be met
- **Battery storage**: Charge/discharge decisions for cost optimization
- **Grid interaction**: Buy/sell energy at dynamic prices

**Agent Goal**: Minimize energy costs while meeting consumption requirements through strategic battery management.

### ISOEnv (Independent System Operator)

Simulates a grid operator managing:
- **Pricing**: Set electricity prices for each time period
- **Dispatch**: Determine energy supply to meet predicted demand

**Agent Goal**: Maximize revenue while maintaining grid stability and minimizing forecast errors.

### AlternatingISOEnv (Multi-Agent)

Coordinates ISO and PCS agents in an alternating training framework:
- ISO sets prices → PCS responds → ISO adapts → PCS adapts
- Enables study of strategic interactions between grid operators and consumers
- Configurable reward structures (competitive or cooperative)

---

## Documentation

### Energy prediction

Select the target explicitly when creating a predictor. Demand, generation, net load, and other energy series use the same API, and tabular outputs are named for the selected target.

```python
from prediction.energy_predictor import create_energy_predictor

predictor = create_energy_predictor(
    "generation.csv",
    target_col="Production",
)
forecast = predictor.predict("2025-06-15", "12:00")
daily_forecasts = predictor.predict_days("2025-06-15")
```

Consumption-named callables and the former module import path remain available only as deprecated compatibility interfaces.

For detailed tutorials, examples, and advanced usage, see:

📓 **[EnergyNet Tutorial](Energy_Net_Tutorial.ipynb)** - Comprehensive guide covering:
- Environment setup and configuration
- Training agents with different algorithms
- Visualization and analysis
- Multi-agent coordination scenarios

📈 **[Prediction Tutorial](Prediction_Tutorial.ipynb)** - Target-neutral forecasting for explicitly selected demand, generation, or other energy series.

Run the standalone prediction example from the repository root:

```bash
python tests/unittests/energy_prediction_demonstration.py
```

---

## Project Structure

```
energy-net/
├── prediction/                    # Target-neutral energy forecasting
│   └── energy_predictor.py        # Primary prediction API
├── energy_net/                    # Core package
│   ├── gym_envs/                  # Gym environments (PCS, ISO, Alternating)
│   ├── grid_entities/             # Grid components (batteries, consumption units, etc.)
│   └── foundation/                # Base classes and dynamics
├── tests/                         # Test suite and test data
│   ├── gym/                       # Environment test data and generators
│   └── unittests/                 # Unit tests and prediction demonstration
├── Energy_Net_Tutorial.ipynb      # Comprehensive environment tutorial
├── Prediction_Tutorial.ipynb      # Energy-target prediction tutorial
└── README.md                      # This file
```

---

## Dependencies

**Core Requirements:**
- Python 3.10+
- numpy, scipy, pandas
- gymnasium
- pyyaml
- scikit-learn

**For RL Training:**
- stable-baselines3
- torch
- sb3-contrib
- matplotlib
- tensorboard

Install RL dependencies with: `pip install -e .[rl]`

Install development and testing dependencies with: `pip install -e .[test]`

Install Excel import support for `process_consumption_file` with:
`pip install -e .[excel]`

---

## Contributing

This project is developed by [CLAIR-LAB-TECHNION](https://github.com/CLAIR-LAB-TECHNION). Contributions, issues, and feature requests are welcome!

---

## License

This project is licensed under the MIT License.
### GPU-hour electricity and carbon experiment

The GPU-hour scenario defines hardware capacity, active/idle power, PUE,
checkpoint traffic, rental prices and user delay costs with explicit units.
Generation-side carbon costs enter OPF bids and electricity bills once. A
separate, voluntary company carbon valuation enters its own decision objective;
cash profit, consumption footprint and actual grid emissions remain separate
reported quantities. See [the mechanism](docs/gpu_carbon_mechanism.md).

```sh
python scripts/train_gpu_coordination.py --seed 0 --output results/gpu-seed0
```

This trains three independent user PPOs, compares platform PPOs using identical
77-menu actions with electricity-only or electricity-and-carbon observations,
and then updates all four actors in alternating phases. Both information
variants have the same company objective and initial user models. The
`cash_only` variant separately measures a change in company preferences.
Full costs, dispatch-dependent costs and paired background-load attribution
are all retained; common equipment costs are reported explicitly.

### Conditional contracts and resource sharing

The next experiment keeps the same GPU jobs, 77 quote choices and four PPO
actors. A completed migration or delay qualifies for the user's quoted rebate.
A cost-recovery resource procurement account shares verified net resource
savings, offsets the platform's existing electricity-bill gains, and retains
its own share. The low-carbon version makes positive sharing conditional on
actual grid emissions being no higher than the original-contract baseline.
The platform learns its settled cash profit. The accounting and incentive
identities are described in [the contract design](docs/flexibility_contract.md).

```sh
python scripts/train_flex_coordination.py --seed 0 --users results/gpu-seed0/initial_users --output results/flex-seed0
python scripts/evaluate_flex_coordination.py --run-root results/flex-seed0 --output results/flex-comparison --include-menu-potential
```

For the equal-budget continuation used in the contract comparison, retain the
pilot run and add a zero-initialised learned intercept for each tariff. The
existing network, Adam state and three user models are retained:

```sh
python scripts/continue_flex_coordination.py --seed 0 --source results/flex-seed0 --output results/flex-seed0-intercepts
python scripts/evaluate_flex_coordination.py --run-root results/flex-seed0-intercepts --output results/flex-comparison --include-menu-potential
```

Each regime receives the same additional platform sample budget. Joint
checkpoints contain all four actors from one validation-selected cycle, with
cycle zero representing the frozen-user stage.

### Sequential binding execution contracts

The posted-offer experiment replaces each user's directory of qualified
alternatives with one published contract at a time. The platform chooses an
execution region, delay and extra rebate; the user accepts it or retains its
already accepted old contract. Publication checks physical feasibility and
funding, and acceptance executes that binding offer. Electricity is cleared
by network OPF under a common carbon cap; the generation carbon policy price
is exogenous. These are task-specific contract quotes rather than a single
public tariff shared by all customers.

One platform PPO has 61 actions and makes 12 decisions per day. Three separate
binary user PPOs each make four own-task decisions. With gamma one, their
returns equal actual daily cash or private utility increments. A public day
is repeated with independently sampled private preferences that remain fixed
within that day. The full and no-explicit-history controls use the same
initial user models and one root algorithm seed, zero.

For each accepted contract, net procurement saving is M=G-R, where G is the
verified resource saving and R is the extra user rebate. Platform cash grows
by 0.8M. True social saving is S=G-dD=M+U and is reported separately; the
platform never observes private inconvenience dD. Existing capacity costs
are common to the policies, and absolute profit is retained alongside its
increment. See [the mechanism notes](docs/posted-offer-mechanism-notes.txt).

```sh
python scripts/train_posted_offers.py --stage users --legacy LEGACY_DIR --users WARM_USERS_DIR --output RUN/initial-users --steps 8192 --seed 0
python scripts/train_posted_offers.py --stage train --legacy LEGACY_DIR --users RUN/initial-users --output RUN/full-history --steps 131072 --joint-cycles 1 --seed 0
python scripts/train_posted_offers.py --stage train --legacy LEGACY_DIR --users RUN/initial-users --output RUN/no-history --steps 131072 --joint-cycles 0 --no-history --seed 0
python scripts/evaluate_posted_offers.py --stage select --legacy LEGACY_DIR --users RUN/initial-users --output RUN/baselines-initial --workers 4 --seed 0
python scripts/evaluate_posted_offers.py --stage test --legacy LEGACY_DIR --users RUN/initial-users --output RUN/baselines-initial --model-dir RUN/full-history --extra-model no_history_ppo=RUN/no-history --seed 0
python scripts/evaluate_posted_offers.py --stage select --legacy LEGACY_DIR --users RUN/full-history/best_joint --output RUN/baselines-joint --workers 4 --seed 0
python scripts/evaluate_posted_offers.py --stage test --legacy LEGACY_DIR --users RUN/full-history/best_joint --output RUN/baselines-joint --model-dir RUN/full-history --checkpoint best_joint --seed 0
python scripts/summarize_posted_offers.py --experiment RUN
```

Deterministic baselines include a fixed raw contract, a fixed regional/time
price matrix with greedy routing, a dynamic value-share rule, and empirical
response models with or without public-history updates. Selection uses
training cases only; checkpoint selection uses validation mean M. Test
comparisons use the same frozen users and old contracts, with a separate
baseline selection for the jointly trained users. Four private draws are
averaged within each of 32 new public weather days before paired comparisons.
Quote variation itself is never rewarded or treated as an effectiveness metric.
