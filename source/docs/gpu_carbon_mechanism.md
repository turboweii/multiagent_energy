# GPU-hour pricing and the carbon mechanism

The current accepted-booking amendment experiment uses a common physical
carbon cap and zero internal carbon valuation. Its mechanism and comparison
references are documented in
[先补偿用户、再分净收益的算力合同改签机制](net_surplus_amendments.md).
This document records the earlier GPU scenario and its internal-carbon variant.

This study asks how a cloud platform can induce independent users to move or
delay eligible compute jobs through posted regional/time prices. The platform
and three users retain separate objectives. The physical grid operator clears
generation and network flows by OPF; it is not a fifth learning agent. Every
comparison uses the same GPU-hour jobs, deadlines, network, weather, original
contracts, and user preferences.

The GPU scenario is a hardware-referenced synthetic experiment. One work unit
is one H100-equivalent GPU-hour on homogeneous eight-GPU nodes. The new scenario
does not retain the legacy construction in which every day has a binding
system-wide carbon cap. It uses a generation-side external carbon price and
lets actual dispatch emissions vary with the workload allocation.

## Two distinct carbon values

### External generation-side carbon price

Let `pi = GridSpec.carbon_price_yuan_per_t`, in yuan/tCO2. Generator `g` has
variable resource cost `c_g` in yuan/MWh and direct emissions factor `e_g` in
tCO2/MWh. The grid's economic clearing solves

\[
\min_{P,F,\theta}\sum_{t,g}(c_g+\pi e_g)P_{g,t}\Delta t
\]

subject to generation capacities, nodal balance, DC power flow, and transmission
capacities. The optional legacy carbon cap is an independent constraint; the
main GPU scenario passes `None` for that cap. There is no carbon-market auction
or strategic allowance trader in this first implementation: `pi` is an external
scenario input, while electricity prices are endogenous OPF nodal duals.

The model assigns the carbon payment `pi * E_grid` to generators. That payment
is part of their offers and is reflected in cleared nodal electricity prices.
The platform pays its ordinary nodal electricity bill. It does not pay the same
generation-side carbon charge again as an additional platform invoice.

Dispatch reporting separates:

- physical generation resource cost, `sum(c_g * P_g,t * dt)`;
- generation-side carbon cash payment, `pi * E_grid`;
- the dispatch optimisation value, their sum.

A carbon payment is a monetary transfer within an accounting boundary that
includes the recipient; it is not automatically an additional physical resource
cost or an estimate of climate damage. The current `system_cost_yuan` field uses
generation resource cost, variable compute/network costs, and users' monetary
inconvenience. It does not add carbon cash transfers to those resource costs.

### The platform's voluntary internal carbon value

Let `eta = config.internal_carbon_value_yuan_per_t`, also in yuan/tCO2. It is an
additional private sustainability preference of the platform. It is not an ETS
invoice, an offset purchase, a payment received by another agent, or the same
external charge being collected twice. This distinction must be retained when
describing the experiment.

The firm's cash operating profit is

\[
\Pi=R_{\rm compute}-B_{\rm electricity}
   -C_{\rm variable\ compute/network}-C_{\rm fixed\ fleet}.
\]

Its private decision objective is

\[
J_{\rm platform}=\Pi-\eta E_{\rm DC}.
\]

`E_DC` is the consumption footprint allocated to the platform's electricity,
including idle, compute, and migration energy. The PPO platform reward is

\[
r=\frac{J_{\rm platform}-J_{\rm original}}{1000},
\]

in fixed thousand-yuan units. Subtracting the same original-contract reference
does not change which action maximises the firm's objective on a given day.
This objective represents a firm with a stated internal carbon preference;
it is not a claim that a cash-profit-only firm voluntarily values an unpriced
environmental benefit. Users continue to compare their locked service bill
and private delay/migration inconvenience; no green preference is imposed on
them.

Both cash participation (`delta_platform_profit_yuan >= 0`) and objective
participation (`delta_platform_objective_yuan >= 0`) are reported. A firm with
positive internal-objective gain can still lose cash profit. Those outcomes
must not be merged into a single claim of increased profit.

## Consumption footprint, actual emissions, and marginal signals

The code in `energy_net/coordination/carbon.py` computes realised consumption
carbon intensity by proportional mixing along the solved physical flows.
At each time and node `n`, let `G_n` be local generation, `Q_n = sum_g(e_g P_g)`
its emissions rate, and `I_mn` the positive power actually flowing from node
`m` into node `n`. The mixing intensities `h_n` satisfy

\[
\left(G_n+\sum_m I_{mn}\right)h_n-\sum_m I_{mn}h_m=Q_n.
\]

`h_n` has units tCO2/MWh, equivalently kgCO2/kWh. Local load and outgoing flows
share that intensity. Therefore, for every period in the lossless grid,

\[
\sum_n L_n h_n=\sum_g e_g P_g.
\]

The platform's allocated footprint is

\[
E_{\rm DC}=\sum_{t,n}D^{\rm DC}_{n,t}h_{n,t}\Delta t.
\]

`nodal_carbon_intensity(spec, clearing, load_mw)` returns `[time, bus]`
intensities; `consumption_emissions_t(power_mw, intensity, dt_hours)` performs
the energy-weighted summation. Line signs determine direction; negative
emissions are not manufactured for clean imports. Empty islands have zero
allocated intensity. The code checks nodal mixing and total emissions
conservation.

This allocated consumption footprint is different from the system consequence
of moving a job. A destination with a cleaner average supply mix need not have
a cleaner marginal generator. The reported actual grid reduction must use
paired, fully re-cleared dispatches:

\[
\Delta E_{\rm grid}^{\rm reduction}
=E_{\rm grid,original}-E_{\rm grid,method}.
\]

The optional `reference_lme(grid, load_mw, perturbation_mw=0.001)` diagnostic
perturbs one node/time load at a time and re-clears the uncapped grid. It returns
the local derivative in tCO2/MWh. The external generation-side carbon price
remains in the offers. This local marginal-emissions signal can be negative,
and it must not be clipped into an average-intensity measure. It is valid near
the reference dispatch regime; a large relocation crossing generation or
transmission limits requires a new dispatch calculation. The main observer
currently uses reference consumption intensities, not this optional expensive
diagnostic.

## Identical 77-action tariff space

`FixedGpuTariffCatalogue` contains the original tariff and nineteen fixed
discount masks:

- four masks, each selecting one of the four time blocks across all regions;
- three masks, each selecting one region across all time blocks;
- twelve masks, each selecting one region/time cell.

Each mask has four discount magnitudes: `0.0025`, `0.005`, `0.01`, `0.02`
(0.25%, 0.5%, 1%, 2%). Every action offers the full physical quota, `1.0`.
Thus the catalogue has `1 + 19 * 4 = 77` actions. Magnitudes must exist exactly
in `config.discount_levels`; silently rounding them to zero in an older
configuration is rejected.

These masks never rank electricity prices, average carbon intensity, marginal
emissions, or realised candidate returns. The same index decodes to the same
regional/time mask and discount under either observation mode, on every day.
The catalogue therefore does not give the carbon-information policy additional
actions or hide a carbon-ranking rule in its action decoder.

`sample()` supplies varied public quotes during user training: 60% catalogue
menus, 20% independently drawn cell discounts, and 20% uniform discounts, all
within configured discount levels and full quotas. These are exploration
menus, not optimal-response labels.

## Information experiment versus preference experiment

The main information comparison holds fixed:

- external carbon price `pi` and platform internal carbon value `eta`;
- the 77 raw actions, discount limits, and public physical feasibility rules;
- the platform objective `J`, users' objectives, and private participation rules;
- learning budgets, evaluation days, jobs, grid, and deadlines.

It changes only direct carbon-information columns in `GpuTariffObserver`:

| Observation mode | Common public state | Direct carbon columns |
| --- | --- | --- |
| Electricity-information baseline | Reference electricity costs, grid/renewable status, capacities, workload, and candidate monetary forecasts | All zero |
| Electricity-and-carbon information | Exactly the same common state | Reference nodal intensity, external price, internal value, DC footprint, candidate carbon forecasts |

Both observations have shape `[77, 136]`. The 33 columns whose names begin
with `carbon_` are the only columns ablated. Their indices are exposed as
`observer.carbon_feature_indices`. The last column is the exact-duplicate
raw-action mask. No action is masked because an oracle says it will lose money,
violate private utility, or cause emissions.

The electricity-information baseline still observes physical renewable
availability and nodal electricity prices that already contain the external
carbon price. It also trains on the same carbon-valued objective. Thus this
experiment tests the value of **explicit carbon information**, not the removal
of all indirect carbon information or all environmental incentives.

Separately, `eta = 0` gives a cash-profit preference baseline. Comparing it
with `eta > 0` changes the firm's objective. Such a result is a private
cost/carbon-preference trade-off; it cannot by itself establish that one
information representation or PPO algorithm is better. That preference
comparison must not be labelled the same as the fixed-objective information
ablation.

### Public candidate features and normalisation

The common state contains regional/block reference electricity costs in
yuan/GPU-hour, original compute occupancy, renewable available/curtailed MW,
site GPU capacities, line utilisation, owner aggregate GPU-hours, and the
reference DC electricity bill per GPU-hour.

For each candidate, the observer uses public job sizes and hypothetical
physical execution options to compute quoted discounts, original-contract
discount windfalls, potential bill concessions, and forecast margins. These
forecasts hold the original-contract electricity prices fixed. Carbon
forecasts analogously hold the original reference intensities fixed. Candidate
carbon-adjusted margins are only information provided to the learned scorer;
they are not inserted as logits, optimal labels, or realised-policy outcomes.

The observer does not read private `beta`, private migration aversion,
`user_loss_yuan`, a user's learned action, or a candidate final dispatch. It
does not publish any menu while constructing observations.

Each mode fits per-column mean and standard deviation using only requested
training scenarios. Standard deviations have a 0.01 floor. Electricity-mode
carbon columns are exactly zero in every training observation, so their fitted
mean is zero and their normalised value remains zero. Non-carbon columns and
their normalisation statistics are identical between modes for identical
training scenarios. Cached arrays are copied before training masks are applied.
The temporary discount curriculum changes only action availability during
training; final evaluation makes all unique menus available in both modes.

## Reporting fields and units

| Field | Meaning | Unit |
| --- | --- | --- |
| `work_gpu_hours` | Completed work; same workload across schemes | GPU-hour |
| `policy_carbon_price_yuan_per_t` | External generator-side price `pi` | yuan/tCO2 |
| `cap_shadow_price_yuan_per_t` | Dual value of optional total-emissions cap; zero when absent | yuan/tCO2 |
| `generation_cost_yuan` | Physical generation variable resource cost | yuan/day |
| `generator_carbon_payment_yuan` | Generator-side `pi * E_grid` cash payment | yuan/day |
| `grid_dispatch_objective_yuan` | Generation resource cost plus carbon payment | yuan/day |
| `electricity_bill_yuan` | DC nodal bill, including idle/compute/transfer electricity | yuan/day |
| `fixed_compute_cost_yuan` | Declared daily existing-fleet cost | yuan/day |
| `platform_profit_yuan` | Cash operating profit after the declared fleet cost | yuan/day |
| `internal_carbon_value_yuan_per_t` | Voluntary internal value `eta` | yuan/tCO2 |
| `internal_carbon_valuation_yuan` | Noncash `eta * E_DC` decision valuation | yuan/day |
| `platform_objective_yuan` | Cash profit minus internal carbon valuation | yuan/day equivalent |
| `dc_consumption_emissions_t` | Flow-traced DC consumption footprint | tCO2/day |
| `carbon_emissions_t` | Actual emissions from all dispatched generators | tCO2/day |
| `dc_footprint_reduction_t` | Original footprint minus method footprint | tCO2/day |
| `grid_emissions_reduction_t` | Original actual emissions minus method actual emissions | tCO2/day |
| `dispatch_system_cost_yuan` | Resource generation, variable compute/network, user inconvenience | yuan/day |
| `full_system_cost_yuan` | Dispatch system cost plus fixed fleet cost | yuan/day |
| `incremental_system_cost_yuan` | Dispatch cost less matched background-only generation resource cost | yuan/day |
| `dc_energy_share` | DC electricity divided by complete grid electricity | fraction |

`delta_*` fields use method minus original, so a negative cost delta represents
savings. The `*_reduction_t` fields reverse that sign for readable positive
reductions. Average footprint and actual grid reduction are both required.

Full-grid cost, platform-attributable incremental cost, and full cost including
common fleet costs are different denominators. Report their definitions
alongside absolute paired savings; changing the denominator does not create
additional physical savings. Every comparison should also report completion,
deadline compliance, user participation, cash/objective participation, and
violation or loss days without dropping unfavourable days.

## Carbon-price sensitivity and interpretation

For the inherited coal/gas resource costs and factors, the simple uncongested
merit-order crossover is

\[
\pi^*=\frac{450-180}{0.9-0.35}\simeq490.91\ \text{yuan/tCO2}.
\]

At lower external prices the coal/gas order can stay unchanged. A carbon-price
change can therefore alter electricity bills without changing the dispatched
generation mix. Network constraints, renewable saturation, different active
generators, and workload relocation can create other thresholds. This formula
is a diagnostic for this synthetic fleet, not a calibrated real carbon price
or a universal requirement for carbon policy to work.

Predeclare external-price, internal-value, load-share, congestion, and renewable
sensitivities. Do not choose a test-set parameter only because it produces a
large reduction. A positive internal value may lower allocated DC footprint
while causing little or even adverse change in actual system emissions; paired
full-grid re-clearing is needed to identify that outcome.

The addition of 33 input columns, a carbon penalty, or a multi-agent PPO solver
is not by itself a research novelty. Relevant questions are whether independent
user price responses unlock feasible flexibility, when private objectives and
network/carbon outcomes align, and how much explicit carbon information improves
decisions at a fixed objective and comparable participation requirements.

## Sources and implementation checks

- [Radovanovic et al., Carbon-Aware Computing for Datacenters](https://arxiv.org/abs/2106.11750):
  operational carbon-aware time shifting while preserving total daily compute
  capacity; useful for workload-completion constraints and carbon forecasts.
- [Lindberg et al., A Guide to Reducing Carbon Emissions through Data Center
  Geographical Load Shifting](https://arxiv.org/abs/2105.09120): locational
  marginal carbon emissions and network-aware geographical shifting; supports
  distinguishing marginal consequences from average supply mix.
- [Cote and Sun, Locational Marginal Emissions for Carbon-Aware Data Center
  Operations in Large-Scale Power Grids](https://arxiv.org/abs/2512.18819):
  nodal dispatch evaluation and verification of LME-guided interventions against
  actual emissions changes.
- [NVIDIA DGX H100 infrastructure planning](https://docs.nvidia.com/dgx-superpod/design-guides/dgx-superpod-data-center-design-h100/latest/planning.html):
  hardware power/capacity reference for the separate synthetic GPU scenario;
  the study's operating powers and monetary parameters remain explicit assumptions.

Tests in `test_carbon_accounting.py` check hourly carbon conservation, reversed
line flows, mixed imports/local generation, empty islands, and actual emissions
changes from an equal-energy shift. Tests in `test_gpu_tariffs.py` check exact
action-space identity, isolation from private preferences and user losses,
carbon-only observation ablation, normalisation, duplicate masks, curriculum
masks, and absence of market-state mutations while constructing forecasts.
