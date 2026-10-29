# Conditional flexibility contracts and resource-savings settlement

The current compensation-before-sharing experiment is documented in
[先补偿用户、再分净收益的算力合同改签机制](net_surplus_amendments.md).
This document describes the earlier original-contract sharing experiment.

The experiment studies whether a cloud platform can earn additional cash by
inducing independent users to provide useful load flexibility. The platform
publishes a regional/time rebate menu; each of three users chooses its own
execution contracts. The grid is cleared by OPF. A preannounced procurement
contract allocates the resulting resource benefit. There are four independent
PPO actors: the platform and the three users.

## Contract and participant roles

The original place/time execution contract remains available at the original
GPU-hour price. Under a conditional rebate, an alternative qualifies only when
it actually migrates or delays the task. Rebooking the same physical execution
as option 1 receives no rebate. The user's accepted invoice is locked before
final grid clearing; settlement does not change it or undo execution.

The sharing counterparty is a **modeled cost-recovery system-resource
procurement account**. Its benefit ledger measures resource-cost savings net
of benefits already captured in the platform's electricity settlement. This
account does not assert that an actual ISO owns all generators, receives all
fuel-cost savings as cash, or operates the contract described here. A study
of actual wholesale-market funding would additionally require generator
revenues, background-load payments, congestion rents and regulatory accounts.

Platform participation is measured against action 0 (original task contracts
and original electricity settlement); because the four arms change settlement
institutions, this comparison does not establish an advantage over the
platform's best flexible quote under the previous institution, which is
checked separately using the strong baseline.

The procurement account announces the main sharing fraction **alpha = 0.8**
before platform actions. The platform does not choose alpha. This fraction
determines how much positive resource benefit is available to cover the
platform and users; it is a study mechanism parameter, not a measured market
rule. All four main experiment arms have internal carbon valuation eta = 0.
The platform therefore receives an ordinary cash-profit objective. The
generation-side policy carbon price continues to enter OPF and electricity
prices in every arm.

## Fixed reference and observable quantities

Superscript 0 denotes the same day's original task contracts, using the same
work quantity, deadlines, background demand, renewable availability, network,
generator inputs and policy carbon price. The reference is fixed before the
platform quotes; the platform cannot move it to manufacture settlement value.
The deterministic simulation can solve this reference counterfactual exactly.

All monetary quantities below are yuan per day:

| Symbol | Account |
| --- | --- |
| C | Actual grid generation resource cost, excluding carbon payments and electricity-payment transfers |
| K | Platform variable computing and migration operating cost |
| B | Platform electricity bill at the final nodal clearing prices |
| Q | Revenue billed to compute users |
| D | Users' private monetary delay and migration inconvenience |
| E | Actual whole-grid emissions in tonnes per day |

Fixed equipment-capacity costs are disclosed separately and cancel in all
same-day changes. Carbon payments enter the ordinary electricity settlement
once; they are not added again to C. The account uses the public coefficients
and measured physical quantities underlying K. The private D affects user
choice and system-cost evaluation, but does not enter the procurement payment.

Define resource savings before private inconvenience, the platform's own
pre-rebate savings, and the total user rebate:

\[
G=(C^0+K^0)-(C+K),\qquad
a=(B^0+K^0)-(B+K),\qquad
R=Q^0-Q.
\]

The existing electricity-and-service contracts give the platform a profit
change of a - R. A direct additional payment alpha G would leave the already
captured a in place. The new settlement offsets a explicitly.

## Sharing and signed settlement

Without a carbon qualification condition, the platform's resource entitlement
is

\[
\phi(G)=\alpha\max(G,0)+\min(G,0).
\]

In the main low-carbon arm, positive sharing additionally requires actual
whole-grid emissions E <= E0 + epsilon, where epsilon = 1e-7 tonnes handles
solver tolerances. If this condition fails, phi(G) = min(G, 0). A decrease in
an allocated data-center footprint alone does not establish this condition.

The contract's final settlement adjustment is

\[
T=\phi(G)-a.
\]

For T > 0 the procurement account pays the platform. For T < 0 the platform
pays the account. The latter can offset electricity-price gains that exceed
the contracted resource entitlement, or charge for an unsuccessful resource
outcome. These payment directions are agreed before actions. They do not
screen or roll back final platform profit, reverse accepted tasks, or alter
user invoices. A platform can lose money after settlement.

The platform's actual additional cash profit and the account's residual
resource benefit are

\[
\Delta\Pi=a-R+T=\phi(G)-R,
\]
\[
V_{buyer}=G-a-T=G-\phi(G).
\]

When sharing is enabled, Vbuyer is nonnegative for any G: negative resource
benefits are fully charged to the platform; positive benefits are retained
in the contracted proportions, or retained entirely by the account when
positive carbon eligibility fails. When sharing is disabled, T = 0 and the
unadjusted account G - a can be negative. That value remains in the report.

The system's actual operating-cost saving is

\[
S=G-D.
\]

The payment is a transfer. The complete allocation identity is

\[
S=\Delta\Pi+\underbrace{(R-D)}_{user\ utility\ gain}+V_{buyer}.
\]

Consequently, introducing a payment cannot by itself reduce system cost.
Only a change in physical dispatch, variable operations or user inconvenience
changes S. The report keeps both payment directions, buyer benefits, platform
profit before and after settlement, and the numerical error of this identity.

## What positive platform profit establishes

Suppose the sharing contract is active, every user satisfies its original
reservation utility, and the platform has a strictly positive **profit
increment after settlement**, Delta Pi > 0. User participation implies
R >= D >= 0. It follows that phi(G) > R >= 0, so G > 0. In the low-carbon arm,
this also forces the positive carbon eligibility condition to hold. Then

\[
\Delta\Pi=\alpha G-R>0,
\qquad
G-D=\Delta\Pi+(1-\alpha)G+(R-D)>0.
\]

Thus, for a realized day meeting these conditions, the platform's positive
additional cash profit is sufficient for strictly positive system net-cost
savings and, in the low-carbon arm, no increase in actual grid emissions
beyond epsilon. This is an accounting/incentive implication, not a proof that
PPO finds the best contract or maximizes total welfare. The platform shares
surplus with users and the procurement account.

Zero profit increments do not establish strict cost improvements. In
particular, zero entitlement and zero rebate can yield zero platform gain
without establishing carbon eligibility. A positive average profit increment
also does not establish eligibility on every individual day. Daily losses,
carbon failures and participation rates remain in the evaluation.

## Four-arm comparison and learning

| Arm | User rebate rule | Sharing | Actual-grid carbon qualification |
| --- | --- | --- | --- |
| published_no_share | Published price applies also to unchanged rebooking | Disabled | Disabled |
| conditional_no_share | Actual migration or delay required | Disabled | Disabled |
| conditional_share | Actual migration or delay required | Enabled | Disabled |
| conditional_green_share | Actual migration or delay required | Enabled | Enabled |

The comparison keeps eta = 0, the same raw 77-menu action space, physical
workloads, GPU power model, network and generation-side carbon price. The
conditional arm changes which execution products qualify for its rebate;
the sharing arm changes actual cash settlement. These are mechanism changes,
not hidden additional reward bonuses.

The platform's PPO reward is its realized after-settlement cash-profit change
divided by a fixed 1000 yuan. Each user's PPO reward continues to depend on
its own accepted bill and inconvenience. No private D is supplied to the
buyer payment formula. Platform observations use public reference dispatch
and physical contract forecasts, without final candidate prices or simulated
user responses supplied as observations.

The run uses one algorithm training seed, 0; scenario training seeds 0--63;
validation seeds 1000--1015; and new test seeds 40000--40031. The platform
comparison first freezes the same initial learned user weights in all arms.
Alternating updates of all four PPO actors are reported separately, since
the user models can then differ between arms. Model selection uses validation
days, and the held-out test days are not used to choose quotes or checkpoints.

Relevant measures are system operating-cost savings and complete-cost savings,
realized user utility, pre/post-settlement platform profit, buyer residual,
actual grid emissions, allocated DC footprint, task/deadline completion,
positive/negative transfer amounts, and joint-benefit days. Constant-menu and
complete-information finite-menu references need the same feasibility and
participation conditions before interpreting performance gaps.

## Prior work and contribution boundary

[Zhan et al., Extending Demand Response to Tenants in Cloud Data Centers via
Non-intrusive Workload Flexibility Pricing](https://arxiv.org/abs/1603.05746)
studies the split incentive between a cloud operator and tenants, retaining
usage-based prices while rewarding flexible deadlines. It motivates explicit
user flexibility products and monetary compensation.

[Wu et al., Incentivizing the Spatiotemporal Flexibility of Data Centers Toward
Power System Coordination](https://doi.org/10.1109/TNSE.2023.3234445), IEEE TNSE
10(3), 1766--1778 (2023), describes system/data-center coordination using a
bilevel model and a profit-sharing mechanism. Its publisher abstract is also
available at [IEEE](https://xplorestaging.ieee.org/document/10011558). This
supports studying system-level sharing, but the settlement identities in this
document are the present model's definitions, not a claimed reproduction of
the paper's equations.

Conditional tenant compensation and system/data-center sharing therefore
already have precedents. A distinct contribution would require evidence about
the combination of independent private user decisions, physical transmission
constraints, actual carbon qualification and implementable price contracts,
including when the mechanism improves the attainable cost/profit/emissions
tradeoffs. Merely adding PPO, a carbon feature, or a sharing coefficient does
not establish that contribution.
