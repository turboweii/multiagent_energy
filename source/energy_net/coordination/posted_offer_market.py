"""One pre-funded execution offer per task, accepted or declined privately.

Public funding verifies G-R before publication. Acceptance binds the published
contract without a second resource or budget screen. Its actual cash ledger
is the existing net-surplus settlement; private inconvenience affects only
user participation, user reward and subsequent social-cost measurement.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .amendment_market import AmendmentEpisode


OFFER_DELAYS_HOURS = (0., 1., 2., 3.)
OFFER_RATES = (0., .0025, .005, .01, .02)
N_POSTED_ACTIONS = 1 + 3 * len(OFFER_DELAYS_HOURS) * len(OFFER_RATES)


def decode_posted_action(action):
    """Static order: no offer, then region x original-release delay x rate."""
    value = np.asarray(action)
    if value.size != 1:
        raise ValueError("Provide one posted-offer action")
    value = value.item()
    if not np.isfinite(value) or int(value) != value or not 0 <= value < N_POSTED_ACTIONS:
        raise ValueError("Posted-offer action must be an integer in [0, 61)")
    if int(value) == 0:
        return None
    route, rate_index = divmod(int(value) - 1, len(OFFER_RATES))
    region, delay_index = divmod(route, len(OFFER_DELAYS_HOURS))
    return region, OFFER_DELAYS_HOURS[delay_index], OFFER_RATES[rate_index]


class PostedOfferEpisode(AmendmentEpisode):
    """Sequential offers over the existing physical routes and old contracts.

    Absent original routes stay absent, including remote zero-delay and local
    three-hour-delay routes in the current ten-action task model. A user sees
    just its old contract and the single published offer, not the route menu.
    """

    def __init__(self, scenario, booking, grid=None, config=None):
        # AmendmentEpisode.__init__ invokes self.reset() and self._metrics().
        self._posted_ready = False
        self.public_history = []
        self.offered_actions = []
        self.current_offer = None
        self.current_offer_action = None
        self._offer_position = None
        self._offer_plan = None
        self._route_preview_cache = {}
        self.platform_step_gain = 0.
        super().__init__(scenario, booking, grid=grid, config=config)
        if self.config.settlement_order != "net":
            raise ValueError("Posted offers use compensation-before-net-sharing settlement")
        self._posted_ready = True
        self.reset()

    @property
    def history(self):
        return self.public_history

    def reset(self, platform_action=None):
        self._route_preview_cache.clear()
        if not self._posted_ready:
            return super().reset(platform_action)
        self.public_history = []
        self.offered_actions = []
        self.current_offer = None
        self.current_offer_action = None
        self._offer_position = None
        self._offer_plan = None
        self.platform_step_gain = 0.
        observation = super().reset()
        if platform_action is not None:
            self.set_offer(platform_action)
        return observation

    def _offer_result(self, action, decoded):
        region, delay, rate = (None, None, 0.) if decoded is None else decoded
        result = dict(action=int(np.asarray(action).item()), offered=decoded is not None,
            region=region, delay_hours=delay, rebate_rate=rate,
            start_step=None, end_step=None, additional_delay_hours=0., additional_migrated=0,
            option_index=None, index=None, clearing=None, feasible=False,
            admissible=False, route_exists=False, changed=False,
            resource_saving_yuan=0., pre_rebate_private_saving_yuan=0.,
            additional_rebate_yuan=0., net_surplus_yuan=0.,
            platform_gain_yuan=0., buyer_gain_yuan=0., settlement_yuan=0.,
            G=0., a=0., R=0., M=0.)
        return result

    def _preview_route(self, region, delay):
        """Exact public physics independent of the offer's five rebate rates."""
        cache_key = (self.position, tuple(self.selected), self.scenario.carbon_budget_t,
                     region, delay)
        if cache_key in self._route_preview_cache:
            return self._route_preview_cache[cache_key]
        result = dict(option_index=None, index=None, clearing=None, feasible=False,
            route_exists=False, changed=False, start_step=None, end_step=None,
            additional_delay_hours=0., additional_migrated=0,
            resource_saving_yuan=0., pre_rebate_private_saving_yuan=0.)
        def keep(reason):
            result["reason"] = reason
            self._route_preview_cache[cache_key] = result
            return result
        job = self.jobs[self.position]
        matches = [index for index, option in enumerate(job.options)
                   if self.option_available[self.position][index]
                   and option.region == region
                   and np.isclose(option.delay_hours, delay, atol=1e-10, rtol=0)]
        if not matches:
            return keep("route_unavailable")
        index = matches[0]
        option, outside = job.options[index], job.options[0]
        result.update(option_index=index, index=index, route_exists=True,
            start_step=int(option.start_step), end_step=int(option.end_step),
            additional_delay_hours=float(option.delay_hours - outside.delay_hours),
            additional_migrated=int(option.migrated) - int(outside.migrated))
        if not self.option_is_changed(job, job.options[index]):
            result["feasible"] = True
            return keep("unchanged_execution")
        result["changed"] = True
        proposed = list(self.selected)
        proposed[self.position] = index
        possible, reason = self._precheck(proposed)
        if not possible:
            return keep(reason)
        # The normal allowance is immutable for a day; keeping it in the OPF
        # key also prevents an explicitly changed allowance reusing old solves.
        key = ("posted_route", self.scenario.carbon_budget_t, tuple(proposed))
        if key not in self._cache:
            self._cache[key] = self.grid.solve(self.scenario.load_for(proposed),
                                               self.scenario.carbon_budget_t)
        clearing = self._cache[key]
        if not clearing.feasible:
            return keep("energy_feasibility")
        old_ops, new_ops = self._operations(self.selected), self._operations(proposed)
        resource = float(self.current_clearing.total_generation_cost + old_ops
                         - clearing.total_generation_cost - new_ops)
        private = float(self._energy_bill(self.current_clearing, self.selected) + old_ops
                        - self._energy_bill(clearing, proposed) - new_ops)
        result.update(clearing=clearing, feasible=True, resource_saving_yuan=resource,
                      pre_rebate_private_saving_yuan=private)
        return keep("physical_route")

    def preview_offer(self, action):
        """Public physical/funding preview, with no private preference access."""
        decoded = decode_posted_action(action)
        result = self._offer_result(action, decoded)
        if self.done:
            return dict(result, reason="day_complete")
        if decoded is None:
            return dict(result, reason="no_offer", option_index=0, index=0,
                        clearing=self.current_clearing, feasible=True,
                        admissible=True, route_exists=True)
        region, delay, rate = decoded
        result.update(self._preview_route(region, delay))
        if not result["feasible"] or not result["changed"]:
            return result
        clearing = result["clearing"]
        resource = result["resource_saving_yuan"]
        private = result["pre_rebate_private_saving_yuan"]
        # Every price keeps its own real compensation and funding check.
        # Neither money, user IR nor an acceptance decision is route-cached.
        rebate = float(self.baseline_bills[self.position] * rate)
        net = resource - rebate
        profit = self.config.resource_share * net
        buyer = (1 - self.config.resource_share) * net
        admitted = bool(net >= -self.config.admission_tolerance_yuan)
        return dict(result, clearing=clearing, feasible=True, admissible=admitted,
            resource_saving_yuan=resource, pre_rebate_private_saving_yuan=private,
            additional_rebate_yuan=rebate, net_surplus_yuan=net,
            platform_gain_yuan=float(profit), buyer_gain_yuan=float(buyer),
            settlement_yuan=float(profit - (private - rebate)),
            G=resource, a=private, R=rebate, M=net,
            reason="funded_offer" if admitted else "negative_distributable_surplus")

    def set_offer(self, action):
        """Publish once for this task; an unfunded offer is never published."""
        if self.done:
            raise ValueError("All tasks have been resolved")
        if self.current_offer is not None:
            raise ValueError("Resolve the published offer before issuing another")
        proposal = self.preview_offer(action)
        if not proposal["admissible"]:
            raise ValueError(f"Cannot publish this offer: {proposal['reason']}")
        self.current_offer = deepcopy(proposal)
        self.current_offer_action = int(np.asarray(action).item())
        self._offer_position = self.position
        self._offer_plan = tuple(self.selected)
        self.amendment_open = bool(proposal["offered"])
        self.platform_step_gain = 0.
        # The publishing interface returns only the public contract. Private
        # bill/loss comparison is obtained separately by the responding user.
        return deepcopy(self.current_offer)

    def _own_gain(self):
        if self.current_offer is None or not self.current_offer["offered"]:
            return 0.
        job = self.jobs[self.position]
        option = job.options[self.current_offer["option_index"]]
        return float(self.current_offer["additional_rebate_yuan"]
                     - (option.user_loss_yuan - self.outside_losses[self.position]))

    def option_bill(self, job, option):
        """Locked old/accepted invoices, or the one currently published offer."""
        position = next(i for i, current in enumerate(self.jobs) if current.job_id == job.job_id)
        if not self.option_is_changed(job, option):
            return float(self.baseline_bills[position])
        if position < self.position and option.index == self.selected[position]:
            return float(self.bills[position])
        if (position == self.position and self.current_offer is not None
                and self.current_offer["offered"]
                and option.index == self.current_offer["option_index"]):
            return float(self.baseline_bills[position] - self.current_offer["additional_rebate_yuan"])
        raise ValueError("This execution has no published posted offer")

    def current_user_observation(self):
        if self.done:
            raise ValueError("No pending task")
        job, position = self.jobs[self.position], self.position
        outside = job.options[0]
        capacity = np.asarray(self.config.capacity_cu_per_hour) * self.config.dt_hours
        allocation, _ = self.allocations_for(self.selected)
        features = np.zeros((10, 7), dtype=float)
        features[0] = [1., 1., 0., 0., 0., outside.region / 2,
                       float(np.max(allocation / capacity))]
        if self.current_offer is not None and self.current_offer["offered"]:
            offer = self.current_offer
            index = offer["option_index"]
            option = job.options[index]
            proposed = list(self.selected)
            proposed[position] = index
            allocation, _ = self.allocations_for(proposed)
            features[1] = [1., 1., -offer["additional_rebate_yuan"] / job.work_cu,
                (option.user_loss_yuan - self.outside_losses[position]) / job.work_cu,
                (option.delay_hours - outside.delay_hours) / 3., option.region / 2,
                float(np.max(allocation / capacity))]
        own_history = [row for row in self.public_history
                       if row["user_id"] == job.user_id and row["offered"]][-4:]
        history_features = np.zeros((4, 4), dtype=float)
        for index, row in enumerate(own_history):
            history_features[index] = [row["rebate_yuan_per_gpuh"],
                row["additional_delay_hours"] / 3., row["additional_migrated"],
                float(row["accepted"])]
        return np.r_[[job.user_id / 2, job.work_cu / 20000, job.release_step / 48,
            job.deadline_step / 48, job.beta_yuan_per_cu_hour / .1,
            job.migration_loss_yuan_per_cu / .05], features.ravel(),
            history_features.ravel()].astype(np.float32)

    def best_response_action(self):
        """A myopic private reference for evaluation, never a user replacement."""
        return int(self._own_gain() > 1e-7)

    def step_accept(self, action):
        if self.done or self.current_offer is None:
            raise ValueError("Publish one offer before the user's response")
        value = np.asarray(action)
        if value.size != 1 or value.item() not in (0, 1):
            raise ValueError("The user must accept (1) or retain the old contract (0)")
        if self.position != self._offer_position or tuple(self.selected) != self._offer_plan:
            raise RuntimeError("The booked plan changed after the offer was published")
        requested = int(value.item())
        position, job, offer = self.position, self.jobs[self.position], self.current_offer
        offered = bool(offer["offered"])
        private_gain = self._own_gain()
        accepted = bool(offered and requested == 1 and private_gain >= -1e-7)
        chosen_index = offer["option_index"] if accepted else 0
        option, outside = job.options[chosen_index], job.options[0]
        if accepted:
            self.selected[position] = chosen_index
            # Physical/funding promises were checked and locked at publication.
            # No preview, new OPF or budget screen is performed on acceptance.
            self.current_clearing = offer["clearing"]
            self.amendment_ledger.append(dict(job_id=job.job_id, position=position,
                **{key: val for key, val in offer.items() if key != "clearing"}))
            self.bills[position] = self.baseline_bills[position] - offer["additional_rebate_yuan"]
            self.losses[position] = option.user_loss_yuan
        own_gain = float(self.baseline_bills[position] + self.outside_losses[position]
                         - self.bills[position] - self.losses[position])
        self.platform_step_gain = float(offer["platform_gain_yuan"] / 1000.) if accepted else 0.
        response = "accepted" if accepted else "rejected" if offered else "no_offer"
        offered_option = job.options[offer["option_index"]] if offered else outside
        public = dict(task_position=position, job_id=job.job_id, user_id=job.user_id,
            action=self.current_offer_action, offered=offered,
            region=int(offered_option.region) if offered else None,
            start_step=int(offered_option.start_step) if offered else None,
            delay_hours=float(offered_option.delay_hours) if offered else 0.,
            additional_delay_hours=float(offered_option.delay_hours - outside.delay_hours) if offered else 0.,
            additional_migrated=int(offered_option.migrated) - int(outside.migrated) if offered else 0,
            rebate_rate=float(offer["rebate_rate"]),
            rebate_yuan_per_gpuh=float(offer["additional_rebate_yuan"] / job.work_cu),
            resource_saving_yuan_per_gpuh=float(offer["resource_saving_yuan"] / job.work_cu),
            accepted=accepted if offered else None, response=response)
        self.public_history.append(public)
        reason = ("user_reservation_utility" if offered and requested and private_gain < -1e-7
                  else response)
        info = dict(job_id=job.job_id, user_id=job.user_id, requested=requested,
            chosen=int(accepted), requested_accept=requested, chosen_accept=int(accepted),
            offered=offered, quoted=offered, accepted=accepted,
            rejected=bool(offered and not accepted), no_offer=not offered,
            amendment_accepted=accepted, amended=accepted,
            offered_action=self.current_offer_action, offered_option_index=offer["option_index"],
            chosen_option_index=chosen_index,
            chosen_source_option=self.option_source_indices[position][chosen_index],
            outside_source_option=self.outside_selected_options[position],
            reservation_reason=reason, bill_yuan=float(self.bills[position]),
            loss_yuan=float(self.losses[position]), outside_bill_yuan=float(self.baseline_bills[position]),
            outside_loss_yuan=float(self.outside_losses[position]),
            additional_rebate_yuan=float(self.baseline_bills[position] - self.bills[position]),
            utility_gain_yuan=own_gain, individual_rational=bool(own_gain >= -1e-7),
            migrated=option.migrated, delay_hours=option.delay_hours,
            additional_delay_hours=option.delay_hours - outside.delay_hours,
            work_cu=job.work_cu, platform_step_gain=self.platform_step_gain,
            public_feedback=deepcopy(public))
        self.requested.append(requested)
        self.reservations.append(info)
        self.offered_actions.append(self.current_offer_action)
        self.position += 1
        self.current_offer = None
        self.current_offer_action = None
        self._offer_position = self._offer_plan = None
        self._route_preview_cache.clear()
        return float(own_gain / job.work_cu), self.done, info

    def step_user(self, action):
        return self.step_accept(action)

    def _metrics(self, clearing, bills, losses):
        result = super()._metrics(clearing, bills, losses)
        history = self.public_history
        offered = sum(row["offered"] for row in history)
        accepted = sum(row["accepted"] is True for row in history)
        result.update(posted_offer_mechanism=True, posted_jobs_resolved=len(history),
            posted_offers=offered, offered_jobs=offered, quoted_jobs=offered,
            accepted_offers=accepted, rejected_offers=offered - accepted,
            no_offer_jobs=len(history) - offered,
            procurement_net_saving_yuan=result["net_distributable_surplus_yuan"],
            social_saving_vs_outside_yuan=result["contract_system_cost_saving_yuan"])
        return result
