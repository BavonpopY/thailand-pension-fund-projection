"""
Module 3 -- Scheme active membership (credit-distribution engine).
==================================================================

The heart of ILO-PENS.  The insured population is held as a dense four
dimensional array

``[sex, income band, insurable age, accumulated credit]``

and each year the mass in that array is moved between cells by the transition
probabilities of the Technical Guide, Part II section 1.1.3.

**Old-age scope.**  The active population is decremented by **mortality and
withdrawal only**.  The invalidity decrement ``VACT = Act . Invrate`` of the
full ILO-PENS specification is absent, because invalidity benefits are out of
scope.  The remaining recursion is

.. math::
    \\mathrm{DACT} &= \\mathrm{Act}(x,t)\\;\\mathrm{Mort}(x,t) \\\\
    \\mathrm{DOUT} &= \\mathrm{Act}(x,t)\\;\\mathrm{Inactiverate}(x,t) \\\\
    \\mathrm{ZACT} &= \\mathrm{Act}(x,t) - \\mathrm{DACT} - \\mathrm{DOUT}

Note that dropping invalidity is *not* neutral: members who would have left as
invalids now remain exposed to the old-age decrements, so a share of them will
eventually vest and draw an old-age pension instead.  This is the correct
treatment for a stand-alone old-age valuation -- it is how the liability
actually behaves once the invalidity branch is financed separately -- but it
means old-age costs here are marginally higher than the old-age slice of a
combined multi-branch run.

Why the full credit distribution
---------------------------------
Carrying credit as an explicit 51-deep axis -- rather than an average past
service per age cell -- is what lets the 180-month vesting condition and the
tiered accrual rate be applied *exactly*, cell by cell, instead of to an
age-group average.  For a scheme as young as the Thai SSO's, the share of
members crossing the vesting threshold is the single biggest driver of the
emerging cost, so an averaged treatment misstates it badly.

Credit accrual (``InsIns()`` step 6)
------------------------------------
A member who stays active for a year accrues credit equal to the contribution
density (Technical Guide II.1.1.5):

.. math::
    \\mathrm{Cred}(x+1, t+1) = \\mathrm{Cred}(x,t) + \\mathrm{Dens}(x,t)

Because ``Dens`` is fractional and the credit axis is a whole-year grid, the
mass in cell ``c`` is split linearly: a share ``Dens`` moves to ``c+1`` and
``1 - Dens`` stays at ``c``.  This preserves head count and mean credit exactly.

Reconciliation to the coverage target
-------------------------------------
The active total is pinned to ``Covrate(x,t) x Employed(x,t)``.  The gap
``D(x+1,t+1) = Act(x+1,t+1) - S[Act(x,t)]`` is closed as in Technical Guide
II.1.1.2:

*   ``D >= 0`` (growing scheme) -- filled by new entrants ``NR . D`` at zero
    credit and re-entrants ``(1 - NR) . D`` drawn from the inactive pool,
    capped by the assumed re-entry availability;
*   ``D < 0`` (mature scheme) -- the excess is moved to the inactive population.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from .config import (
    AGE_MAX_INS,
    AGE_MIN_INS,
    Config,
    N_AGE_INS,
    N_CREDIT,
    N_INC,
    N_SEX,
)

CREDIT_GRID = np.arange(N_CREDIT, dtype=float)
INS_AGE_GRID = np.arange(AGE_MIN_INS, AGE_MAX_INS + 1)


# ---------------------------------------------------------------------------
# Flow container
# ---------------------------------------------------------------------------


@dataclass
class ActiveFlows:
    """Flows out of the active and inactive populations in one projection step.

    Every array is ``[sex, income, insurable age, credit]``.  Flows are indexed
    at the **destination** age (``x+1``) and year (``t+1``) unless the attribute
    docstring says otherwise, so they can be fed straight into the benefit
    routines.
    """

    deaths_active: np.ndarray
    """DACT -- deaths of active members, indexed at the SOURCE age."""

    deaths_inactive: np.ndarray
    """Deaths of inactive members, indexed at the SOURCE age."""

    retirement_claims: np.ndarray
    """Members claiming an old-age benefit, indexed at the destination age."""

    new_entrants: np.ndarray
    """NENT -- first-time registrations, at the destination age."""

    reentrants: np.ndarray
    """RENT -- inactive members returning to active, at the destination age."""

    to_inactive: np.ndarray
    """DOUT -- active members moving to inactive, at the SOURCE age."""

    contributors: np.ndarray = field(default=None)  # type: ignore[assignment]
    """``Cont(x,t) = Act(x,t) . Dens(x,t)`` in the SOURCE year -- the
    contribution-base head count."""


@dataclass
class ActiveState:
    """Mutable state of the insured population.

    Attributes
    ----------
    active
        Active members, shape ``(2, 3, 55, 51)``.
    inactive
        Registered but non-contributing members, same shape.
    """

    active: np.ndarray
    inactive: np.ndarray

    @classmethod
    def from_config(cls, cfg: Config) -> "ActiveState":
        """Initialise from the base-year stocks in the configuration."""
        active = np.array(cfg.membership.active_base, dtype=float, copy=True)
        inactive = np.array(cfg.membership.inactive_base, dtype=float, copy=True)
        expected = (N_SEX, N_INC, N_AGE_INS, N_CREDIT)
        for name, arr in (("active_base", active), ("inactive_base", inactive)):
            if arr.shape != expected:
                raise ValueError(
                    f"membership.{name} must have shape {expected}, got {arr.shape}."
                )
        return cls(active=active, inactive=inactive)

    def copy(self) -> "ActiveState":
        return ActiveState(active=self.active.copy(), inactive=self.inactive.copy())

    # -- reporting helpers ---------------------------------------------------
    @property
    def active_by_age(self) -> np.ndarray:
        """Active members by sex and insurable age, shape ``(2, 55)``."""
        return self.active.sum(axis=(1, 3))

    @property
    def total_active(self) -> float:
        return float(self.active.sum())

    @property
    def total_inactive(self) -> float:
        return float(self.inactive.sum())

    def mean_credit(self) -> np.ndarray:
        """Average accumulated credit by sex and insurable age, shape ``(2, 55)``."""
        n = self.active.sum(axis=(1, 3))
        weighted = (self.active * CREDIT_GRID[None, None, None, :]).sum(axis=(1, 3))
        return np.divide(weighted, n, out=np.zeros_like(n), where=n > 0)

    def vested_share(self, vesting_years: float) -> np.ndarray:
        """Share of active members at or above the vesting threshold, by sex and age."""
        mask = CREDIT_GRID >= vesting_years
        n = self.active.sum(axis=(1, 3))
        vested = self.active[:, :, :, mask].sum(axis=(1, 3))
        return np.divide(vested, n, out=np.zeros_like(n), where=n > 0)


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------


def accrue_credit(arr: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Move mass one step up the credit axis in proportion to contribution density.

    Parameters
    ----------
    arr
        Member counts, shape ``(2, 3, 55, 51)``.
    density
        Contribution density by sex and insurable age, shape ``(2, 55)``.

    Returns
    -------
    np.ndarray
        Same shape, with ``density`` of each cell promoted to ``credit + 1``.
        Mass at the top of the credit grid is retained (the grid is absorbing).

    Notes
    -----
    Exactly preserves both the head count and the mean credit increment; this
    is asserted by ``validation.check_credit_axis_conservation``.
    """
    d = np.clip(density, 0.0, 1.0)[:, None, :, None]
    stay = arr * (1.0 - d)
    move = arr * d
    out = stay.copy()
    out[:, :, :, 1:] += move[:, :, :, :-1]
    out[:, :, :, -1] += move[:, :, :, -1]  # absorbing top credit cell
    return out


def age_shift(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Advance an ``[.., insurable age, ..]`` array by one year of age.

    Returns
    -------
    shifted : np.ndarray
        Same shape; cell at age ``x`` moved to ``x + 1``, age ``AGE_MIN_INS``
        left empty (filled later by new entrants).
    overflow : np.ndarray
        Mass that aged past ``AGE_MAX_INS``, shape ``(2, 3, 51)``.  These
        members leave the insurable population entirely and must settle
        immediately -- the caller pays them a pension or a grant.
    """
    shifted = np.zeros_like(arr)
    shifted[:, :, 1:, :] = arr[:, :, :-1, :]
    overflow = arr[:, :, -1, :].copy()
    return shifted, overflow


def _normalise_decrements(*rates: np.ndarray) -> tuple[np.ndarray, ...]:
    """Scale competing decrement rates so their sum never exceeds 1.

    Parameters
    ----------
    *rates
        Broadcast-compatible rate arrays.

    Returns
    -------
    tuple of np.ndarray
        The same rates, proportionally scaled wherever the total exceeded 1.
    """
    total = sum(rates)
    scale = np.where(total > 1.0, 1.0 / np.maximum(total, 1e-12), 1.0)
    return tuple(r * scale for r in rates)


# ---------------------------------------------------------------------------
# The projection step
# ---------------------------------------------------------------------------


def step_active_population(
    cfg: Config,
    state: ActiveState,
    t: int,
    target_active_by_age: np.ndarray,
) -> Tuple[ActiveState, ActiveFlows]:
    """Advance the insured population from year ``t`` to year ``t + 1``.

    Parameters
    ----------
    cfg
        Fully populated configuration.
    state
        Insured-population state at year ``t``.  Not mutated.
    t
        Source projection year index.
    target_active_by_age
        Required active head count at year ``t + 1`` by sex and insurable age,
        shape ``(2, 55)``.  Normally ``coverage_rate x employed``.

    Returns
    -------
    (ActiveState, ActiveFlows)
        The state at ``t + 1`` and the flows generated during the step.
    """
    mem, scheme = cfg.membership, cfg.scheme
    last = cfg.run.n_t - 1

    q_ins = np.asarray(mem.insured_mortality_qx, float)[:, AGE_MIN_INS : AGE_MAX_INS + 1, t]
    leave_rate = np.asarray(mem.leaving_rate, float)[:, :, t]
    density = np.asarray(mem.density, float)[:, :, t]
    retire_rate = np.asarray(mem.retirement_rate, float)[:, :, min(t + 1, last)]
    reentry_cap = np.asarray(mem.reentry_rate, float)[:, :, min(t + 1, last)]
    nr_share = np.asarray(mem.new_entrant_share, float)[:, min(t + 1, last)]
    inc_shares = np.asarray(mem.income_level_shares, float)

    # -- 1. Competing decrements: mortality and withdrawal only -------------
    q_a, leave_a = _normalise_decrements(q_ins, leave_rate)
    b = lambda r: r[:, None, :, None]  # noqa: E731  broadcast (2,55) -> (2,1,55,1)

    A = state.active
    deaths_active = A * b(q_a)
    to_inactive = A * b(leave_a)
    stay_active = A - deaths_active - to_inactive
    np.clip(stay_active, 0.0, None, out=stay_active)

    # -- 2. Credit accrual, then ageing --------------------------------------
    stay_active = accrue_credit(stay_active, density)
    active_next, active_overflow = age_shift(stay_active)

    # -- 3. Inactive population: mortality, then ageing ----------------------
    I = state.inactive
    deaths_inactive = I * b(q_ins)
    stay_inactive = I - deaths_inactive
    np.clip(stay_inactive, 0.0, None, out=stay_inactive)

    inactive_next, inactive_overflow = age_shift(stay_inactive)
    leavers_next, leavers_overflow = age_shift(to_inactive)
    inactive_next += leavers_next

    # Members who age past the top insurable age must settle immediately.
    forced_exit = np.zeros_like(inactive_next)
    forced_exit[:, :, -1, :] = active_overflow + inactive_overflow + leavers_overflow

    # -- 4. Old-age claims from the (now aged) inactive pool ------------------
    # ILO-PENS InsIns() step 4: "If the sum is positive, then consider
    # transition to old-age pension (link: InsRet())."
    #
    # Claiming is gated on the normal retirement age of the DESTINATION year.
    # Without this gate, raising the NRA in a reform scenario would push members
    # below the new age out of the scheme on a grant rather than leaving them
    # deferred until they qualify -- which would wildly overstate the saving.
    nra_next = scheme.nra_for_year(min(t + 1, last))
    can_claim = (INS_AGE_GRID >= nra_next)[None, :]
    retire_rate = np.where(can_claim, retire_rate, 0.0)

    claimable = inactive_next + forced_exit
    retirement_claims = claimable * b(retire_rate)
    # Anyone forced out at the top insurable age settles in full, whatever the rate.
    retirement_claims[:, :, -1, :] = claimable[:, :, -1, :]
    inactive_next = claimable - retirement_claims
    np.clip(inactive_next, 0.0, None, out=inactive_next)

    # -- 5. Close the gap to the coverage target ------------------------------
    current_by_age = active_next.sum(axis=(1, 3))                   # (2, 55)
    gap = np.asarray(target_active_by_age, float) - current_by_age

    new_entrants = np.zeros_like(active_next)

    # --- 5a. Growing scheme: D >= 0 -----------------------------------------
    inflow = np.maximum(gap, 0.0)
    want_new = inflow * nr_share[None, :]
    want_reentry = inflow - want_new

    available = inactive_next * b(reentry_cap)
    available_total = available.sum(axis=(1, 3))
    take = np.minimum(want_reentry, available_total)
    frac = np.divide(take, available_total, out=np.zeros_like(take),
                     where=available_total > 0)
    reentrants = available * b(frac)
    inactive_next -= reentrants

    # Re-entry demand the inactive pool cannot satisfy becomes new entry
    # (Technical Guide II.1.1.2: "Act(x+1,t+1) would need to be adjusted").
    shortfall = want_reentry - take
    new_total = want_new + shortfall
    new_entrants[:, :, :, 0] = new_total[:, None, :] * inc_shares[None, :, None]

    # --- 5b. Mature scheme: D < 0 -- push the surplus to inactive -------------
    outflow = np.maximum(-gap, 0.0)
    surplus_frac = np.divide(outflow, current_by_age, out=np.zeros_like(outflow),
                             where=current_by_age > 0)
    surplus_frac = np.clip(surplus_frac, 0.0, 1.0)
    surplus = active_next * b(surplus_frac)
    active_next -= surplus
    inactive_next += surplus

    active_next = active_next + new_entrants + reentrants
    np.clip(active_next, 0.0, None, out=active_next)
    np.clip(inactive_next, 0.0, None, out=inactive_next)

    flows = ActiveFlows(
        deaths_active=deaths_active,
        deaths_inactive=deaths_inactive,
        retirement_claims=retirement_claims,
        new_entrants=new_entrants,
        reentrants=reentrants,
        to_inactive=to_inactive + surplus,
        contributors=A * b(density),
    )
    return ActiveState(active=active_next, inactive=inactive_next), flows


def coverage_target(cfg: Config, employed: np.ndarray, t: int) -> np.ndarray:
    """Required active head count at year ``t`` by sex and insurable age.

    ``Act(x,t) = Covrate(x,t) . Pop(x,t)`` -- Technical Guide II.1.1.2, with the
    employed population as the base population (the SSO covers formal employees).

    Parameters
    ----------
    cfg
        Fully populated configuration.
    employed
        Employed population by sex, age 0..100 and year, shape ``(2, 101, n_t)``.
    t
        Projection year index.

    Returns
    -------
    np.ndarray
        Shape ``(2, 55)``, persons.
    """
    cov = np.asarray(cfg.membership.coverage_rate, float)[:, :, t]
    emp = np.asarray(employed, float)[:, AGE_MIN_INS : AGE_MAX_INS + 1, t]
    return cov * emp


__all__ = [
    "ActiveState",
    "ActiveFlows",
    "accrue_credit",
    "age_shift",
    "step_active_population",
    "coverage_target",
    "CREDIT_GRID",
    "INS_AGE_GRID",
]
