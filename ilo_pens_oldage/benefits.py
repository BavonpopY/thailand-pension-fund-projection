"""
Module 4 -- Old-age beneficiaries and benefit calculation.
==========================================================

Implements the old-age subset of the ILO-PENS benefit sub-modules --
``PensionR()``, ``RetPos()``, ``InsRet()`` and ``Retireds()``.  The invalidity,
survivors' and orphans' routines (``PensionI()``, ``PensionS()``, ``InvPos()``,
``SurvPos()``, ``SDIST()``, ``Invalids()``, ``Survivors()``) are out of scope
and absent.

Benefits valued (ILO/TF/THAILAND/R.40 Appendix 1)
--------------------------------------------------
**Old-age monthly pension** -- vesting 180 months, normal retirement age 55:

.. math::
    P^{ret}_{monthly} = \\Bigl[\\,0.20
        + 0.015 \\times \\max(0,\\; C - 15)\\,\\Bigr] \\times \\mathrm{FINS}

where :math:`C` is accumulated credit in years and ``FINS`` is the average wage
over the final 60 months.  This is the ILO-PENS "Example 2" shape
(:math:`P = 0.01\\,(40 + 2\\,TT)\\,FINS`) with the Thai parameters substituted.

**Old-age lump-sum grant** -- payable when credit is below the vesting
threshold, as a refund of accumulated contributions:

* under 12 months of contributions: the insured person's own contributions;
* 12 to 179 months: employee **and** employer contributions, accumulated at the
  fund's investment return.

**Lump sum on pre-retirement death** -- the same refund formula, paid when a
member dies before vesting.  This is an old-age lump-sum outflow funded from
this branch, not a survivors' pension, so it is in scope.  Switch it off with
``scheme.pay_grant_on_pre_retirement_death`` only if your terms of reference
say so; doing that understates branch expenditure.

**Lump sum on death inside the guarantee period** -- if an old-age pensioner
dies within the first 60 months of payment, survivors receive 10x the monthly
pension.  An old-age branch liability, so in scope.

The funeral grant is NOT valued here: it belongs to the Death benefits branch.

Pensioner roll-forward (Technical Guide II.1.1.6)
--------------------------------------------------
.. math::
    \\mathrm{Pens\\#}(x+1,t+1) &= \\mathrm{Pens\\#}(x,t)\\,[1 - q(x,t)]
        + \\mathrm{NPens\\#}(x+1,t+1) \\\\
    \\mathrm{Pens\\$}(x+1,t+1) &= \\mathrm{Pens\\$}(x,t)\\,[1 - q(x,t)]\\,
        [1 + \\mathrm{adj}(t)] + \\mathrm{NPens\\$}(x+1,t+1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actives import ActiveFlows, CREDIT_GRID
from .config import (
    AGE_MAX_INS,
    AGE_MIN_INS,
    Config,
    MONTHS_PER_YEAR,
    N_AGE_POP,
    N_SEX,
)
from .earnings import EarningsSurface


# ---------------------------------------------------------------------------
# Benefit formulas
# ---------------------------------------------------------------------------


def old_age_pension_amount(cfg: Config, reference_salary: np.ndarray) -> np.ndarray:
    """Annual old-age pension by sex, income band, age and credit.

    Implements ``PensionR()``.

    Parameters
    ----------
    cfg
        Fully populated configuration.
    reference_salary
        Final average salary, shape ``(2, 3, 55)``, annual THB -- the output of
        :meth:`~ilo_pens_oldage.earnings.EarningsSurface.reference_salary`.

    Returns
    -------
    np.ndarray
        Shape ``(2, 3, 55, 51)``, annual THB.  Zero where credit is below the
        vesting threshold; those members receive a grant instead.
    """
    s = cfg.scheme
    vesting = s.vesting_years

    excess = np.maximum(CREDIT_GRID - vesting, 0.0)
    rate = s.accrual_base_rate + s.accrual_increment_rate * np.floor(excess)
    rate = np.where(CREDIT_GRID >= vesting, rate, 0.0)

    pension = reference_salary[:, :, :, None] * rate[None, None, None, :]

    if s.min_pension_monthly is not None:
        floor = s.min_pension_monthly * MONTHS_PER_YEAR
        pension = np.where(pension > 0, np.maximum(pension, floor), 0.0)
    if s.max_pension_monthly is not None:
        pension = np.minimum(pension, s.max_pension_monthly * MONTHS_PER_YEAR)
    return pension


def retirement_eligible(cfg: Config, t: int) -> np.ndarray:
    """Old-age pension eligibility mask by insurable age and credit.

    Implements ``RetPos()``: credit at or above the vesting threshold **and**
    age at or above the normal retirement age applicable in year ``t``.

    Returns
    -------
    np.ndarray
        Boolean, shape ``(55, 51)``.
    """
    nra = cfg.scheme.nra_for_year(t)
    ages = np.arange(AGE_MIN_INS, AGE_MAX_INS + 1)
    age_ok = (ages >= nra)[:, None]
    credit_ok = (CREDIT_GRID >= cfg.scheme.vesting_years)[None, :]
    return age_ok & credit_ok


def grant_value(cfg: Config, earnings: EarningsSurface, t: int) -> np.ndarray:
    """Old-age lump-sum grant by sex, income band, age and credit.

    Implements ``BALANCE()`` under the SSO refund rule: below
    ``grant_min_months_for_employer_share`` months of contributions only the
    insured person's own contributions are refunded; above it, the employee and
    employer contributions are refunded with accumulated investment return.

    Parameters
    ----------
    cfg
        Fully populated configuration.
    earnings
        The earnings surface.
    t
        Projection year index.

    Returns
    -------
    np.ndarray
        Shape ``(2, 3, 55, 51)``, THB.
    """
    threshold = cfg.scheme.grant_min_months_for_employer_share / MONTHS_PER_YEAR
    bal_all = earnings.accumulated_contributions(t, employee_only=False)
    bal_ee = earnings.accumulated_contributions(t, employee_only=True)
    employee_only = (CREDIT_GRID < threshold)[None, None, None, :]
    return np.where(employee_only, bal_ee, bal_all)


# ---------------------------------------------------------------------------
# Pensioner state
# ---------------------------------------------------------------------------


@dataclass
class PensionerState:
    """Stock of old-age pensions in payment.

    Counts are ``[sex, age 0..100]`` in persons; amounts are ``[sex, age]`` in
    **total annual THB** for the whole cell (not per head).

    ``recent_n`` / ``recent_amt`` track the cohorts still inside the SSO's
    60-month death guarantee, indexed ``[sex, age, duration 0..G-1]``.
    """

    n: np.ndarray
    amt: np.ndarray
    recent_n: np.ndarray
    recent_amt: np.ndarray

    @classmethod
    def from_config(cls, cfg: Config) -> "PensionerState":
        """Initialise from the base-year old-age pensioner stock."""
        m = cfg.membership
        g = max(1, int(round(cfg.scheme.pensioner_death_guarantee_months
                             / MONTHS_PER_YEAR)))

        n = np.array(m.old_age_pensioners_base, dtype=float, copy=True)
        amt = (np.asarray(m.old_age_pensioners_base, float)
               * np.asarray(m.old_age_pension_base_monthly, float)
               * MONTHS_PER_YEAR)

        # Assume the base-year stock is spread evenly across the guarantee
        # window; refine with SSO award-date data if you have it.
        recent_n = np.repeat((n / g)[:, :, None], g, axis=2)
        recent_amt = np.repeat((amt / g)[:, :, None], g, axis=2)

        return cls(n=n, amt=amt, recent_n=recent_n, recent_amt=recent_amt)

    def copy(self) -> "PensionerState":
        return PensionerState(self.n.copy(), self.amt.copy(),
                              self.recent_n.copy(), self.recent_amt.copy())

    @property
    def total_pensioners(self) -> float:
        return float(self.n.sum())

    @property
    def total_pension_expenditure(self) -> float:
        """Annual old-age pension outflow, THB."""
        return float(self.amt.sum())


@dataclass
class BenefitAwards:
    """Old-age benefits newly awarded in one projection step.

    Pension arrays are ``[sex, age 0..100]``; lump sums are scalars in THB.
    """

    new_pension_n: np.ndarray
    new_pension_amt: np.ndarray

    retirement_grant_total: float = 0.0
    """Old-age lump-sum grants to members who reached the NRA unvested."""
    death_grant_total: float = 0.0
    """Old-age lump-sum refunds on death of a member before vesting."""

    retirement_grant_n: float = 0.0
    death_grant_n: float = 0.0

    @classmethod
    def empty(cls) -> "BenefitAwards":
        z = np.zeros((N_SEX, N_AGE_POP), dtype=float)
        return cls(new_pension_n=z.copy(), new_pension_amt=z.copy())

    @property
    def lump_sum_total(self) -> float:
        return self.retirement_grant_total + self.death_grant_total


# ---------------------------------------------------------------------------
# Award routine
# ---------------------------------------------------------------------------


def _to_pop_ages(values_by_ins_age: np.ndarray) -> np.ndarray:
    """Embed a ``(2, 55)`` insurable-age array into a ``(2, 101)`` population grid."""
    out = np.zeros((N_SEX, N_AGE_POP), dtype=float)
    out[:, AGE_MIN_INS : AGE_MAX_INS + 1] = values_by_ins_age
    return out


def award_benefits(
    cfg: Config,
    flows: ActiveFlows,
    earnings: EarningsSurface,
    t_next: int,
) -> BenefitAwards:
    """Convert one step's membership flows into old-age benefit awards.

    Parameters
    ----------
    cfg
        Fully populated configuration.
    flows
        Output of :func:`~ilo_pens_oldage.actives.step_active_population`.
    earnings
        The earnings surface.
    t_next
        Destination projection year index (``t + 1``).

    Returns
    -------
    BenefitAwards
    """
    s_cfg = cfg.scheme
    awards = BenefitAwards.empty()

    fins = earnings.reference_salary(t_next)                     # (2, 3, 55)
    pension = old_age_pension_amount(cfg, fins)                  # (2, 3, 55, 51)
    eligible = retirement_eligible(cfg, t_next)[None, None, :, :]
    grant = grant_value(cfg, earnings, t_next)

    # -- 1. Members claiming at or after the NRA ---------------------------
    claims = flows.retirement_claims
    pensioned = claims * eligible
    granted = claims - pensioned

    awards.new_pension_n = _to_pop_ages(pensioned.sum(axis=(1, 3)))
    awards.new_pension_amt = _to_pop_ages((pensioned * pension).sum(axis=(1, 3)))
    awards.retirement_grant_total = float((granted * grant).sum())
    awards.retirement_grant_n = float(granted.sum())

    # -- 2. Refund of contributions on pre-retirement death -----------------
    if s_cfg.pay_grant_on_pre_retirement_death:
        deaths = flows.deaths_active + flows.deaths_inactive
        awards.death_grant_total = float((deaths * grant).sum())
        awards.death_grant_n = float(deaths.sum())

    return awards


# ---------------------------------------------------------------------------
# Pensioner roll-forward
# ---------------------------------------------------------------------------


def step_pensioners(
    cfg: Config,
    state: PensionerState,
    awards: BenefitAwards,
    t: int,
    pension_uprating: float,
) -> tuple[PensionerState, float, float]:
    """Roll the pensioner stock forward one year and add the new awards.

    Parameters
    ----------
    cfg
        Fully populated configuration.
    state
        Pensioner state at year ``t``.  Not mutated.
    awards
        New awards generated for year ``t + 1``.
    t
        Source projection year index.
    pension_uprating
        Indexation rate applied to pensions in payment for year ``t + 1``.

    Returns
    -------
    (PensionerState, guarantee_lump_sums, pensioner_deaths)
        The state at ``t + 1``; the lump sums paid on deaths inside the
        guarantee period (THB); and the number of pensioner deaths.
    """
    m, s_cfg = cfg.membership, cfg.scheme
    q = np.asarray(m.insured_mortality_qx, float)[:, :, t]  # (2, 101)
    surv = 1.0 - q

    # -- Main stock ---------------------------------------------------------
    n_next = np.zeros_like(state.n)
    a_next = np.zeros_like(state.amt)
    n_next[:, 1:] = (state.n * surv)[:, :-1]
    a_next[:, 1:] = (state.amt * surv * (1.0 + pension_uprating))[:, :-1]
    n_next[:, -1] += (state.n * surv)[:, -1]
    a_next[:, -1] += (state.amt * surv * (1.0 + pension_uprating))[:, -1]
    pensioner_deaths = float((state.n * q).sum())

    # -- Guarantee-period lump sums ------------------------------------------
    # Deaths among cohorts still inside the guarantee window attract a lump sum
    # of ``multiple`` x the monthly pension.  Because the stored amount is
    # annual, the payout is amt * q * multiple / 12.
    deaths_recent_amt = float((state.recent_amt * q[:, :, None]).sum())
    guarantee_lump = (deaths_recent_amt
                      * s_cfg.pensioner_death_lump_sum_multiple
                      / MONTHS_PER_YEAR)

    recent_n = np.zeros_like(state.recent_n)
    recent_amt = np.zeros_like(state.recent_amt)
    surv3 = surv[:, :, None]
    # Duration d -> d+1; the oldest duration cohort leaves the guarantee window.
    recent_n[:, 1:, 1:] = (state.recent_n * surv3)[:, :-1, :-1]
    recent_amt[:, 1:, 1:] = (state.recent_amt * surv3
                             * (1.0 + pension_uprating))[:, :-1, :-1]
    recent_n[:, 1:, 0] = awards.new_pension_n[:, :-1]
    recent_amt[:, 1:, 0] = awards.new_pension_amt[:, :-1]

    # -- Add new awards ------------------------------------------------------
    n_next = n_next + awards.new_pension_n
    a_next = a_next + awards.new_pension_amt

    new_state = PensionerState(n=n_next, amt=a_next,
                               recent_n=recent_n, recent_amt=recent_amt)
    return new_state, guarantee_lump, pensioner_deaths


__all__ = [
    "PensionerState",
    "BenefitAwards",
    "old_age_pension_amount",
    "retirement_eligible",
    "grant_value",
    "award_benefits",
    "step_pensioners",
]
