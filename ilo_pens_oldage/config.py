"""
ILO-PENS (Thailand) -- OLD-AGE BRANCH ONLY -- configuration and input placeholders.
====================================================================================

This module is the SINGLE entry point for every exogenous input the model
consumes.  Nothing below this file contains a hard-coded national statistic.
Every item that must be replaced with a real national dataset is flagged::

    # [DATA INPUT REQUIRED: <description> | <expected shape> | <unit>]

Scope
-----
**Old-Age Benefit Branch only.**  The model values exactly two benefits:

* the **Old-Age Monthly Pension** (บำนาญชราภาพ) -- payable from the normal
  retirement age once 180 months of contributions have been made;
* the **Old-Age Lump-Sum Grant** (บำเหน็จชราภาพ) -- a refund of contributions
  payable when the 180-month vesting condition is *not* met.

Invalidity, survivors', orphans' and all short-term benefits (sickness,
maternity, child allowance, funeral, unemployment) are **out of scope**.  They
are not toggled off -- the code paths do not exist.  Contribution income should
therefore be entered net of the branches you are not valuing: the SSO's 7%
"two benefits type" rate funds old-age *and* child allowance, so the old-age
share is roughly 6.35% once the child-allowance cost is removed.

One benefit sits on the scope boundary and IS included, because it is an
old-age lump sum funded from this branch:

* the refund of accumulated contributions on the **death of a member before
  vesting**.  Excluding it would silently understate branch expenditure.
  Controlled by :attr:`SchemeConfig.pay_grant_on_pre_retirement_death`.

The funeral grant is *not* included -- that is the Death benefits branch.

Index conventions
-----------------
====================  ==========================================================
Symbol                Meaning
====================  ==========================================================
``s``                 Sex index.  0 = male, 1 = female.  (``N_SEX = 2``)
``j``                 Income band (ILO-PENS ``JKC``).  0 = low (bottom 30th
                      percentile of insurable earnings), 1 = medium (30th-70th),
                      2 = high (top 30th).  (``N_INC = 3``)
``a``                 Single year of age in the NATIONAL population, 0..100.
                      Age 100 is an open-ended absorbing group.
``x``                 Single year of age in the INSURED population,
                      ``AGE_MIN_INS``..``AGE_MAX_INS`` (default 15..69).
                      Array position is ``x - AGE_MIN_INS``.
``c``                 Accumulated contribution credit in completed years,
                      0..``CREDIT_MAX``.  Fractional accrual (density < 1) is
                      handled by linear redistribution between ``c`` and ``c+1``.
``t``                 Projection year index, 0 = base year.
====================  ==========================================================

All monetary inputs are **THB per month** (matching SSO reporting practice) and
are converted internally to **THB per year**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 0. Structural constants (array geometry)
# ---------------------------------------------------------------------------

N_SEX: int = 2  # 0 = male, 1 = female
N_INC: int = 3  # 0 = low, 1 = medium, 2 = high income band (ILO-PENS JKC)

SEX_LABELS = ("male", "female")
INCOME_LABELS = ("low", "medium", "high")

AGE_MIN_POP: int = 0
AGE_MAX_POP: int = 100  # open-ended absorbing group
N_AGE_POP: int = AGE_MAX_POP - AGE_MIN_POP + 1  # 101

AGE_MIN_INS: int = 15  # ILO-PENS "Jmin": minimum age of coverage
AGE_MAX_INS: int = 69  # ILO-PENS "Jmax": maximum age of coverage
N_AGE_INS: int = AGE_MAX_INS - AGE_MIN_INS + 1  # 55

CREDIT_MAX: int = 50  # maximum accumulated credit tracked, in years
N_CREDIT: int = CREDIT_MAX + 1  # 51

MONTHS_PER_YEAR: int = 12


def ins_slice() -> slice:
    """Slice selecting insurable ages out of a full 0..100 population axis."""
    return slice(AGE_MIN_INS, AGE_MAX_INS + 1)


# ---------------------------------------------------------------------------
# 1. Run control
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Top-level projection control (ILO-PENS ``InputS`` worksheet)."""

    title: str = "Thailand SSO -- Old-Age Benefit Branch actuarial valuation"

    base_year: int = 2024
    # [DATA INPUT REQUIRED: Valuation (base) year | int | calendar year]
    # Membership data are taken as at 31 December of this year.

    n_years: int = 100
    # ILO-PENS caps the projection at 100 years; the 2013 Thai valuation used
    # 2014-2113.

    salary_history_years: int = 5
    # Pre-base-year salary years needed to compute the reference salary (FINSAL)
    # in the first projection years.  Must be >= ``final_salary_years``.

    random_seed: int = 20240101  # used only by the dummy-data generators

    def __post_init__(self) -> None:
        if not 1 <= self.n_years <= 100:
            raise ValueError("n_years must be between 1 and 100 (ILO-PENS limit).")

    @property
    def years(self) -> np.ndarray:
        """Calendar years covered by the projection, shape ``(n_years + 1,)``."""
        return np.arange(self.base_year, self.base_year + self.n_years + 1)

    @property
    def n_t(self) -> int:
        """Number of projection points including the base year."""
        return self.n_years + 1


# ---------------------------------------------------------------------------
# 2. Module 1 -- Demographic assumptions
# ---------------------------------------------------------------------------


@dataclass
class DemographyConfig:
    """Inputs to the cohort-component national population projection.

    The recursion implemented in :mod:`ilo_pens_oldage.demography` is

    .. math::
        P_{x+1,t+1} = P_{x,t}\\,(1 - q_{x,t}) + M_{x,t}

    with births added at age 0 from age-specific fertility rates.
    """

    base_population: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: National population by sex and single year of age at
    #  31 Dec of the base year | ndarray (2, 101) -> [sex, age 0..100] | persons]
    # Thai source: NESDC population projection, or NSO census / registration.
    # Disaggregate 5-year groups (Sprague / Beers) BEFORE loading.

    mortality_qx: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Probability a person of exact age a in year t dies
    #  before age a+1, GENERAL population | ndarray (2, 101, n_years+1) |
    #  probability 0-1]
    # Thai source: NESDC / UN WPP medium variant; OIC Thai Mortality Table (TMO).

    mortality_improvement: float = 0.010
    # [DATA INPUT REQUIRED: Annual rate of mortality improvement | float |
    #  proportion per year]
    # Used only by the dummy generator when a full (2, 101, T) surface is not
    # supplied.  The 2013 ILO Thailand valuation followed the UN "medium"
    # pattern (male e0 71.1 -> 82.8 over 2013-2088).

    fertility_asfr_base: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Age-specific fertility rates in the base year |
    #  ndarray (101,), non-zero at ages 15-49 | births per woman per year]
    # Thai source: NSO vital registration.  Only the SHAPE matters -- the
    # profile is rescaled each year to hit ``tfr_path``.

    tfr_path: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Total fertility rate by projection year |
    #  ndarray (n_years+1,) | children per woman]
    # The 2013 ILO valuation assumed 1.60 falling to 1.50 over 20 years.  Recent
    # Thai fertility is far lower (~1.0-1.2), so this MUST be refreshed.

    sex_ratio_at_birth: float = 1.05
    # [DATA INPUT REQUIRED: Male births per female birth | float | ratio]

    net_migration_total: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Net migration (in minus out) by projection year |
    #  ndarray (n_years+1,) | persons per year]
    # The 2013 ILO valuation held this at +40,000 p.a.  Thai source: NSO /
    # Department of Employment migrant worker registry.

    migration_age_profile: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Share of annual net migration by sex and age |
    #  ndarray (2, 101), renormalised to sum to 1 | proportion]


# ---------------------------------------------------------------------------
# 3. Module 2 -- Labour force and macroeconomic assumptions
# ---------------------------------------------------------------------------


@dataclass
class LabourConfig:
    """Labour-force participation, unemployment and the macroeconomic frame."""

    lfpr: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Labour force participation rate by sex, age, year |
    #  ndarray (2, 101, n_years+1) | proportion 0-1]
    # Thai source: NSO Labour Force Survey (LFS).  The 2013 ILO valuation held
    # age-specific rates constant from 2015; the aggregate rate then falls
    # through ageing alone (72% in 2013 -> 55% in 2113).

    unemployment_rate: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Unemployment rate by sex, age, year |
    #  ndarray (2, 101, n_years+1) | proportion 0-1]
    # Thai source: NSO LFS.  2013 aggregate actual 0.7%.

    inflation: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: CPI inflation rate by projection year |
    #  ndarray (n_years+1,) | proportion per year]
    # Thai source: MOC / BOT; long-run anchor = BOT target band midpoint.
    ultimate_inflation: float = 0.02

    real_wage_growth: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Real increase in average insurable earnings by year |
    #  ndarray (n_years+1,) | proportion per year]
    # Equals real labour-productivity growth in the ILO frame.  The 2013 Thai
    # valuation used an ultimate 3.0% p.a.
    ultimate_real_wage_growth: float = 0.020

    real_gdp_growth: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Real GDP growth by projection year |
    #  ndarray (n_years+1,) | proportion per year]
    # Thai source: NESDC national accounts; IMF WEO short term.  Used for
    # reporting ratios (expenditure / GDP), not for the core recursion.
    ultimate_real_gdp_growth: float = 0.020

    base_gdp: float = 18_000_000e6
    # [DATA INPUT REQUIRED: Nominal GDP in the base year | float | THB]
    # Thai source: NESDC.  Used only to express results as % of GDP.

    real_investment_return: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Real rate of return on scheme assets by year |
    #  ndarray (n_years+1,) | proportion per year]
    # Thai source: SSO Investment Bureau portfolio yield; BOT government bond
    # curve.  The 2013 ILO valuation used a constant 2.0% real return.
    ultimate_real_investment_return: float = 0.020


# ---------------------------------------------------------------------------
# 4. Modules 3 and 4 -- Old-age scheme design
# ---------------------------------------------------------------------------

IndexationBasis = Literal["price", "wage", "fixed", "none"]


@dataclass
class SchemeConfig:
    """Statutory parameters of the Old-Age Benefit Branch.

    Defaults reproduce the Thai SSO Article 33/39 old-age package as summarised
    in ILO/TF/THAILAND/R.40 Appendix 1.  Values marked *statutory* are law, not
    statistics, but must still be checked against the SSO Act and ministerial
    regulations in force at your valuation date.
    """

    # -- Contribution base --------------------------------------------------
    min_insurable_earnings_monthly: float = 1_650.0
    # Statutory (SSO). THB/month floor on the contribution base.
    max_insurable_earnings_monthly: float = 15_000.0
    # Statutory (SSO). THB/month ceiling. A long-standing policy debate; the
    # ILO report models a rise to THB 20,000 as a reform scenario.

    ceiling_indexation: IndexationBasis = "wage"
    # [DATA INPUT REQUIRED: Basis for future uprating of the earnings ceiling |
    #  "price" / "wage" / "fixed" / "none" | -]
    # The 2013 ILO valuation assumed the ceiling follows salary increases.
    # "none" (a frozen ceiling) is the realistic Thai status quo and materially
    # raises the PAYG cost rate -- run it as a scenario.
    floor_indexation: IndexationBasis = "wage"

    # -- Contribution rate ---------------------------------------------------
    contribution_rate_total: float = 0.0635
    # [DATA INPUT REQUIRED: Contribution rate allocated to the OLD-AGE branch |
    #  float or ndarray (n_years+1,) | proportion]
    # NOTE the scope adjustment.  The SSO's statutory 7% (employer 3% +
    # employee 3% + government 1%) funds the "two benefits type" branch, which
    # is old-age pension AND child allowance.  ILO/TF/THAILAND/R.40 assesses
    # the child-allowance cost at 0.65%, leaving about 6.35% for old age.
    # Enter the old-age share here, not the full 7%.

    contribution_rate_employee: float = 0.03
    contribution_rate_employer: float = 0.03
    contribution_rate_government: float = 0.01
    # Statutory split of the full branch rate.  Used only for the grant refund
    # rule below, which distinguishes the employee's own contributions from the
    # employer's.

    collection_rate: float = 0.98
    # [DATA INPUT REQUIRED: Contribution collection / compliance rate |
    #  float or ndarray (n_years+1,) | proportion 0-1]
    # Contributions actually collected / contributions legally due.
    # Thai source: SSO financial statements vs. declared insurable earnings.

    # -- Retirement ----------------------------------------------------------
    normal_retirement_age: int = 55
    # Statutory (SSO). Low by international standards; the ILO report models a
    # phased increase to 60/65 as the primary reform lever.

    nra_path: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Normal retirement age by projection year, for a
    #  phased reform (OPTIONAL) | ndarray (n_years+1,) dtype int | age in years]
    # Leave None to hold ``normal_retirement_age`` constant.

    max_contributory_age: int = 60
    # Statutory: new registration is not permitted past 60, but an existing
    # member may continue contributing after 60.

    # -- Old-age pension: eligibility and formula -----------------------------
    vesting_months: int = 180
    # Statutory (SSO): 180 months (15 years) of contributions for a pension.

    accrual_base_rate: float = 0.20
    # Statutory: 20% of the reference salary for the first 180 months.
    accrual_increment_rate: float = 0.015
    # Statutory: +1.5% for each further 12 months of contribution beyond 180.

    final_salary_years: int = 5
    # Statutory: reference salary = average of the final 60 months of wages.

    revalue_past_salaries: bool = False
    # [DATA INPUT REQUIRED: Whether wages in the averaging window are revalued
    #  to the retirement date | bool | -]
    # The SSO takes a NOMINAL 60-month average with no revaluation -- a real
    # benefit erosion when inflation is high.  Set True to test the reform.

    pension_indexation: IndexationBasis = "price"
    # [DATA INPUT REQUIRED: Indexation of pensions in payment |
    #  "price" / "wage" / "fixed" / "none" | -]
    # The SSO Act contains NO statutory cost-of-living adjustment.  The 2013 ILO
    # valuation nonetheless assumed price indexation, arguing a scheme without
    # one would not be viable.  Run "none" as a scenario to size the gap.

    pension_indexation_fixed_rate: float = 0.0
    # Used only when ``pension_indexation == "fixed"``.

    min_pension_monthly: Optional[float] = None
    # [DATA INPUT REQUIRED: Statutory minimum old-age pension | float or None |
    #  THB/month at base-year prices]
    # None = no minimum (the current SSO position).  The ILO report models
    # introducing one as a reform option.
    max_pension_monthly: Optional[float] = None
    # [DATA INPUT REQUIRED: Statutory maximum pension | float or None |
    #  THB/month]

    # -- Old-age lump-sum grant (below the vesting threshold) ------------------
    grant_min_months_for_employer_share: int = 12
    # Statutory: below 12 months of contributions the grant refunds only the
    # INSURED PERSON'S contributions; from 12 to 179 months it refunds the
    # employee + employer contributions accumulated at the investment return.

    pay_grant_on_pre_retirement_death: bool = True
    # Statutory, and IN SCOPE for this branch: when a member dies before
    # vesting, the accumulated contributions are refunded as a lump sum under
    # the old-age formula.  This is an old-age lump-sum outflow, not a
    # survivors' pension, so it belongs in this valuation.  Setting False
    # removes it, which will UNDERSTATE branch expenditure -- do that only if
    # your terms of reference explicitly exclude it.

    # -- Death of a pensioner inside the guarantee period ---------------------
    pensioner_death_guarantee_months: int = 60
    pensioner_death_lump_sum_multiple: float = 10.0
    # Statutory: if an old-age pensioner dies within the first 60 months of
    # receiving the pension, survivors receive a lump sum of 10x the monthly
    # pension.  An old-age branch liability, so in scope.

    # -- Administration -------------------------------------------------------
    admin_expense_rate: float = 0.0047
    # [DATA INPUT REQUIRED: Administrative and other expenditure as a share of
    #  total insurable earnings | float or ndarray (n_years+1,) | proportion]
    # 0.47% is the 2009-13 SSO all-branch average used in ILO/TF/THAILAND/R.40
    # section 3.  If your terms of reference allocate administration by branch,
    # enter the old-age share instead.

    def nra_for_year(self, t: int) -> int:
        """Normal retirement age applicable in projection year ``t``."""
        if self.nra_path is not None:
            return int(self.nra_path[t])
        return int(self.normal_retirement_age)

    @property
    def vesting_years(self) -> float:
        """Vesting threshold expressed in years of credit."""
        return self.vesting_months / MONTHS_PER_YEAR


# ---------------------------------------------------------------------------
# 5. Module 3 -- Insured population (scheme membership)
# ---------------------------------------------------------------------------


@dataclass
class MembershipConfig:
    """Base-year membership stocks and the transition-rate surfaces.

    Note the absence of an invalidity incidence rate: in this old-age-only
    model the active population is decremented by **mortality and withdrawal
    only**.
    """

    # -- Base-year stocks -----------------------------------------------------
    active_base: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Active members (at least one contribution in the
    #  base year) by sex, income band, age and accumulated credit |
    #  ndarray (2, 3, 55, 51) -> [sex, income, age 15-69, credit 0-50] | persons]
    # Thai source: SSO member register.  If you hold only the marginals --
    # members by age x sex, and average past service by age x sex (the shape of
    # ILO/TF/THAILAND/R.40 Tables A2.6 and A2.13) -- use
    # ``dummy_data.build_credit_distribution`` to spread them over the credit
    # axis; it reproduces your stated mean exactly.

    inactive_base: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Inactive (registered, non-contributing) members by
    #  sex, income, age, credit | ndarray (2, 3, 55, 51) | persons]
    # Thai source: SSO register -- members with contributions in the last 15
    # years but none in the base year (Table A2.14).

    old_age_pensioners_base: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Old-age pensioners in payment at the base date, by
    #  sex and age | ndarray (2, 101) | persons]
    old_age_pension_base_monthly: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Average monthly old-age pension in payment by sex
    #  and age | ndarray (2, 101) | THB/month]
    # Thai source: SSO benefit register.  The SSO paid its first old-age
    # pensions in 2014, so the stock is young and growing fast.

    # -- Coverage --------------------------------------------------------------
    coverage_rate: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Insured (active) persons as a share of the EMPLOYED
    #  population, by sex, insurable age and year |
    #  ndarray (2, 55, n_years+1) | proportion 0-1]
    # ILO-PENS ``Covrate(x,t)``; Act(x,t) = Covrate(x,t) * Pop(x,t).
    # Thai source: SSO insured count (Art. 33 + 39) / NSO LFS employed.
    # The 2013 ILO valuation had coverage rising over 40 years then flat.

    # -- Transition rates -------------------------------------------------------
    density: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Contribution density -- contributors / active
    #  members, by sex, insurable age and year | ndarray (2, 55, n_years+1) |
    #  proportion 0-1]
    # ILO-PENS ``Dens(x,t)``.  Drives BOTH the contribution base and the speed
    # of credit accrual, making it one of the highest-leverage inputs in the
    # model.  Thai source: SSO contribution records.
    # Table A2.12: ~50% at 15-19 rising to ~93-95% at 50+.

    leaving_rate: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Probability an active member becomes inactive during
    #  the year (withdrawal + exit), by sex, insurable age, year |
    #  ndarray (2, 55, n_years+1) | probability 0-1]
    # ILO-PENS ``Leaving`` / ``Inactiverate(x,t)``.

    retirement_rate: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Share of those leaving the active population who
    #  claim an old-age benefit, by sex, insurable age, year |
    #  ndarray (2, 55, n_years+1) | proportion 0-1]
    # ILO-PENS: RACT = DOUT * Retrate(x,t).  Should be ~0 below the NRA (the SSO
    # does not permit early retirement) and ~1 at and above it.  The engine also
    # gates claiming on the NRA independently, so a stray non-zero rate below
    # the NRA cannot leak benefits.

    reentry_rate: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Probability an inactive member re-enters the active
    #  population, by sex, insurable age, year | ndarray (2, 55, n_years+1) |
    #  probability 0-1]
    # ILO-PENS ``REent``.  Acts as the availability cap on re-entrants when the
    # active population must grow to meet the coverage target.

    insured_mortality_qx: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Mortality of the INSURED population by sex, age,
    #  year | ndarray (2, 101, n_years+1) | probability 0-1]
    # Materially lighter than general-population mortality (selection effect).
    # ILO/TF/THAILAND/R.40 Tables A2.8 / A2.9.

    new_entrant_share: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Share of the required inflow into the active
    #  population that is NEW entrants rather than re-entrants, by insurable age
    #  and year | ndarray (55, n_years+1) | proportion 0-1]
    # ILO-PENS ``NR(x,t)``: Nent = NR * D; Rent = (1 - NR) * D.

    # -- Earnings ----------------------------------------------------------------
    salary_scale_monthly: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Average monthly insurable earnings by sex, income
    #  band and age in the BASE year | ndarray (2, 3, 55) | THB/month]
    # ILO-PENS ``SalL`` / ``SalM`` / ``SalH``: the mean of the bottom-30 /
    # middle-40 / top-30 percentile bands of the earnings distribution.
    # Enter GROSS wages -- the statutory floor and ceiling are applied by the
    # model.  Thai source: SSO declared wages (Table A2.11 gives the all-band
    # average by age).

    income_level_shares: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Share of members in each income band |
    #  ndarray (3,), sums to 1 | proportion]
    # Structurally 0.30 / 0.40 / 0.30 under the percentile definition above.


# ---------------------------------------------------------------------------
# 6. Module 5 -- Fund and financing
# ---------------------------------------------------------------------------


@dataclass
class FinanceConfig:
    """Fund opening position and the financing-system indicators to report."""

    initial_reserve: float = 2_400_000e6
    # [DATA INPUT REQUIRED: Reserve of the OLD-AGE branch at the base date |
    #  float | THB]
    # Thai source: SSO audited financial statements.  If the statements report
    # the two-benefits-type branch as a whole, net off the child-allowance
    # share before entering it here.

    nominal_interest_rate: Optional[np.ndarray] = None
    # [DATA INPUT REQUIRED: Nominal rate of return on the reserve by year |
    #  ndarray (n_years+1,) | proportion per year]
    # Built from (1 + real return)(1 + inflation) - 1 when left None.

    gap_horizon_years: Optional[int] = None
    # Horizon for the General Average Premium / level contribution rate.
    # None = the full projection period.  The 2013 ILO valuation used 100 years
    # and obtained a GAP of 21.1% for the SSO old-age branch.

    target_reserve_ratio: float = 3.0
    # [DATA INPUT REQUIRED: Target reserve ratio a0 = F(t-1) / S(t) at the end of
    #  the funding horizon | float | multiple of annual expenditure]

    target_balance_ratio: float = 1.0
    # Target balance ratio b0 = (S_t - P_t) / R_t.  b0 = 1 reproduces Thullen's
    # scaled premium (zero change in the reserve at the end of the period).


# ---------------------------------------------------------------------------
# 7. Master configuration object
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Container aggregating every configuration block."""

    run: RunConfig = field(default_factory=RunConfig)
    demography: DemographyConfig = field(default_factory=DemographyConfig)
    labour: LabourConfig = field(default_factory=LabourConfig)
    scheme: SchemeConfig = field(default_factory=SchemeConfig)
    membership: MembershipConfig = field(default_factory=MembershipConfig)
    finance: FinanceConfig = field(default_factory=FinanceConfig)

    scenario_name: str = "Baseline"

    def copy(self) -> "Config":
        """Deep copy suitable for scenario perturbation."""
        import copy as _copy

        return _copy.deepcopy(self)

    def with_overrides(self, **blocks: Dict[str, object]) -> "Config":
        """Return a copy with dataclass fields replaced block by block.

        Example
        -------
        >>> cfg2 = cfg.with_overrides(scheme={"normal_retirement_age": 60})
        """
        new = self.copy()
        for block_name, updates in blocks.items():
            if not hasattr(new, block_name):
                raise KeyError(f"Unknown configuration block: {block_name!r}")
            block = getattr(new, block_name)
            for key, value in updates.items():  # type: ignore[union-attr]
                if not hasattr(block, key):
                    raise KeyError(f"Unknown field {key!r} in block {block_name!r}")
                setattr(block, key, value)
        return new


# ---------------------------------------------------------------------------
# 8. Helpers
# ---------------------------------------------------------------------------


def rates_to_index(rates: np.ndarray, base_value: float = 1.0) -> np.ndarray:
    """Convert a series of annual growth rates into a cumulative index.

    ``index[0] = base_value`` and ``index[t] = index[t-1] * (1 + rates[t])``.
    ``rates[0]`` is ignored (there is no growth into the base year).

    Parameters
    ----------
    rates
        Annual growth rates, shape ``(T,)``.
    base_value
        Value of the index in the base year.

    Returns
    -------
    np.ndarray
        Cumulative index, shape ``(T,)``.
    """
    rates = np.asarray(rates, dtype=float)
    index = np.empty_like(rates)
    index[0] = base_value
    if rates.size > 1:
        index[1:] = base_value * np.cumprod(1.0 + rates[1:])
    return index


def indexation_series(
    basis: IndexationBasis,
    inflation: np.ndarray,
    wage_growth: np.ndarray,
    fixed_rate: float = 0.0,
) -> np.ndarray:
    """Return the annual uprating rate implied by an indexation basis.

    Parameters
    ----------
    basis
        ``"price"``, ``"wage"``, ``"fixed"`` or ``"none"``.
    inflation
        Nominal CPI inflation by year, shape ``(T,)``.
    wage_growth
        Nominal wage growth by year, shape ``(T,)``.
    fixed_rate
        Constant rate applied when ``basis == "fixed"``.

    Returns
    -------
    np.ndarray
        Uprating rate by year, shape ``(T,)``.
    """
    if basis == "price":
        return np.asarray(inflation, dtype=float)
    if basis == "wage":
        return np.asarray(wage_growth, dtype=float)
    if basis == "fixed":
        return np.full_like(np.asarray(inflation, dtype=float), float(fixed_rate))
    if basis == "none":
        return np.zeros_like(np.asarray(inflation, dtype=float))
    raise ValueError(f"Unknown indexation basis: {basis!r}")


__all__ = [
    "N_SEX", "N_INC", "SEX_LABELS", "INCOME_LABELS",
    "AGE_MIN_POP", "AGE_MAX_POP", "N_AGE_POP",
    "AGE_MIN_INS", "AGE_MAX_INS", "N_AGE_INS",
    "CREDIT_MAX", "N_CREDIT", "MONTHS_PER_YEAR", "ins_slice",
    "RunConfig", "DemographyConfig", "LabourConfig", "SchemeConfig",
    "MembershipConfig", "FinanceConfig", "Config",
    "rates_to_index", "indexation_series", "IndexationBasis",
]
