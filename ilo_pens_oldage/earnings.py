"""
Earnings surface, reference salary (FINS) and accumulated contributions (BAL).
==============================================================================

This module builds the exogenous salary surface that Modules 3-5 consume, and
implements the two ILO-PENS financial sub-modules that translate a member's
earnings history into a benefit base:

``FINSAL()`` -- reference (final average) salary
    Technical Guide, Part III section 3.3(3):

    .. math::
        \\mathrm{FINS}(I, JKC) = \\frac{1}{IE}\\sum_{J=1}^{IE}
            \\mathrm{SAL}(X - J, T - J, JKC)\\;\\mathrm{adj}(T, J)

    with :math:`IE = \\min\\{k,\\;T+5,\\;X-15\\}` and

    .. math::
        \\mathrm{adj}(T, J) = \\begin{cases}
            1 & \\text{past salaries not revalued} \\\\
            \\mathrm{ARINFS}(T-1) / \\mathrm{ARINFS}(T-J) & \\text{revalued}
        \\end{cases}

    The Thai SSO takes a **nominal** 60-month average with no revaluation, i.e.
    ``adj = 1``.

``BALANCE()`` -- accumulated value of contributions
    Technical Guide, Part III section 3.3(3):

    .. math::
        \\mathrm{BAL}(I, JKC) = \\sum_{J=1}^{IE} \\mathrm{SAL}(X-J, T-J, JKC)\\,
            \\mathrm{CONT}(T-J)\\,
            \\frac{\\mathrm{ARINT}(T-J-1)\\,\\mathrm{RINT}(T-J)^{1/2}}
                 {\\mathrm{ARINT}(T - IE)}

    In this implementation the accumulation is carried forward to the *current*
    year rather than to ``T - IE``, which is what the old-age grant (refund of
    contributions with interest) actually requires.  The computation is
    vectorised with a diagonal cumulative sum, so it is O(1) per cell rather
    than O(credit).

Pre-base-year history
---------------------
Both formulas look back along a member's career diagonal.  A member aged 69
with 50 years of credit in the base year contributed as far back as ``t = -50``.
The surface is therefore extended :data:`HISTORY_YEARS` years before the base
year using the pre-base wage growth assumption; this only affects the grant
values of the oldest base-year members and washes out within a few decades.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    AGE_MIN_INS,
    Config,
    MONTHS_PER_YEAR,
    N_AGE_INS,
    N_CREDIT,
    N_INC,
    N_SEX,
    indexation_series,
    rates_to_index,
)

HISTORY_YEARS: int = N_AGE_INS + 10
"""Number of pre-base-year columns carried in the earnings surface."""


def _as_series(value, n_t: int) -> np.ndarray:
    """Broadcast a scalar or array into a length-``n_t`` float series."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_t, float(arr))
    if arr.shape[0] != n_t:
        raise ValueError(f"Expected a series of length {n_t}, got shape {arr.shape}.")
    return arr


@dataclass
class EarningsSurface:
    """Salary, insurable-earnings and benefit-base surfaces.

    All monetary arrays are **annual THB** unless the name says ``_monthly``.

    Attributes
    ----------
    salary
        Gross average earnings by ``[sex, income, insurable age, year]``,
        shape ``(2, 3, 55, n_t)``, annual THB.  Uncapped.
    insurable
        The same, clipped to the statutory floor and ceiling, shape
        ``(2, 3, 55, n_t)``, annual THB.  This is the contribution base per
        contributor and the input to the pension formula.
    floor_annual, ceiling_annual
        Statutory limits by year, shape ``(n_t,)``, annual THB.
    contribution_rate
        Total contribution rate by year, shape ``(n_t,)``.
    interest_index
        Cumulative nominal investment return index, base year = 1, shape
        ``(n_t,)``.
    """

    salary: np.ndarray
    insurable: np.ndarray
    floor_annual: np.ndarray
    ceiling_annual: np.ndarray
    contribution_rate: np.ndarray
    interest_index: np.ndarray

    # -- internal extended-history surfaces (index 0 == t = -HISTORY_YEARS) --
    _ins_ext: np.ndarray = None  # type: ignore[assignment]
    _bal_cum: np.ndarray = None  # type: ignore[assignment]
    _bal_cum_employee: np.ndarray = None  # type: ignore[assignment]
    _arint_ext: np.ndarray = None  # type: ignore[assignment]
    _wage_index_ext: np.ndarray = None  # type: ignore[assignment]
    _final_salary_years: int = 5
    _revalue: bool = False
    _n_t: int = 0

    # ------------------------------------------------------------------
    def reference_salary(self, t: int) -> np.ndarray:
        """Reference (final average) salary for a member retiring from age x in year t.

        Implements ``FINSAL()``.  The averaging window is the last
        ``final_salary_years`` years of the member's career, i.e. ages
        ``x-1 ... x-IE`` in years ``t-1 ... t-IE``.

        Parameters
        ----------
        t
            Projection year index, ``0 <= t < n_t``.

        Returns
        -------
        np.ndarray
            Shape ``(2, 3, 55)`` -> ``[sex, income, insurable age]``, annual THB.
        """
        k = self._final_salary_years
        ti = t + HISTORY_YEARS
        ages = np.arange(N_AGE_INS)

        total = np.zeros((N_SEX, N_INC, N_AGE_INS), dtype=float)
        count = np.zeros(N_AGE_INS, dtype=float)

        for j in range(1, k + 1):
            src_age = ages - j
            valid = src_age >= 0
            if not valid.any():
                break
            adj = 1.0
            if self._revalue:
                adj = self._wage_index_ext[ti - 1] / self._wage_index_ext[ti - j]
            contribution = np.zeros((N_SEX, N_INC, N_AGE_INS), dtype=float)
            contribution[:, :, valid] = self._ins_ext[:, :, src_age[valid], ti - j] * adj
            total += contribution
            count[valid] += 1.0

        # Members younger than the averaging window average over what exists;
        # at age 15 (no history at all) fall back to the current-year salary.
        count = np.where(count > 0, count, 1.0)
        fallback = self._ins_ext[:, :, :, ti]
        fins = np.where(total > 0, total / count[None, None, :], fallback)
        return fins

    # ------------------------------------------------------------------
    def accumulated_contributions(self, t: int, employee_only: bool = False) -> np.ndarray:
        """Accumulated value of past contributions by credit, in year ``t``.

        Implements ``BALANCE()``.  Used for the old-age / invalidity / survivor
        **grant** (lump sum) payable to members who have not met the vesting
        condition.

        Parameters
        ----------
        t
            Projection year index.
        employee_only
            If True, accumulate only the insured person's own share (the SSO
            rule below 12 months of contributions); otherwise accumulate the
            employee + employer share.

        Returns
        -------
        np.ndarray
            Shape ``(2, 3, 55, 51)`` -> ``[sex, income, age, credit]``, THB.
        """
        ti = t + HISTORY_YEARS
        cum = self._bal_cum_employee if employee_only else self._bal_cum
        ages = np.arange(N_AGE_INS)
        credits = np.arange(N_CREDIT)

        # G(x, t) - G(x - c, t - c), scaled forward to year t.
        g_now = cum[:, :, ages, ti]                       # (2, 3, 55)
        src_age = ages[:, None] - credits[None, :]        # (55, 51)
        src_ti = ti - credits[None, :]                    # (1, 51) broadcast
        src_age_clipped = np.clip(src_age, 0, N_AGE_INS - 1)
        src_ti_clipped = np.clip(np.broadcast_to(src_ti, src_age.shape), 0, cum.shape[3] - 1)

        g_then = cum[:, :, src_age_clipped, src_ti_clipped]  # (2, 3, 55, 51)
        invalid = src_age < 0
        # A member cannot have more credit than years since the minimum age;
        # clamp those cells to the full career value.
        g_then = np.where(invalid[None, None, :, :], 0.0, g_then)

        bal = (g_now[:, :, :, None] - g_then) * self._arint_ext[ti]
        return np.maximum(bal, 0.0)


def build_earnings_surface(cfg: Config, labour_wage_growth: np.ndarray,
                           nominal_interest: np.ndarray) -> EarningsSurface:
    """Construct the full earnings and benefit-base surfaces.

    Parameters
    ----------
    cfg
        Fully populated configuration.
    labour_wage_growth
        Nominal wage growth by projection year, shape ``(n_t,)``.
    nominal_interest
        Nominal investment return by projection year, shape ``(n_t,)``.

    Returns
    -------
    EarningsSurface
    """
    run, scheme, mem, lab = cfg.run, cfg.scheme, cfg.membership, cfg.labour
    n_t = run.n_t

    wage_growth = np.asarray(labour_wage_growth, dtype=float)
    interest = np.asarray(nominal_interest, dtype=float)
    inflation = np.asarray(lab.inflation, dtype=float)

    # -- Extended time axis: index 0 == t = -HISTORY_YEARS ------------------
    n_ext = n_t + HISTORY_YEARS
    pre_wage_growth = float(wage_growth[1]) if n_t > 1 else 0.03
    pre_inflation = float(inflation[1]) if n_t > 1 else 0.02
    pre_interest = float(interest[1]) if n_t > 1 else 0.04

    wage_growth_ext = np.concatenate([np.full(HISTORY_YEARS, pre_wage_growth), wage_growth])
    inflation_ext = np.concatenate([np.full(HISTORY_YEARS, pre_inflation), inflation])
    interest_ext = np.concatenate([np.full(HISTORY_YEARS, pre_interest), interest])

    # Wage index anchored so that index == 1 in the base year.
    wage_index_ext = np.empty(n_ext, dtype=float)
    wage_index_ext[HISTORY_YEARS] = 1.0
    for i in range(HISTORY_YEARS + 1, n_ext):
        wage_index_ext[i] = wage_index_ext[i - 1] * (1.0 + wage_growth_ext[i])
    for i in range(HISTORY_YEARS - 1, -1, -1):
        wage_index_ext[i] = wage_index_ext[i + 1] / (1.0 + wage_growth_ext[i + 1])

    price_index_ext = np.empty(n_ext, dtype=float)
    price_index_ext[HISTORY_YEARS] = 1.0
    for i in range(HISTORY_YEARS + 1, n_ext):
        price_index_ext[i] = price_index_ext[i - 1] * (1.0 + inflation_ext[i])
    for i in range(HISTORY_YEARS - 1, -1, -1):
        price_index_ext[i] = price_index_ext[i + 1] / (1.0 + inflation_ext[i + 1])

    # Cumulative interest index ARINT, base year = 1.
    arint_ext = np.empty(n_ext, dtype=float)
    arint_ext[0] = 1.0
    for i in range(1, n_ext):
        arint_ext[i] = arint_ext[i - 1] * (1.0 + interest_ext[i])
    arint_ext /= arint_ext[HISTORY_YEARS]

    # -- Salary surface -----------------------------------------------------
    base_scale_annual = np.asarray(mem.salary_scale_monthly, dtype=float) * MONTHS_PER_YEAR
    salary_ext = base_scale_annual[:, :, :, None] * wage_index_ext[None, None, None, :]

    # -- Statutory floor and ceiling ----------------------------------------
    floor_growth = indexation_series(
        scheme.floor_indexation, inflation_ext, wage_growth_ext, 0.0
    )
    ceiling_growth = indexation_series(
        scheme.ceiling_indexation, inflation_ext, wage_growth_ext, 0.0
    )

    def _limit_index(growth: np.ndarray) -> np.ndarray:
        idx = np.empty(n_ext, dtype=float)
        idx[HISTORY_YEARS] = 1.0
        for i in range(HISTORY_YEARS + 1, n_ext):
            idx[i] = idx[i - 1] * (1.0 + growth[i])
        for i in range(HISTORY_YEARS - 1, -1, -1):
            idx[i] = idx[i + 1] / (1.0 + growth[i + 1])
        return idx

    floor_ext = scheme.min_insurable_earnings_monthly * MONTHS_PER_YEAR * _limit_index(floor_growth)
    ceiling_ext = (
        scheme.max_insurable_earnings_monthly * MONTHS_PER_YEAR * _limit_index(ceiling_growth)
    )

    # A frozen ceiling combined with an indexed floor can invert the two limits
    # over a long projection.  The ceiling is the binding statutory constraint,
    # so clamp the floor to it explicitly rather than relying on clip ordering.
    floor_ext = np.minimum(floor_ext, ceiling_ext)

    ins_ext = np.clip(
        salary_ext, floor_ext[None, None, None, :], ceiling_ext[None, None, None, :]
    )

    # -- Contribution rates --------------------------------------------------
    rate_total = _as_series(scheme.contribution_rate_total, n_t)
    rate_ee = _as_series(scheme.contribution_rate_employee, n_t)
    rate_er = _as_series(scheme.contribution_rate_employer, n_t)

    rate_total_ext = np.concatenate([np.full(HISTORY_YEARS, rate_total[0]), rate_total])
    rate_ee_ext = np.concatenate([np.full(HISTORY_YEARS, rate_ee[0]), rate_ee])
    rate_er_ext = np.concatenate([np.full(HISTORY_YEARS, rate_er[0]), rate_er])

    # -- Diagonal cumulative for BAL ----------------------------------------
    # D(x, t) = S_ins(x, t) * rate(t) / ARINT(t); G(x, t) = G(x-1, t-1) + D(x-1, t-1)
    def _diagonal_cumsum(rate_ext: np.ndarray) -> np.ndarray:
        d = ins_ext * rate_ext[None, None, None, :] / arint_ext[None, None, None, :]
        g = np.zeros_like(d)
        for x in range(1, N_AGE_INS):
            g[:, :, x, 1:] = g[:, :, x - 1, :-1] + d[:, :, x - 1, :-1]
        return g

    bal_cum_all = _diagonal_cumsum(rate_ee_ext + rate_er_ext)
    bal_cum_ee = _diagonal_cumsum(rate_ee_ext)

    surface = EarningsSurface(
        salary=salary_ext[:, :, :, HISTORY_YEARS:],
        insurable=ins_ext[:, :, :, HISTORY_YEARS:],
        floor_annual=floor_ext[HISTORY_YEARS:],
        ceiling_annual=ceiling_ext[HISTORY_YEARS:],
        contribution_rate=rate_total,
        interest_index=arint_ext[HISTORY_YEARS:],
    )
    surface._ins_ext = ins_ext
    surface._bal_cum = bal_cum_all
    surface._bal_cum_employee = bal_cum_ee
    surface._arint_ext = arint_ext
    surface._wage_index_ext = wage_index_ext
    surface._final_salary_years = int(scheme.final_salary_years)
    surface._revalue = bool(scheme.revalue_past_salaries)
    surface._n_t = n_t
    return surface


__all__ = ["EarningsSurface", "build_earnings_surface", "HISTORY_YEARS"]
