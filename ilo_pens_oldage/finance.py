"""
Module 5 -- Financial projection and fund solvency.
===================================================

Implements the ILO-PENS financing model (Technical Guide, Part II section 1.2).

Accounting identities
---------------------
.. math::
    I_t &= P_t + R_t \\\\
    R_t &= \\bigl(\\sqrt{1 + i_t} - 1\\bigr)\\,(P_t - S_t) + i_t\\,F_{t-1} \\\\
    F_t &= F_{t-1} + I_t - S_t \\\\
    P_t &= p_t\\,G_t

with :math:`F` the reserve, :math:`I` total income, :math:`P` contribution
income, :math:`R` investment income, :math:`S` total expenditure, :math:`G` total
insurable earnings, :math:`p` the contribution rate and :math:`i` the nominal
yield.  Net cash flow is assumed to arise uniformly over the year, hence the
half-year interest factor :math:`\\sqrt{1 + i_t} - 1`.

Financing systems and performance indicators
--------------------------------------------
**Pay-as-you-go cost rate** (II.1.2.2(1)):

.. math::
    C^{PAYG}_t = \\frac{S_t}{G_t} = d_t \\times r_t

decomposed into the *system demographic dependency ratio* :math:`d_t`
(pensioners / contributors) and the *system replacement ratio* :math:`r_t`
(average pension / average insurable earnings).

**Level contribution rate / General Average Premium** (II.1.2.2(2)):

.. math::
    C^{Level}_{[n,m]} = \\frac{\\bar S_m - \\bar S_{n-1} - F_{n-1} V_{n-1}}
                             {\\bar G_m - \\bar G_{n-1}}

where :math:`V_t = \\prod_{k=1}^{t} v_k`, :math:`W_t = V_{t-1} v_t^{1/2}`,
:math:`\\bar G_t = \\sum_{k=1}^{t} G_k W_k` and
:math:`\\bar S_t = \\sum_{k=1}^{t} S_k W_k`.

**Contribution rate holding a target reserve ratio** :math:`a_0 = F_{t-1}/S_t`
(II.1.2.2(3)):

.. math::
    C(a = a_0; n, m) = \\frac{a_0 V_{m-1} S_m - V_{n-1} F_{n-1}
                              + (\\bar S_{m-1} - \\bar S_{n-1})}
                             {\\bar G_{m-1} - \\bar G_{n-1}}

**Contribution rate holding a target balance ratio** :math:`b_0 = (S_t-P_t)/R_t`
(II.1.2.2(4)); :math:`b_0 = 1` reproduces Thullen's scaled premium.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Cash-flow container
# ---------------------------------------------------------------------------


@dataclass
class FinancialProjection:
    """Year-by-year financial results.  All arrays have shape ``(n_t,)``."""

    years: np.ndarray

    insurable_earnings: np.ndarray      # G_t, THB
    contribution_income: np.ndarray     # P_t, THB
    investment_income: np.ndarray       # R_t, THB
    total_income: np.ndarray            # I_t, THB

    pension_expenditure: np.ndarray     # THB
    lump_sum_expenditure: np.ndarray    # THB
    admin_expenditure: np.ndarray       # THB
    total_expenditure: np.ndarray       # S_t, THB

    reserve: np.ndarray                 # F_t, THB
    contribution_rate: np.ndarray       # p_t

    payg_rate: np.ndarray               # S_t / G_t
    reserve_ratio: np.ndarray           # F_{t-1} / S_t
    reserve_to_expenditure: np.ndarray  # F_t / S_t
    balance_ratio: np.ndarray           # (S_t - P_t) / R_t

    dependency_ratio: np.ndarray        # pensioners / contributors
    replacement_ratio: np.ndarray       # avg pension / avg insurable earnings

    expenditure_pct_gdp: np.ndarray
    reserve_pct_gdp: np.ndarray

    depletion_year: Optional[int] = None
    """First calendar year in which the reserve turns negative, or ``None``."""

    # -- Summary indicators -------------------------------------------------
    general_average_premium: float = float("nan")
    rate_for_target_reserve_ratio: float = float("nan")
    rate_for_target_balance_ratio: float = float("nan")

    def to_frame(self):
        """Return the projection as a :class:`pandas.DataFrame` indexed by year."""
        import pandas as pd

        data = {
            "insurable_earnings": self.insurable_earnings,
            "contribution_income": self.contribution_income,
            "investment_income": self.investment_income,
            "total_income": self.total_income,
            "pension_expenditure": self.pension_expenditure,
            "lump_sum_expenditure": self.lump_sum_expenditure,
            "admin_expenditure": self.admin_expenditure,
            "total_expenditure": self.total_expenditure,
            "reserve": self.reserve,
            "contribution_rate": self.contribution_rate,
            "payg_rate": self.payg_rate,
            "reserve_ratio": self.reserve_ratio,
            "reserve_to_expenditure": self.reserve_to_expenditure,
            "balance_ratio": self.balance_ratio,
            "dependency_ratio": self.dependency_ratio,
            "replacement_ratio": self.replacement_ratio,
            "expenditure_pct_gdp": self.expenditure_pct_gdp,
            "reserve_pct_gdp": self.reserve_pct_gdp,
        }
        return pd.DataFrame(data, index=pd.Index(self.years, name="year"))


# ---------------------------------------------------------------------------
# Core recursion
# ---------------------------------------------------------------------------


def project_fund(
    years: np.ndarray,
    insurable_earnings: np.ndarray,
    pension_expenditure: np.ndarray,
    lump_sum_expenditure: np.ndarray,
    admin_expenditure: np.ndarray,
    contribution_rate: np.ndarray,
    collection_rate: np.ndarray,
    nominal_interest: np.ndarray,
    initial_reserve: float,
    n_pensioners: np.ndarray,
    n_contributors: np.ndarray,
    gdp_nominal: np.ndarray,
) -> FinancialProjection:
    """Run the year-by-year fund recursion and compute every performance indicator.

    Parameters
    ----------
    years
        Calendar years, shape ``(n_t,)``.
    insurable_earnings
        Total insurable earnings :math:`G_t`, shape ``(n_t,)``, THB.
    pension_expenditure, lump_sum_expenditure, admin_expenditure
        Expenditure components, shape ``(n_t,)``, THB.
    contribution_rate
        Legal contribution rate :math:`p_t`, shape ``(n_t,)``.
    collection_rate
        Compliance factor applied to contribution income, shape ``(n_t,)``.
    nominal_interest
        Nominal yield on the reserve :math:`i_t`, shape ``(n_t,)``.
    initial_reserve
        Reserve at the end of the base year, THB.
    n_pensioners, n_contributors
        Head counts for the dependency-ratio decomposition, shape ``(n_t,)``.
    gdp_nominal
        Nominal GDP, shape ``(n_t,)``, THB.

    Returns
    -------
    FinancialProjection
    """
    n_t = len(years)
    G = np.asarray(insurable_earnings, dtype=float)
    S = (
        np.asarray(pension_expenditure, dtype=float)
        + np.asarray(lump_sum_expenditure, dtype=float)
        + np.asarray(admin_expenditure, dtype=float)
    )
    p = np.asarray(contribution_rate, dtype=float)
    coll = np.asarray(collection_rate, dtype=float)
    i = np.asarray(nominal_interest, dtype=float)

    P = p * G * coll
    R = np.zeros(n_t, dtype=float)
    F = np.zeros(n_t, dtype=float)
    F[0] = float(initial_reserve)

    half_year = np.sqrt(np.maximum(1.0 + i, 0.0)) - 1.0

    for t in range(1, n_t):
        R[t] = half_year[t] * (P[t] - S[t]) + i[t] * F[t - 1]
        F[t] = F[t - 1] + P[t] + R[t] - S[t]

    I_total = P + R

    safe = lambda num, den: np.divide(  # noqa: E731
        num, den, out=np.zeros_like(np.asarray(num, dtype=float)), where=np.abs(den) > 1e-9
    )

    payg = safe(S, G)
    reserve_ratio = np.zeros(n_t, dtype=float)
    reserve_ratio[1:] = safe(F[:-1], S[1:])
    rer = safe(F, S)
    balance_ratio = safe(S - P, R)

    dependency = safe(np.asarray(n_pensioners, float), np.asarray(n_contributors, float))
    avg_pension = safe(np.asarray(pension_expenditure, float), np.asarray(n_pensioners, float))
    avg_earnings = safe(G, np.asarray(n_contributors, float))
    replacement = safe(avg_pension, avg_earnings)

    depletion_year: Optional[int] = None
    negative = np.where(F < 0)[0]
    if negative.size:
        depletion_year = int(years[negative[0]])

    return FinancialProjection(
        years=np.asarray(years),
        insurable_earnings=G,
        contribution_income=P,
        investment_income=R,
        total_income=I_total,
        pension_expenditure=np.asarray(pension_expenditure, float),
        lump_sum_expenditure=np.asarray(lump_sum_expenditure, float),
        admin_expenditure=np.asarray(admin_expenditure, float),
        total_expenditure=S,
        reserve=F,
        contribution_rate=p,
        payg_rate=payg,
        reserve_ratio=reserve_ratio,
        reserve_to_expenditure=rer,
        balance_ratio=balance_ratio,
        dependency_ratio=dependency,
        replacement_ratio=replacement,
        expenditure_pct_gdp=safe(S, np.asarray(gdp_nominal, float)),
        reserve_pct_gdp=safe(F, np.asarray(gdp_nominal, float)),
        depletion_year=depletion_year,
    )


# ---------------------------------------------------------------------------
# Discounting machinery shared by the contribution-rate formulas
# ---------------------------------------------------------------------------


def _discount_factors(nominal_interest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the ILO-PENS discount series :math:`V_t` and :math:`W_t`.

    .. math::
        v_t = (1 + i_t)^{-1},\\quad
        V_t = \\prod_{k=1}^{t} v_k,\\quad
        W_t = V_{t-1}\\, v_t^{1/2}

    Parameters
    ----------
    nominal_interest
        Nominal yield by year, shape ``(n_t,)``.

    Returns
    -------
    (V, W)
        Both shape ``(n_t,)``, with ``V[0] = 1``.
    """
    i = np.asarray(nominal_interest, dtype=float)
    v = 1.0 / (1.0 + i)
    V = np.ones_like(v)
    V[1:] = np.cumprod(v[1:])
    W = np.empty_like(v)
    W[0] = 1.0
    W[1:] = V[:-1] * np.sqrt(v[1:])
    return V, W


def general_average_premium(
    insurable_earnings: np.ndarray,
    total_expenditure: np.ndarray,
    nominal_interest: np.ndarray,
    initial_reserve: float,
    n_start: int = 1,
    n_end: Optional[int] = None,
) -> float:
    """Level contribution rate balancing the scheme over ``[n_start, n_end]``.

    Implements Technical Guide II.1.2.2(2).  With ``n_start = 1`` and
    ``n_end = n_t - 1`` this is the *General Average Premium* -- the flat rate
    that finances the whole projection period.  The 2013 ILO Thailand valuation
    obtained 21.1% for the SSO old-age branch over 2014-2113.

    Returns
    -------
    float
        The level contribution rate, as a proportion of insurable earnings.
    """
    G = np.asarray(insurable_earnings, dtype=float)
    S = np.asarray(total_expenditure, dtype=float)
    n_t = len(G)
    m = (n_t - 1) if n_end is None else int(n_end)
    n = int(n_start)

    V, W = _discount_factors(nominal_interest)
    G_bar = np.cumsum(G * W)
    S_bar = np.cumsum(S * W)

    numerator = (S_bar[m] - S_bar[n - 1]) - float(initial_reserve) * V[n - 1]
    denominator = G_bar[m] - G_bar[n - 1]
    return float(numerator / denominator) if abs(denominator) > 1e-9 else float("nan")


def rate_for_target_reserve_ratio(
    insurable_earnings: np.ndarray,
    total_expenditure: np.ndarray,
    nominal_interest: np.ndarray,
    initial_reserve: float,
    target_ratio: float,
    n_start: int = 1,
    n_end: Optional[int] = None,
) -> float:
    """Contribution rate that leaves the reserve ratio at ``target_ratio``.

    Implements Technical Guide II.1.2.2(3):

    .. math::
        C(a = a_0; n, m) = \\frac{a_0 V_{m-1} S_m - V_{n-1} F_{n-1}
                                  + (\\bar S_{m-1} - \\bar S_{n-1})}
                                 {\\bar G_{m-1} - \\bar G_{n-1}}

    Substituting ``target_ratio = 0`` recovers the level premium over
    ``[n, m-1]``.
    """
    G = np.asarray(insurable_earnings, dtype=float)
    S = np.asarray(total_expenditure, dtype=float)
    n_t = len(G)
    m = (n_t - 1) if n_end is None else int(n_end)
    n = int(n_start)

    V, W = _discount_factors(nominal_interest)
    G_bar = np.cumsum(G * W)
    S_bar = np.cumsum(S * W)

    numerator = (
        float(target_ratio) * V[m - 1] * S[m]
        - V[n - 1] * float(initial_reserve)
        + (S_bar[m - 1] - S_bar[n - 1])
    )
    denominator = G_bar[m - 1] - G_bar[n - 1]
    return float(numerator / denominator) if abs(denominator) > 1e-9 else float("nan")


def rate_for_target_balance_ratio(
    insurable_earnings: np.ndarray,
    total_expenditure: np.ndarray,
    nominal_interest: np.ndarray,
    initial_reserve: float,
    target_ratio: float = 1.0,
    n_start: int = 1,
    n_end: Optional[int] = None,
) -> float:
    """Contribution rate that leaves the balance ratio at ``target_ratio``.

    Implements Technical Guide II.1.2.2(4).  ``target_ratio = 1`` gives
    Thullen's scaled premium -- the rate under which the change in the reserve
    is zero at the end of the period, i.e. the classic "period of equilibrium"
    financing system.

    The published formula is algebraically heavy; the equivalent and more
    numerically stable statement used here is that at the horizon

    .. math::
        S_m - P_m = b_0 R_m
        \\;\\Longleftrightarrow\\;
        S_m - p\\,G_m = b_0\\Bigl[(\\sqrt{1+i_m}-1)(p G_m - S_m) + i_m F_{m-1}(p)\\Bigr]

    which is linear in ``p`` because the reserve recursion is linear in ``p``.
    The coefficients are obtained by running the recursion at ``p = 0`` and
    ``p = 1`` and interpolating.
    """
    G = np.asarray(insurable_earnings, dtype=float)
    S = np.asarray(total_expenditure, dtype=float)
    i = np.asarray(nominal_interest, dtype=float)
    n_t = len(G)
    m = (n_t - 1) if n_end is None else int(n_end)
    half = np.sqrt(np.maximum(1.0 + i, 0.0)) - 1.0

    def reserve_at(p: float, upto: int) -> np.ndarray:
        F = np.zeros(n_t, dtype=float)
        F[0] = float(initial_reserve)
        for t in range(1, upto + 1):
            P_t = p * G[t]
            R_t = half[t] * (P_t - S[t]) + i[t] * F[t - 1]
            F[t] = F[t - 1] + P_t + R_t - S[t]
        return F

    def residual(p: float) -> float:
        F = reserve_at(p, m)
        P_m = p * G[m]
        R_m = half[m] * (P_m - S[m]) + i[m] * F[m - 1]
        return (S[m] - P_m) - float(target_ratio) * R_m

    r0, r1 = residual(0.0), residual(1.0)
    if abs(r1 - r0) < 1e-12:
        return float("nan")
    return float(-r0 / (r1 - r0))


def solve_contribution_rate_for_horizon(
    insurable_earnings: np.ndarray,
    total_expenditure: np.ndarray,
    nominal_interest: np.ndarray,
    initial_reserve: float,
    horizon_index: int,
    target_reserve: float = 0.0,
) -> float:
    """Flat contribution rate leaving exactly ``target_reserve`` at ``horizon_index``.

    A convenience wrapper around the linear structure of the reserve recursion,
    useful for reform costing ("what rate keeps the fund solvent to 2080?").
    """
    G = np.asarray(insurable_earnings, dtype=float)
    S = np.asarray(total_expenditure, dtype=float)
    i = np.asarray(nominal_interest, dtype=float)
    half = np.sqrt(np.maximum(1.0 + i, 0.0)) - 1.0

    def reserve_at(p: float) -> float:
        F = float(initial_reserve)
        for t in range(1, horizon_index + 1):
            P_t = p * G[t]
            R_t = half[t] * (P_t - S[t]) + i[t] * F
            F = F + P_t + R_t - S[t]
        return F

    f0, f1 = reserve_at(0.0), reserve_at(1.0)
    if abs(f1 - f0) < 1e-9:
        return float("nan")
    return float((target_reserve - f0) / (f1 - f0))


__all__ = [
    "FinancialProjection",
    "project_fund",
    "general_average_premium",
    "rate_for_target_reserve_ratio",
    "rate_for_target_balance_ratio",
    "solve_contribution_rate_for_horizon",
]
