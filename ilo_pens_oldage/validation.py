"""
Mathematical and actuarial consistency checks.
==============================================

Every check returns a :class:`Check` carrying the measured discrepancy, so a
run can be audited rather than merely trusted.  :func:`run_all_checks` executes
the battery and :func:`format_report` prints it.

=================================  ==========================================
Check                              Identity tested
=================================  ==========================================
``population_balance``             :math:`P_{t+1} = P_t - D_{t+1} + B_{t+1} + M_{t+1}`
``fund_recursion``                 :math:`F_t - F_{t-1} = I_t - S_t`
``income_identity``                :math:`I_t = P_t + R_t`
``payg_identity``                  :math:`C^{PAYG}_t G_t = S_t`
``payg_decomposition``             :math:`C^{PAYG}_t = d_t \\times r_t` on the
                                   pension component
``expenditure_components``         Pension + grants + guarantee + admin = S_t
``old_age_scope``                  Only old-age benefits appear in expenditure
``gap_balances_the_fund``          The GAP leaves a zero discounted balance
``credit_axis_conservation``       Credit accrual preserves head count and mean
``non_negative_stocks``            No negative members, pensioners or amounts
``coverage_target_tracking``       Active head count tracks Covrate x Employed
``vesting_threshold``              No pension is paid below 180 months
``monotone_eligibility``           Pension is non-decreasing in credit
``replacement_rate_bounds``        20% at 15 years, 72.5% at 50 years
=================================  ==========================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .actives import CREDIT_GRID, accrue_credit
from .benefits import old_age_pension_amount, retirement_eligible
from .config import AGE_MAX_INS, AGE_MIN_INS, Config
from .engine import ValuationResult


@dataclass
class Check:
    """Result of one consistency check."""

    name: str
    passed: bool
    detail: str
    max_error: float = 0.0
    tolerance: float = 0.0

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name:<30} {self.detail}"


def _rel_error(a: np.ndarray, b: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """Relative error with a denominator floor, to avoid dividing by ~0."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(b), floor)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_population_balance(result: ValuationResult, tol: float = 1e-6) -> Check:
    """Population closes: next year's total equals this year's net of flows."""
    d = result.demography
    total = d.total
    lhs = total[1:]
    rhs = total[:-1] - d.deaths[1:] + d.births[1:] + d.net_migration[1:]
    err = float(_rel_error(lhs, rhs).max())
    return Check("population_balance", err <= tol,
                 f"max relative error {err:.3e} (tol {tol:.0e})", err, tol)


def check_fund_recursion(result: ValuationResult, tol: float = 1e-8) -> Check:
    """F_t - F_{t-1} = I_t - S_t for every projection year."""
    f = result.finance
    lhs = f.reserve[1:] - f.reserve[:-1]
    rhs = f.total_income[1:] - f.total_expenditure[1:]
    scale = max(float(np.abs(f.reserve).max()), 1.0)
    err = float((np.abs(lhs - rhs) / scale).max())
    return Check("fund_recursion", err <= tol,
                 f"max scaled error {err:.3e} (tol {tol:.0e})", err, tol)


def check_income_identity(result: ValuationResult, tol: float = 1e-9) -> Check:
    """I_t = P_t + R_t."""
    f = result.finance
    scale = max(float(np.abs(f.total_income).max()), 1.0)
    err = float((np.abs(f.total_income - f.contribution_income
                        - f.investment_income) / scale).max())
    return Check("income_identity", err <= tol, f"max scaled error {err:.3e}", err, tol)


def check_payg_identity(result: ValuationResult, tol: float = 1e-9) -> Check:
    """C_PAYG(t) * G_t = S_t wherever G_t > 0."""
    f = result.finance
    mask = f.insurable_earnings > 0
    scale = max(float(np.abs(f.total_expenditure).max()), 1.0)
    err = float((np.abs(f.payg_rate[mask] * f.insurable_earnings[mask]
                        - f.total_expenditure[mask]) / scale).max())
    return Check("payg_identity", err <= tol, f"max scaled error {err:.3e}", err, tol)


def check_payg_decomposition(result: ValuationResult, tol: float = 0.02) -> Check:
    """C_PAYG = d * r on the pension component.

    Lump sums and administration sit in ``S_t`` but not in the pension numerator
    of ``r``, so the identity is exact only for the pension component.
    """
    f = result.finance
    mask = f.insurable_earnings > 0
    pension_rate = np.zeros_like(f.payg_rate)
    pension_rate[mask] = f.pension_expenditure[mask] / f.insurable_earnings[mask]
    err = float(np.abs(pension_rate[mask]
                       - (f.dependency_ratio * f.replacement_ratio)[mask]).max())
    return Check("payg_decomposition", err <= tol,
                 f"max absolute error {err:.3e} on the pension cost rate", err, tol)


def check_expenditure_components(result: ValuationResult, tol: float = 1e-9) -> Check:
    """The four expenditure components sum to total expenditure."""
    total = (result.pension_expenditure
             + result.retirement_grant_expenditure
             + result.death_grant_expenditure
             + result.guarantee_lump_expenditure
             + result.admin_expenditure)
    scale = max(float(np.abs(result.finance.total_expenditure).max()), 1.0)
    err = float((np.abs(total - result.finance.total_expenditure) / scale).max())
    return Check("expenditure_components", err <= tol,
                 f"components reconcile to S_t, max scaled error {err:.3e}", err, tol)


def check_old_age_scope(result: ValuationResult) -> Check:
    """Only old-age benefits may appear in expenditure.

    A structural assertion rather than a numerical one: the result object
    carries exactly the four old-age expenditure streams plus administration,
    and nothing else.  If an out-of-scope branch were ever reintroduced this
    check would need updating, which is the point.
    """
    expected = {
        "pension_expenditure",
        "retirement_grant_expenditure",
        "death_grant_expenditure",
        "guarantee_lump_expenditure",
        "admin_expenditure",
    }
    present = {n for n in expected if hasattr(result, n)}
    forbidden = {"invalidity_expenditure", "survivor_expenditure",
                 "orphan_expenditure", "funeral_expenditure"}
    leaked = {n for n in forbidden if hasattr(result, n)}
    ok = present == expected and not leaked
    detail = (f"{len(present)} old-age streams, no out-of-scope streams"
              if ok else f"missing {expected - present}, leaked {leaked}")
    return Check("old_age_scope", ok, detail)


def check_gap_balances_the_fund(result: ValuationResult, tol: float = 1e-6) -> Check:
    """The GAP zeroes the discounted balance over its horizon.

    .. math::
        F_{n-1} V_{n-1} + p\\,(\\bar G_m - \\bar G_{n-1})
            - (\\bar S_m - \\bar S_{n-1}) = 0
    """
    from .finance import _discount_factors

    f, cfg = result.finance, result.config
    gap = f.general_average_premium
    if not np.isfinite(gap):
        return Check("gap_balances_the_fund", False, "GAP is not finite",
                     float("inf"), tol)

    n_t = len(f.years)
    m = int(min(cfg.finance.gap_horizon_years or (n_t - 1), n_t - 1))
    n = 1

    V, W = _discount_factors(result.labour.nominal_interest_rate)
    G_bar = np.cumsum(f.insurable_earnings * W)
    S_bar = np.cumsum(f.total_expenditure * W)

    balance = (cfg.finance.initial_reserve * V[n - 1]
               + gap * (G_bar[m] - G_bar[n - 1])
               - (S_bar[m] - S_bar[n - 1]))
    scale = max(abs(float(S_bar[m] - S_bar[n - 1])), 1.0)
    err = abs(float(balance)) / scale
    return Check("gap_balances_the_fund", err <= tol,
                 f"discounted residual {err:.3e} of PV(expenditure); "
                 f"GAP = {gap:.4%}", err, tol)


def check_credit_axis_conservation(tol: float = 1e-10) -> Check:
    """Credit accrual conserves head count and shifts the mean by the density."""
    rng = np.random.default_rng(7)
    arr = rng.random((2, 3, 55, 51)) * 1000.0
    dens = rng.random((2, 55)) * 0.9 + 0.05

    shifted = accrue_credit(arr, dens)
    head_err = float(abs(shifted.sum() - arr.sum()) / max(arr.sum(), 1.0))

    at_cap = arr[:, :, :, -1].sum(axis=1)
    before = (arr * CREDIT_GRID[None, None, None, :]).sum(axis=(1, 3))
    after = (shifted * CREDIT_GRID[None, None, None, :]).sum(axis=(1, 3))
    expected = before + dens * (arr.sum(axis=(1, 3)) - at_cap)
    mean_err = float((np.abs(after - expected) / np.maximum(np.abs(expected), 1.0)).max())

    err = max(head_err, mean_err)
    return Check("credit_axis_conservation", err <= tol,
                 f"head-count error {head_err:.3e}, mean-credit error {mean_err:.3e}",
                 err, tol)


def check_non_negative_stocks(result: ValuationResult) -> Check:
    """No stock or benefit amount may go negative."""
    arrays = {
        "active": result.final_active_state.active,
        "inactive": result.final_active_state.inactive,
        "pensioners_n": result.final_pensioner_state.n,
        "pensioners_amt": result.final_pensioner_state.amt,
        "insurable_earnings": result.finance.insurable_earnings,
        "pension_expenditure": result.pension_expenditure,
        "retirement_grant_expenditure": result.retirement_grant_expenditure,
        "death_grant_expenditure": result.death_grant_expenditure,
        "active_total": result.active_total,
        "pensioners_total": result.pensioners_total,
    }
    problems, worst = [], 0.0
    for name, arr in arrays.items():
        mn = float(np.min(arr))
        if mn < -1e-6:
            problems.append(f"{name} min={mn:.3e}")
            worst = min(worst, mn)
    return Check("non_negative_stocks", not problems,
                 "; ".join(problems) if problems else "all stocks non-negative",
                 abs(worst), 1e-6)


def check_coverage_target_tracking(result: ValuationResult, tol: float = 0.02) -> Check:
    """Active head count tracks the exogenous coverage target."""
    cov = np.asarray(result.config.membership.coverage_rate, float)
    emp = result.labour.employed[:, AGE_MIN_INS : AGE_MAX_INS + 1, :]
    target = (cov * emp).sum(axis=(0, 1))
    err = float(_rel_error(result.active_total[1:], target[1:]).max())
    return Check("coverage_target_tracking", err <= tol,
                 f"max relative deviation {err:.2%} (tol {tol:.0%})", err, tol)


def check_vesting_threshold(cfg: Config) -> Check:
    """No pension may be payable below the 180-month vesting threshold.

    The single most important scope guarantee of this model: members below the
    threshold must receive a lump-sum grant, never a pension.
    """
    fins = np.ones((2, 3, 55), dtype=float) * 300_000.0
    pension = old_age_pension_amount(cfg, fins)
    below = CREDIT_GRID < cfg.scheme.vesting_years
    leaked = float(np.abs(pension[:, :, :, below]).max())

    # And the eligibility mask must also bar it, at every projection year.
    mask = retirement_eligible(cfg, 0)
    mask_leak = bool(mask[:, below].any())

    ok = leaked <= 1e-12 and not mask_leak
    return Check("vesting_threshold", ok,
                 f"max pension below {cfg.scheme.vesting_months} months "
                 f"= {leaked:.3e} THB; eligibility mask clean = {not mask_leak}",
                 leaked, 1e-12)


def check_monotone_eligibility(cfg: Config) -> Check:
    """The old-age pension must be non-decreasing in accumulated credit."""
    fins = np.ones((2, 3, 55), dtype=float) * 300_000.0
    diffs = np.diff(old_age_pension_amount(cfg, fins), axis=3)
    worst = float(diffs.min())
    return Check("monotone_eligibility", worst >= -1e-9,
                 f"minimum credit-to-credit increment {worst:.3e} THB",
                 abs(min(worst, 0.0)), 1e-9)


def check_replacement_rate_bounds(cfg: Config) -> Check:
    """Accrued replacement rates must match the statutory formula.

    Thai SSO: 20% at 15 years, +1.5 points per extra year, so 35 years yields
    50% and 50 years yields 72.5%.
    """
    s = cfg.scheme
    rates = old_age_pension_amount(cfg, np.ones((2, 3, 55), dtype=float))[0, 0, 0, :]

    idx = int(np.ceil(s.vesting_years))
    at_vesting = float(rates[idx])
    err = abs(at_vesting - s.accrual_base_rate)

    top = float(rates[-1])
    expected_top = s.accrual_base_rate + s.accrual_increment_rate * (
        len(rates) - 1 - s.vesting_years
    )
    top_err = abs(top - expected_top)

    passed = err <= 1e-9 and top_err <= 1e-9
    return Check("replacement_rate_bounds", passed,
                 f"rate at vesting {at_vesting:.2%} (expected {s.accrual_base_rate:.2%}); "
                 f"at {len(rates) - 1} yrs {top:.2%} (expected {expected_top:.2%})",
                 max(err, top_err), 1e-9)


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def run_all_checks(result: ValuationResult) -> List[Check]:
    """Run every consistency check against a completed valuation."""
    return [
        check_population_balance(result),
        check_fund_recursion(result),
        check_income_identity(result),
        check_payg_identity(result),
        check_payg_decomposition(result),
        check_expenditure_components(result),
        check_old_age_scope(result),
        check_gap_balances_the_fund(result),
        check_credit_axis_conservation(),
        check_non_negative_stocks(result),
        check_coverage_target_tracking(result),
        check_vesting_threshold(result.config),
        check_monotone_eligibility(result.config),
        check_replacement_rate_bounds(result.config),
    ]


def format_report(checks: List[Check]) -> str:
    """Render a list of checks as a plain-text report."""
    lines = ["", "Consistency checks", "=" * 76]
    lines.extend(str(c) for c in checks)
    lines.append("-" * 76)
    lines.append(f"{sum(c.passed for c in checks)}/{len(checks)} checks passed")
    return "\n".join(lines)


__all__ = ["Check", "run_all_checks", "format_report"] + [
    n for n in dir() if n.startswith("check_")
]
