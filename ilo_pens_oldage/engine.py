"""
Projection engine -- the ILO-PENS ``Projection()`` orchestration loop.
======================================================================

Ties Modules 1-5 together and produces a single :class:`ValuationResult`,
restricted to the Old-Age Benefit Branch.

Execution pipeline
------------------
1. :func:`~ilo_pens_oldage.demography.project_population` -- national
   population (cohort-component).
2. :func:`~ilo_pens_oldage.labour.project_labour` -- labour force, employment
   and every economic index.
3. :func:`~ilo_pens_oldage.earnings.build_earnings_surface` -- salary surface,
   statutory floor and ceiling, reference salary (``FINSAL``) and accumulated
   contributions (``BALANCE``).
4. Year loop ``t = 0 .. n_years - 1``:

   a. :func:`~ilo_pens_oldage.actives.step_active_population` -- move the
      ``[sex, income, age, credit]`` mass and emit the flows.
   b. :func:`~ilo_pens_oldage.benefits.award_benefits` -- apply the vesting
      condition and the pension formula to those flows.
   c. :func:`~ilo_pens_oldage.benefits.step_pensioners` -- roll the pensioner
      stock, index pensions in payment, add the new awards.
   d. Accumulate income and expenditure.

5. :func:`~ilo_pens_oldage.finance.project_fund` -- reserve recursion, PAYG
   rate, reserve ratio, depletion year.
6. Summary financing indicators: General Average Premium, the rate holding a
   target reserve ratio, the rate holding a target balance ratio.

The year loop is the only sequential part; everything inside a year is
vectorised over ``(sex, income, age, credit)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .actives import ActiveState, coverage_target, step_active_population
from .benefits import PensionerState, award_benefits, step_pensioners
from .config import (
    AGE_MAX_INS,
    AGE_MIN_INS,
    Config,
    N_AGE_INS,
    N_AGE_POP,
    N_SEX,
    indexation_series,
)
from .demography import DemographyResult, project_population
from .earnings import EarningsSurface, build_earnings_surface
from .finance import (
    FinancialProjection,
    general_average_premium,
    project_fund,
    rate_for_target_balance_ratio,
    rate_for_target_reserve_ratio,
)
from .labour import LabourResult, project_labour


@dataclass
class ValuationResult:
    """Everything produced by one run of the old-age model."""

    config: Config
    scenario_name: str

    demography: DemographyResult
    labour: LabourResult
    earnings: EarningsSurface
    finance: FinancialProjection

    # -- Membership time series, shape (n_t,) --------------------------------
    active_total: np.ndarray
    inactive_total: np.ndarray
    contributors_total: np.ndarray
    new_entrants_total: np.ndarray

    pensioners_total: np.ndarray
    new_pensioners: np.ndarray
    retirement_grants_n: np.ndarray
    death_grants_n: np.ndarray

    mean_credit_active: np.ndarray
    vested_share_active: np.ndarray
    avg_insurable_earnings: np.ndarray
    avg_pension: np.ndarray

    # -- Expenditure components, shape (n_t,), THB ----------------------------
    pension_expenditure: np.ndarray
    retirement_grant_expenditure: np.ndarray
    death_grant_expenditure: np.ndarray
    guarantee_lump_expenditure: np.ndarray
    admin_expenditure: np.ndarray

    # -- Detailed cross-sections ----------------------------------------------
    active_by_sex_age: np.ndarray            # (2, 55, n_t)
    pensioners_by_sex_age: np.ndarray        # (2, 101, n_t)
    credit_distribution: np.ndarray          # (n_t, 51)

    final_active_state: ActiveState
    final_pensioner_state: PensionerState

    diagnostics: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def years(self) -> np.ndarray:
        return self.config.run.years

    def summary(self) -> Dict[str, object]:
        """Headline indicators for reporting."""
        f = self.finance
        return {
            "scenario": self.scenario_name,
            "base_year": int(self.config.run.base_year),
            "final_year": int(self.years[-1]),
            "contribution_rate": float(self.config.scheme.contribution_rate_total),
            "payg_rate_final": float(f.payg_rate[-1]),
            "payg_rate_at_25y": float(f.payg_rate[min(25, len(f.payg_rate) - 1)]),
            "general_average_premium": float(f.general_average_premium),
            "rate_for_target_reserve_ratio": float(f.rate_for_target_reserve_ratio),
            "rate_for_target_balance_ratio": float(f.rate_for_target_balance_ratio),
            "depletion_year": f.depletion_year,
            "reserve_ratio_final": float(f.reserve_ratio[-1]),
            "dependency_ratio_final": float(f.dependency_ratio[-1]),
            "replacement_ratio_final": float(f.replacement_ratio[-1]),
            "active_final": float(self.active_total[-1]),
            "pensioners_final": float(self.pensioners_total[-1]),
            "vested_share_final": float(self.vested_share_active[-1]),
        }


def run_valuation(cfg: Config, scenario_name: Optional[str] = None) -> ValuationResult:
    """Execute the full six-module old-age pipeline.

    Parameters
    ----------
    cfg
        Fully populated configuration.  Use
        :func:`~ilo_pens_oldage.dummy_data.build_dummy_config` first if any
        array input is still ``None``.
    scenario_name
        Label carried through to the outputs.  Defaults to ``cfg.scenario_name``.

    Returns
    -------
    ValuationResult
    """
    name = scenario_name or cfg.scenario_name
    run, scheme, mem, fin = cfg.run, cfg.scheme, cfg.membership, cfg.finance
    n_t = run.n_t

    # -- Modules 1 and 2 ----------------------------------------------------
    demo = project_population(cfg)
    lab = project_labour(cfg, demo)

    # -- Earnings surface ---------------------------------------------------
    earn = build_earnings_surface(cfg, lab.nominal_wage_growth,
                                  lab.nominal_interest_rate)

    pension_adj = indexation_series(
        scheme.pension_indexation, lab.inflation, lab.nominal_wage_growth,
        scheme.pension_indexation_fixed_rate,
    )

    # -- Modules 3 and 4 state -----------------------------------------------
    active_state = ActiveState.from_config(cfg)
    pens_state = PensionerState.from_config(cfg)

    density = np.asarray(mem.density, float)

    def _series(value) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        return np.full(n_t, float(arr)) if arr.ndim == 0 else arr

    admin_rate = _series(scheme.admin_expense_rate)
    collection = _series(scheme.collection_rate)
    contribution_rate = _series(scheme.contribution_rate_total)

    # -- Accumulators --------------------------------------------------------
    z = lambda: np.zeros(n_t, dtype=float)  # noqa: E731
    active_total, inactive_total, contributors_total = z(), z(), z()
    new_entrants_total = z()
    pensioners_ts, new_pensioners_ts = z(), z()
    ret_grant_n_ts, death_grant_n_ts = z(), z()
    mean_credit_ts, vested_ts = z(), z()
    G = z()
    pension_exp = z()
    ret_grant_exp, death_grant_exp, guarantee_exp = z(), z(), z()

    active_by_sex_age = np.zeros((N_SEX, N_AGE_INS, n_t), dtype=float)
    pens_by_sex_age = np.zeros((N_SEX, N_AGE_POP, n_t), dtype=float)
    credit_dist = np.zeros((n_t, active_state.active.shape[3]), dtype=float)
    credits = np.arange(active_state.active.shape[3], dtype=float)

    def _record(t: int) -> None:
        """Snapshot the stocks and the year's contribution base."""
        active_total[t] = active_state.total_active
        inactive_total[t] = active_state.total_inactive
        active_by_sex_age[:, :, t] = active_state.active_by_age
        credit_dist[t] = active_state.active.sum(axis=(0, 1, 2))

        # Contributors: Cont(x,t) = Act(x,t) . Dens(x,t)
        act_sjx = active_state.active.sum(axis=3)          # (2, 3, 55)
        cont_sjx = act_sjx * density[:, None, :, t]
        contributors_total[t] = float(cont_sjx.sum())
        G[t] = float((cont_sjx * earn.insurable[:, :, :, t]).sum())

        pensioners_ts[t] = pens_state.total_pensioners
        pens_by_sex_age[:, :, t] = pens_state.n
        pension_exp[t] = pens_state.total_pension_expenditure

        n_act = active_state.active.sum()
        if n_act > 0:
            mean_credit_ts[t] = float(
                (active_state.active * credits[None, None, None, :]).sum() / n_act
            )
            vested = active_state.active[:, :, :, credits >= scheme.vesting_years].sum()
            vested_ts[t] = float(vested / n_act)

    _record(0)

    # -- Year loop -----------------------------------------------------------
    for t in range(n_t - 1):
        target = coverage_target(cfg, lab.employed, t + 1)
        active_state, flows = step_active_population(cfg, active_state, t, target)

        awards = award_benefits(cfg, flows, earn, t + 1)

        pens_state, guarantee_lump, _ = step_pensioners(
            cfg, pens_state, awards, t, pension_uprating=float(pension_adj[t + 1])
        )

        _record(t + 1)

        new_entrants_total[t + 1] = float(flows.new_entrants.sum())
        new_pensioners_ts[t + 1] = float(awards.new_pension_n.sum())
        ret_grant_n_ts[t + 1] = awards.retirement_grant_n
        death_grant_n_ts[t + 1] = awards.death_grant_n

        ret_grant_exp[t + 1] = awards.retirement_grant_total
        death_grant_exp[t + 1] = awards.death_grant_total
        guarantee_exp[t + 1] = guarantee_lump

    # -- Module 5 -------------------------------------------------------------
    admin_exp = admin_rate * G
    lump_exp = ret_grant_exp + death_grant_exp + guarantee_exp

    fin_proj = project_fund(
        years=run.years,
        insurable_earnings=G,
        pension_expenditure=pension_exp,
        lump_sum_expenditure=lump_exp,
        admin_expenditure=admin_exp,
        contribution_rate=contribution_rate,
        collection_rate=collection,
        nominal_interest=lab.nominal_interest_rate,
        initial_reserve=fin.initial_reserve,
        n_pensioners=pensioners_ts,
        n_contributors=contributors_total,
        gdp_nominal=lab.gdp_nominal,
    )

    horizon = fin.gap_horizon_years if fin.gap_horizon_years is not None else n_t - 1
    horizon = int(min(horizon, n_t - 1))
    S_total = fin_proj.total_expenditure

    fin_proj.general_average_premium = general_average_premium(
        G, S_total, lab.nominal_interest_rate, fin.initial_reserve, 1, horizon
    )
    fin_proj.rate_for_target_reserve_ratio = rate_for_target_reserve_ratio(
        G, S_total, lab.nominal_interest_rate, fin.initial_reserve,
        fin.target_reserve_ratio, 1, horizon,
    )
    fin_proj.rate_for_target_balance_ratio = rate_for_target_balance_ratio(
        G, S_total, lab.nominal_interest_rate, fin.initial_reserve,
        fin.target_balance_ratio, 1, horizon,
    )

    safe = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)  # noqa: E731

    return ValuationResult(
        config=cfg,
        scenario_name=name,
        demography=demo,
        labour=lab,
        earnings=earn,
        finance=fin_proj,
        active_total=active_total,
        inactive_total=inactive_total,
        contributors_total=contributors_total,
        new_entrants_total=new_entrants_total,
        pensioners_total=pensioners_ts,
        new_pensioners=new_pensioners_ts,
        retirement_grants_n=ret_grant_n_ts,
        death_grants_n=death_grant_n_ts,
        mean_credit_active=mean_credit_ts,
        vested_share_active=vested_ts,
        avg_insurable_earnings=safe(G, contributors_total),
        avg_pension=safe(pension_exp, pensioners_ts),
        pension_expenditure=pension_exp,
        retirement_grant_expenditure=ret_grant_exp,
        death_grant_expenditure=death_grant_exp,
        guarantee_lump_expenditure=guarantee_exp,
        admin_expenditure=admin_exp,
        active_by_sex_age=active_by_sex_age,
        pensioners_by_sex_age=pens_by_sex_age,
        credit_distribution=credit_dist,
        final_active_state=active_state,
        final_pensioner_state=pens_state,
        diagnostics={"insurable_earnings": G, "lump_sum_expenditure": lump_exp},
    )


__all__ = ["ValuationResult", "run_valuation"]
