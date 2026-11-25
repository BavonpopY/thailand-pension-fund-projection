"""
Module 2 -- Labour force and macroeconomic frame.
=================================================

Turns the national population projection into the employed population that the
scheme's coverage rates are applied to, and builds every economic index the
benefit and financial modules need.

Labour market
-------------
.. math::
    LF_{s,a,t}   &= P_{s,a,t} \\times \\mathrm{LFPR}_{s,a,t} \\\\
    E_{s,a,t}    &= LF_{s,a,t} \\times (1 - u_{s,a,t})

Economic indices
----------------
Nominal wage growth compounds price inflation with real wage growth:

.. math::
    (1 + w_t) = (1 + \\pi_t)(1 + \\rho_t)

and the nominal investment return likewise compounds inflation with the real
return.  All of these are returned both as annual *rates* and as cumulative
*indices* with the base year equal to 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config, rates_to_index
from .demography import DemographyResult


@dataclass
class LabourResult:
    """Labour-market and macroeconomic projection outputs.

    All arrays indexed ``[sex, age 0..100, year]`` or ``[year]``.
    """

    labour_force: np.ndarray          # (2, 101, n_t) persons
    employed: np.ndarray              # (2, 101, n_t) persons
    unemployed: np.ndarray            # (2, 101, n_t) persons

    inflation: np.ndarray             # (n_t,) rate
    real_wage_growth: np.ndarray      # (n_t,) rate
    nominal_wage_growth: np.ndarray   # (n_t,) rate
    real_gdp_growth: np.ndarray       # (n_t,) rate
    nominal_interest_rate: np.ndarray  # (n_t,) rate

    price_index: np.ndarray           # (n_t,) base year = 1
    wage_index: np.ndarray            # (n_t,) base year = 1
    gdp_nominal: np.ndarray           # (n_t,) THB

    @property
    def total_labour_force(self) -> np.ndarray:
        return self.labour_force.sum(axis=(0, 1))

    @property
    def total_employed(self) -> np.ndarray:
        return self.employed.sum(axis=(0, 1))

    @property
    def aggregate_lfpr(self) -> np.ndarray:
        """Labour force / population 15+, shape ``(n_t,)``."""
        return self._aggregate_lfpr

    @property
    def aggregate_unemployment(self) -> np.ndarray:
        lf = self.total_labour_force
        return np.divide(
            self.unemployed.sum(axis=(0, 1)), lf, out=np.zeros_like(lf), where=lf > 0
        )


def project_labour(cfg: Config, demo: DemographyResult) -> LabourResult:
    """Project the labour force, employment and the macroeconomic indices.

    Parameters
    ----------
    cfg
        Fully populated configuration.
    demo
        Output of :func:`ilo_pens_th.demography.project_population`.

    Returns
    -------
    LabourResult
    """
    l = cfg.labour

    lfpr = np.asarray(l.lfpr, dtype=float)
    urate = np.asarray(l.unemployment_rate, dtype=float)

    labour_force = demo.population * lfpr
    unemployed = labour_force * urate
    employed = labour_force - unemployed

    inflation = np.asarray(l.inflation, dtype=float)
    real_wage = np.asarray(l.real_wage_growth, dtype=float)
    real_gdp = np.asarray(l.real_gdp_growth, dtype=float)
    real_return = np.asarray(l.real_investment_return, dtype=float)

    nominal_wage = (1.0 + inflation) * (1.0 + real_wage) - 1.0

    if cfg.finance.nominal_interest_rate is not None:
        nominal_interest = np.asarray(cfg.finance.nominal_interest_rate, dtype=float)
    else:
        nominal_interest = (1.0 + inflation) * (1.0 + real_return) - 1.0

    price_index = rates_to_index(inflation)
    wage_index = rates_to_index(nominal_wage)

    nominal_gdp_growth = (1.0 + inflation) * (1.0 + real_gdp) - 1.0
    gdp_nominal = rates_to_index(nominal_gdp_growth, base_value=l.base_gdp)

    result = LabourResult(
        labour_force=labour_force,
        employed=employed,
        unemployed=unemployed,
        inflation=inflation,
        real_wage_growth=real_wage,
        nominal_wage_growth=nominal_wage,
        real_gdp_growth=real_gdp,
        nominal_interest_rate=nominal_interest,
        price_index=price_index,
        wage_index=wage_index,
        gdp_nominal=gdp_nominal,
    )

    pop_15p = demo.population[:, 15:, :].sum(axis=(0, 1))
    lf_total = labour_force.sum(axis=(0, 1))
    result._aggregate_lfpr = np.divide(  # type: ignore[attr-defined]
        lf_total, pop_15p, out=np.zeros_like(lf_total), where=pop_15p > 0
    )
    return result


__all__ = ["LabourResult", "project_labour"]
