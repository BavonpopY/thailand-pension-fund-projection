"""
Module 6 -- Sensitivity and scenario engine (old-age reform options).
=====================================================================

A declarative wrapper around :class:`~ilo_pens_oldage.config.Config` that
applies parameter overrides, re-runs the pipeline, and tabulates the headline
indicators side by side.

Two kinds of override are supported:

``fields``
    Plain dataclass-field replacements, e.g.
    ``{"scheme": {"normal_retirement_age": 60}}``.

``transforms``
    Callables applied to an existing array input, for perturbations that are
    naturally multiplicative or additive, e.g.
    ``{"membership.insured_mortality_qx": lambda a: a * 0.8}``.

The standard battery mirrors the sensitivity tests of ILO/TF/THAILAND/R.40
section 4.2 and the old-age reform options of section 5.

Example
-------
>>> from ilo_pens_oldage import Config, build_dummy_config
>>> from ilo_pens_oldage.scenarios import ScenarioSet, default_scenarios, compare
>>> cfg = build_dummy_config(Config())
>>> results = ScenarioSet(default_scenarios(cfg.run.n_t)).run(cfg)
>>> compare(results)["general_average_premium"]  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from .config import Config
from .engine import ValuationResult, run_valuation


@dataclass
class Scenario:
    """A named set of parameter overrides applied to a baseline configuration.

    Attributes
    ----------
    name
        Label used in the comparison table and the Excel workbook.
    description
        One line on what the scenario tests and why it matters.
    fields
        Nested mapping ``{block: {field: value}}`` of dataclass replacements.
    transforms
        Mapping ``{"block.field": callable}`` applied to the existing array
        value.  Applied after ``fields``.
    """

    name: str
    description: str = ""
    fields: Dict[str, Dict[str, object]] = field(default_factory=dict)
    transforms: Dict[str, Callable[[np.ndarray], np.ndarray]] = field(default_factory=dict)

    def apply(self, base: Config) -> Config:
        """Return a new configuration with this scenario's overrides applied."""
        cfg = base.copy()
        for block_name, updates in self.fields.items():
            block = getattr(cfg, block_name, None)
            if block is None:
                raise KeyError(f"Unknown configuration block {block_name!r}")
            for key, value in updates.items():
                if not hasattr(block, key):
                    raise KeyError(f"Unknown field {key!r} in block {block_name!r}")
                setattr(block, key, value)

        for path, fn in self.transforms.items():
            block_name, _, field_name = path.partition(".")
            block = getattr(cfg, block_name, None)
            if block is None or not hasattr(block, field_name):
                raise KeyError(f"Unknown transform target {path!r}")
            current = getattr(block, field_name)
            if current is None:
                raise ValueError(
                    f"Cannot transform {path!r}: the field is None. Populate the "
                    "configuration with dummy or real data first."
                )
            setattr(block, field_name, fn(np.asarray(current, dtype=float)))

        cfg.scenario_name = self.name
        return cfg


@dataclass
class ScenarioSet:
    """A collection of scenarios to run against a common baseline."""

    scenarios: List[Scenario]
    include_baseline: bool = True
    baseline_name: str = "Baseline"

    def run(self, base: Config) -> Dict[str, ValuationResult]:
        """Run the baseline and every scenario.

        Parameters
        ----------
        base
            Fully populated baseline configuration.

        Returns
        -------
        dict
            Scenario name -> :class:`ValuationResult`, baseline first.
        """
        results: Dict[str, ValuationResult] = {}
        if self.include_baseline:
            baseline = base.copy()
            baseline.scenario_name = self.baseline_name
            results[self.baseline_name] = run_valuation(baseline, self.baseline_name)
        for scenario in self.scenarios:
            results[scenario.name] = run_valuation(scenario.apply(base), scenario.name)
        return results


def compare(results: Dict[str, ValuationResult]):
    """Tabulate the headline indicators of several scenarios.

    Parameters
    ----------
    results
        Output of :meth:`ScenarioSet.run`.

    Returns
    -------
    pandas.DataFrame
        One row per scenario, with the difference from the first row appended
        for the GAP and the depletion year.
    """
    import pandas as pd

    df = pd.DataFrame([r.summary() for r in results.values()]).set_index("scenario")

    if len(df) > 1:
        base_gap = df["general_average_premium"].iloc[0]
        df["gap_diff_pp"] = (df["general_average_premium"] - base_gap) * 100.0
        base_depletion = df["depletion_year"].iloc[0]
        if base_depletion is not None and not df["depletion_year"].isna().all():
            df["depletion_shift_years"] = (
                df["depletion_year"].astype("Float64") - float(base_depletion)
            )
    return df


def phased_nra(n_t: int, start: int, end: int, start_year: int,
               years_to_phase: int) -> np.ndarray:
    """Build a phased normal-retirement-age path.

    Parameters
    ----------
    n_t
        Number of projection points.
    start, end
        Initial and ultimate normal retirement age.
    start_year
        Projection-year index at which the increase begins.
    years_to_phase
        Number of years over which the increase completes.

    Returns
    -------
    np.ndarray
        Integer NRA by projection year, shape ``(n_t,)``.
    """
    path = np.full(n_t, float(start))
    stop = min(n_t, start_year + years_to_phase)
    if stop > start_year:
        path[start_year:stop] = np.linspace(start, end, stop - start_year)
    path[stop:] = end
    return np.round(path).astype(int)


def default_scenarios(n_t: int = 101) -> List[Scenario]:
    """The standard old-age sensitivity and reform battery.

    Mirrors ILO/TF/THAILAND/R.40 sections 4.2 and 5, restricted to tests that
    bear on the old-age branch.

    Parameters
    ----------
    n_t
        Number of projection points, needed to size the NRA paths.

    Returns
    -------
    list of Scenario
    """
    from .dummy_data import (
        make_leaving_rate_from_nra_path,
        make_retirement_rate_from_nra_path,
    )

    def _nra_reform(path: np.ndarray) -> Dict[str, Dict[str, object]]:
        """Overrides for a retirement-age reform.

        A retirement-age reform is not only a change of eligibility age: exit
        and claiming behaviour must move with it, or members simply leave
        earlier and sit deferred.  This sets all three together.
        """
        return {
            "scheme": {"nra_path": path},
            "membership": {
                "leaving_rate": make_leaving_rate_from_nra_path(path),
                "retirement_rate": make_retirement_rate_from_nra_path(path),
            },
        }

    return [
        # ---- Section 4.2 sensitivity tests --------------------------------
        Scenario(
            "Low investment return (-1pp)",
            "Real return on assets 1 percentage point below baseline; tests the "
            "fund's dependence on investment income.",
            transforms={"labour.real_investment_return": lambda a: a - 0.01,
                        "finance.nominal_interest_rate": lambda a: a - 0.01},
        ),
        Scenario(
            "High investment return (+1pp)",
            "Real return 1 percentage point above baseline.",
            transforms={"labour.real_investment_return": lambda a: a + 0.01,
                        "finance.nominal_interest_rate": lambda a: a + 0.01},
        ),
        Scenario(
            "Rapid ageing (TFR -0.2)",
            "Total fertility rate 0.2 below baseline: fewer future contributors "
            "supporting the same pensioner cohorts.",
            transforms={"demography.tfr_path": lambda a: np.maximum(a - 0.2, 0.5)},
        ),
        Scenario(
            "Higher fertility (TFR +0.2)",
            "Total fertility rate 0.2 above baseline.",
            transforms={"demography.tfr_path": lambda a: a + 0.2},
        ),
        Scenario(
            "Mortality improvement (-20% qx)",
            "Insured mortality 20% lighter: longer payout durations at unchanged "
            "contribution income. The classic longevity stress.",
            transforms={"membership.insured_mortality_qx": lambda a: a * 0.80,
                        "demography.mortality_qx": lambda a: a * 0.80},
        ),
        Scenario(
            "Slower coverage growth",
            "Coverage rate 10% below baseline throughout: a smaller contributor "
            "base against the same accrued liabilities.",
            transforms={"membership.coverage_rate": lambda a: a * 0.90},
        ),
        Scenario(
            "Lower contribution density (-10%)",
            "Members contribute for a smaller share of each year. Cuts the "
            "contribution base AND slows credit accrual, so fewer members vest.",
            transforms={"membership.density": lambda a: a * 0.90},
        ),
        Scenario(
            "Lower real wage growth (-1pp)",
            "Real wage growth 1 percentage point below baseline.",
            transforms={"labour.real_wage_growth": lambda a: a - 0.01},
        ),
        Scenario(
            "Higher inflation (+1pp)",
            "Inflation 1 percentage point above baseline, with price-indexed "
            "pensions in payment.",
            transforms={"labour.inflation": lambda a: a + 0.01},
        ),
        Scenario(
            "No pension indexation",
            "Pensions in payment frozen in nominal terms -- the literal SSO Act "
            "position, which contains no cost-of-living clause.",
            fields={"scheme": {"pension_indexation": "none"}},
        ),
        Scenario(
            "Wage-indexed pensions",
            "Pensions in payment indexed to wages rather than prices.",
            fields={"scheme": {"pension_indexation": "wage"}},
        ),
        # ---- Section 5 old-age reform options ------------------------------
        Scenario(
            "Retirement age 60 (phased)",
            "Normal retirement age raised from 55 to 60 over 20 years starting "
            "in year 5, with exit behaviour moving in step.",
            fields=_nra_reform(phased_nra(n_t, 55, 60, 5, 20)),
        ),
        Scenario(
            "Retirement age 65 (phased)",
            "Normal retirement age raised from 55 to 65 over 50 years -- the ILO "
            "report's central reform option.",
            fields=_nra_reform(phased_nra(n_t, 55, 65, 5, 50)),
        ),
        Scenario(
            "Ceiling frozen at THB 15,000",
            "Earnings ceiling never uprated -- the realistic status quo. Erodes "
            "the contribution base and, with a lag, benefits too.",
            fields={"scheme": {"ceiling_indexation": "none"}},
        ),
        Scenario(
            "Ceiling raised to THB 20,000",
            "Ceiling lifted at the start of the projection, then wage-indexed "
            "(ILO report Table 5.3).",
            fields={"scheme": {"max_insurable_earnings_monthly": 20_000.0}},
        ),
        Scenario(
            "Contribution rate 9%",
            "Old-age contribution rate raised from 6.35% to 9%. Note this cannot "
            "change the GAP -- the GAP is the rate the scheme requires -- but it "
            "moves the depletion year.",
            fields={"scheme": {"contribution_rate_total": 0.09}},
        ),
        Scenario(
            "Vesting cut to 120 months",
            "Vesting reduced from 180 to 120 months: more members qualify for a "
            "pension instead of a grant, converting a one-off lump sum into a "
            "lifetime liability.",
            fields={"scheme": {"vesting_months": 120}},
        ),
        Scenario(
            "Salary revaluation in reference wage",
            "Wages in the 60-month averaging window revalued to the retirement "
            "date, removing the nominal-average erosion.",
            fields={"scheme": {"revalue_past_salaries": True}},
        ),
    ]


__all__ = ["Scenario", "ScenarioSet", "compare", "default_scenarios", "phased_nra"]
