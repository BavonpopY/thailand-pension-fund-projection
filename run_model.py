#!/usr/bin/env python3
"""
Entry point -- run the ILO-PENS Thailand valuation end to end.
==============================================================

Usage
-----
Baseline only, with the consistency report and an Excel workbook::

    python run_model.py

Baseline plus the full sensitivity / reform battery::

    python run_model.py --scenarios

Options::

    --base-year 2024        Valuation (base) year
    --years 100             Length of the projection (max 100)
    --contribution-rate     Override the legal contribution rate, e.g. 0.09
    --out results           Output directory
    --no-excel              Skip the Excel workbook (CSV only)
    --no-csv                Skip the CSV bundle
    --quiet                 Suppress the console tables

Everything it prints is produced from the dummy inputs in
:mod:`ilo_pens_oldage.dummy_data`.  Replace those with real national datasets --
see ``DATA_INSERTION_GUIDE_TH.md`` -- and nothing else needs to change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ilo_pens_oldage import (
    Config,
    ScenarioSet,
    build_dummy_config,
    compare,
    default_scenarios,
    format_report,
    run_all_checks,
    run_valuation,
    write_csv_bundle,
    write_excel_report,
)
from ilo_pens_oldage.outputs import (
    actuarial_balance_table,
    demographic_table,
    financial_table,
    missing_inputs,
)


def build_config(args: argparse.Namespace) -> Config:
    """Assemble the configuration from the command line, then fill in dummy data."""
    cfg = Config()
    cfg.run.base_year = args.base_year
    cfg.run.n_years = args.years
    if args.contribution_rate is not None:
        cfg.scheme.contribution_rate_total = args.contribution_rate
    return build_dummy_config(cfg)


def print_headline(result) -> None:
    """Print the headline indicators of a completed valuation."""
    s = result.summary()
    print()
    print("=" * 78)
    print(f" {result.config.run.title}")
    print(f" Scenario: {s['scenario']}   Base year: {s['base_year']}   "
          f"Horizon: {s['final_year']}")
    print("=" * 78)
    print(f"  Old-age contribution rate            {s['contribution_rate']:>10.2%}")
    print(f"  PAYG cost rate, year 25              {s['payg_rate_at_25y']:>10.2%}")
    print(f"  PAYG cost rate, final year           {s['payg_rate_final']:>10.2%}")
    print(f"  General Average Premium (GAP)        {s['general_average_premium']:>10.2%}")
    print(f"  Rate for target reserve ratio        {s['rate_for_target_reserve_ratio']:>10.2%}")
    print(f"  Rate for target balance ratio        {s['rate_for_target_balance_ratio']:>10.2%}")
    depletion = s["depletion_year"] or "not depleted within the horizon"
    print(f"  Year of fund depletion               {str(depletion):>10}")
    print(f"  Active members, final year           {s['active_final']:>10,.0f}")
    print(f"  Pensioners, final year               {s['pensioners_final']:>10,.0f}")
    print(f"  Vested share of actives, final year  {s['vested_share_final']:>10.1%}")
    print(f"  Dependency ratio, final year         {s['dependency_ratio_final']:>10.3f}")
    print(f"  System replacement ratio, final year {s['replacement_ratio_final']:>10.1%}")
    print("=" * 78)


def print_selected_years(result, step: int = 10) -> None:
    """Print the demographic and financial projections at decade intervals."""
    demo = demographic_table(result)
    fin = financial_table(result)
    bal = actuarial_balance_table(result)

    idx = demo.index[::step]
    if demo.index[-1] not in idx:
        idx = idx.append(pd.Index([demo.index[-1]]))

    print("\nScheme demographic projection (selected years)")
    print("-" * 78)
    cols = ["contributors", "old_age_pensioners", "old_age_grants_at_retirement",
            "mean_credit_years", "vested_share", "demographic_dependency_ratio"]
    print(demo.loc[idx, cols].to_string(
        float_format=lambda v: f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"
    ))

    print("\nFinancial projection, THB millions (selected years)")
    print("-" * 78)
    cols = ["insurable_earnings", "contribution_income", "investment_income",
            "old_age_pension_outflow", "retirement_grant_outflow",
            "total_expenditure", "reserve_end_of_year"]
    print(fin.loc[idx, cols].to_string(float_format=lambda v: f"{v:,.0f}"))

    print("\nActuarial balance (selected years)")
    print("-" * 78)
    cols = ["payg_cost_rate", "contribution_rate",
            "reserve_to_expenditure_F_over_S", "expenditure_pct_gdp"]
    print(bal.loc[idx, cols].to_string(float_format=lambda v: f"{v:,.4f}"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ILO-PENS Thailand -- long-term actuarial valuation."
    )
    parser.add_argument("--base-year", type=int, default=2024)
    parser.add_argument("--years", type=int, default=100)
    parser.add_argument("--contribution-rate", type=float, default=None,
                        help="Override the OLD-AGE branch contribution rate, e.g. 0.09.")
    parser.add_argument("--scenarios", action="store_true",
                        help="Run the full sensitivity and reform battery.")
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--no-excel", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg = build_config(args)

    missing = missing_inputs(cfg)
    n_missing = int((missing["status"] == "MISSING").sum())
    print(f"Data inputs: {len(missing) - n_missing}/{len(missing)} supplied "
          f"({n_missing} still missing).")
    print("NOTE: all supplied values are DUMMY placeholders unless you have "
          "replaced them.\n      See DATA_INSERTION_GUIDE_TH.md.")

    result = run_valuation(cfg)
    print_headline(result)

    checks = run_all_checks(result)
    print(format_report(checks))

    if not args.quiet:
        print_selected_years(result)

    comparison = None
    if args.scenarios:
        print("\nRunning sensitivity and reform scenarios...")
        results = ScenarioSet(default_scenarios(cfg.run.n_t)).run(cfg)
        comparison = compare(results)
        print("\nScenario comparison")
        print("-" * 78)
        show = ["general_average_premium", "payg_rate_final", "depletion_year",
                "gap_diff_pp"]
        show = [c for c in show if c in comparison.columns]
        print(comparison[show].to_string(
            float_format=lambda v: f"{v:,.4f}" if abs(v) < 1000 else f"{v:,.0f}"
        ))

    out_dir = Path(args.out)
    if not args.no_csv:
        files = write_csv_bundle(result, out_dir / "csv")
        print(f"\nWrote {len(files)} CSV files to {out_dir / 'csv'}")
    if not args.no_excel:
        xlsx = write_excel_report(
            result, out_dir / "ilo_pens_oldage_valuation.xlsx",
            scenario_comparison=comparison, checks=checks,
        )
        print(f"Wrote Excel workbook to {xlsx}")

    failed = [c for c in checks if not c.passed]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
