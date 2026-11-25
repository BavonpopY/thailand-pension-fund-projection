#!/usr/bin/env python3
"""
Excel-driven entry point -- create the input template, then run from it.
========================================================================

Two commands are all you need.

**1. Create the blank-but-pre-filled input workbook**

    python run_from_excel.py --create ILO_PENS_INPUTS.xlsx

Every input cell is written with the current dummy value and shaded amber,
meaning *placeholder -- replace me*.  Each sheet's header block carries the
config path, the unit and the Thai source agency.  Add ``--detail`` if you have
the full 330 x 51 joint distribution of age and accumulated credit and want the
``*_Detail`` membership sheets instead of the marginals.

**2. Fill it in, then run**

    python run_from_excel.py --inputs ILO_PENS_INPUTS.xlsx --out results

Reads the workbook, runs the six-module engine, executes the eleven consistency
checks, and writes ``results/ilo_pens_oldage_valuation.xlsx`` plus a CSV
bundle.  Add ``--scenarios`` for the sensitivity and reform battery.

**Check progress at any time**

    python run_from_excel.py --status ILO_PENS_INPUTS.xlsx

The model runs at every stage of completion, so you can fill in one sheet,
re-run, and watch the numbers move.  Nothing is ever silently defaulted: if a
cell is blank or non-numeric the reader stops and names the sheet and cell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from ilo_pens_oldage.workbook_io import (
    read_input_workbook,
    workbook_status,
    write_input_workbook,
)


def cmd_create(path: Path, base_year: int, years: int, detail: bool) -> int:
    """Write the input template, pre-filled with dummy placeholders."""
    cfg = Config()
    cfg.run.base_year = base_year
    cfg.run.n_years = years
    cfg = build_dummy_config(cfg)

    out = write_input_workbook(cfg, path, detail=detail)
    size_mb = out.stat().st_size / 1e6

    print(f"Created {out}  ({size_mb:.1f} MB)")
    print()
    print("  Base year          ", base_year)
    print("  Projection length  ", years, "years")
    print("  Membership mode    ", "detail (330 x 51)" if detail else "marginal")
    print()
    print("  Every amber cell is a DUMMY placeholder. Replace it with real data.")
    print("  Read the _README sheet first, then work through _STATUS sheet by sheet.")
    print()
    print(f"  When ready:  python run_from_excel.py --inputs {out} --out results")
    return 0


def cmd_status(path: Path, check_fills: bool) -> int:
    """Print the per-sheet data-collection status."""
    df = workbook_status(path, check_fills=check_fills)
    counts = df["status"].value_counts()

    print(f"Data-collection status for {path}")
    print("=" * 78)
    for status in ("VERIFIED", "IN PROGRESS", "TO REPLACE"):
        print(f"  {status:<14} {int(counts.get(status, 0)):>3} sheets")
    print("=" * 78)
    print()

    cols = ["sheet", "status", "source_agency"]
    if check_fills:
        cols.append("placeholder_cells")
    remaining = df[df["status"] != "VERIFIED"]
    if remaining.empty:
        print("All sheets marked VERIFIED.")
    else:
        print("Still to do:")
        print(remaining[cols].to_string(index=False))
    return 0


def cmd_run(inputs: Path, out_dir: Path, scenarios: bool, quiet: bool) -> int:
    """Read the workbook, run the valuation, write the outputs."""
    print(f"Reading {inputs} ...")
    try:
        cfg = read_input_workbook(inputs)
    except ValueError as exc:
        print(f"\nERROR reading the workbook:\n  {exc}\n", file=sys.stderr)
        print("Fix that cell and run again. Nothing was calculated.", file=sys.stderr)
        return 2

    status = workbook_status(inputs)
    n_verified = int((status["status"] == "VERIFIED").sum())
    n_total = len(status)
    print(f"  Base year {cfg.run.base_year}, {cfg.run.n_years}-year projection.")
    print(f"  {n_verified}/{n_total} sheets marked VERIFIED.")
    if n_verified < n_total:
        print("  WARNING: the remaining sheets still hold dummy placeholders, so "
              "these results are NOT a valuation.")

    result = run_valuation(cfg)
    s = result.summary()

    print()
    print("=" * 78)
    print(f" {cfg.run.title}")
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
    print(f"  Old-age pensioners, final year       {s['pensioners_final']:>10,.0f}")
    print(f"  Vested share of actives, final year  {s['vested_share_final']:>10.1%}")
    print("=" * 78)

    checks = run_all_checks(result)
    print(format_report(checks))

    comparison = None
    if scenarios:
        print("\nRunning sensitivity and reform scenarios ...")
        results = ScenarioSet(default_scenarios(cfg.run.n_t)).run(cfg)
        comparison = compare(results)
        show = [c for c in ["general_average_premium", "payg_rate_final",
                            "depletion_year", "gap_diff_pp"]
                if c in comparison.columns]
        print("\nScenario comparison")
        print("-" * 78)
        print(comparison[show].to_string(
            float_format=lambda v: f"{v:,.4f}" if abs(v) < 1000 else f"{v:,.0f}"
        ))

    files = write_csv_bundle(result, out_dir / "csv")
    xlsx = write_excel_report(result, out_dir / "ilo_pens_oldage_valuation.xlsx",
                              scenario_comparison=comparison, checks=checks)
    print(f"\nWrote {len(files)} CSV files to {out_dir / 'csv'}")
    print(f"Wrote Excel workbook to {xlsx}")

    return 1 if any(not c.passed for c in checks) else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="ILO-PENS Thailand -- Excel-driven valuation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--create", metavar="FILE",
                   help="Write a fresh input template to FILE.")
    p.add_argument("--inputs", metavar="FILE",
                   help="Run the valuation from a filled-in input workbook.")
    p.add_argument("--status", metavar="FILE",
                   help="Print the data-collection status of an input workbook.")
    p.add_argument("--out", default="results", help="Output directory.")
    p.add_argument("--base-year", type=int, default=2024,
                   help="Base year for --create.")
    p.add_argument("--years", type=int, default=100,
                   help="Projection length for --create.")
    p.add_argument("--detail", action="store_true",
                   help="With --create, also write the 330 x 51 membership sheets.")
    p.add_argument("--check-fills", action="store_true",
                   help="With --status, also count remaining amber cells.")
    p.add_argument("--scenarios", action="store_true",
                   help="With --inputs, run the sensitivity and reform battery.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.create:
        return cmd_create(Path(args.create), args.base_year, args.years, args.detail)
    if args.status:
        return cmd_status(Path(args.status), args.check_fills)
    if args.inputs:
        return cmd_run(Path(args.inputs), Path(args.out), args.scenarios, args.quiet)

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
