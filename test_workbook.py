#!/usr/bin/env python3
"""
Self-test for the Excel input workbook (Old-Age branch).
=========================================================

Run with::

    python test_workbook.py

Verifies four things:

1. **Round-trip fidelity.**  Arrays with no year axis must come back bit-exact.
   Year-varying arrays come back through anchor interpolation, so the test
   reports the error rather than demanding zero.
2. **Marginal reconstruction.**  In ``marginal`` mode the model rebuilds the
   credit distribution from the member counts and mean credit you type in.  The
   *marginals themselves* -- head count by age and sex, and mean credit by age
   and sex -- must survive exactly, because those are what you actually entered.
3. **Detail mode.**  With the ``*_Detail`` sheets the full 330 x 51 array must
   come back bit-exact.
4. **Error handling.**  A blank or non-numeric cell must raise a message naming
   the sheet and the cell, and must not silently default.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from ilo_pens_oldage import Config, build_dummy_config, run_valuation
from ilo_pens_oldage.config import N_CREDIT
from ilo_pens_oldage.workbook_io import (
    read_input_workbook,
    workbook_status,
    write_input_workbook,
)

CREDITS = np.arange(N_CREDIT, dtype=float)


def _marginals(active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Head count and mean credit by sex and insurable age."""
    counts = active.sum(axis=(1, 3))
    weighted = (active * CREDITS[None, None, None, :]).sum(axis=(1, 3))
    mean = np.divide(weighted, counts, out=np.zeros_like(counts), where=counts > 0)
    return counts, mean


def _max_rel(a: np.ndarray, b: np.ndarray, floor: float = 1e-9) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((np.abs(a - b) / np.maximum(np.abs(a), floor)).max())


def main() -> int:
    failures: list[str] = []
    tmp = Path(tempfile.mkdtemp())

    direct = build_dummy_config(Config())

    # ---- 1. round trip, marginal mode ----------------------------------
    print("1. Round-trip fidelity (marginal mode)")
    print("-" * 70)
    write_input_workbook(direct, tmp / "m.xlsx", detail=False)
    back = read_input_workbook(tmp / "m.xlsx")

    exact = [
        ("demography.base_population", direct.demography.base_population,
         back.demography.base_population),
        ("demography.migration_age_profile", direct.demography.migration_age_profile,
         back.demography.migration_age_profile),
        ("demography.fertility_asfr_base", direct.demography.fertility_asfr_base,
         back.demography.fertility_asfr_base),
        ("membership.salary_scale_monthly", direct.membership.salary_scale_monthly,
         back.membership.salary_scale_monthly),
        ("membership.income_level_shares", direct.membership.income_level_shares,
         back.membership.income_level_shares),
        ("membership.old_age_pensioners_base",
         direct.membership.old_age_pensioners_base,
         back.membership.old_age_pensioners_base),
        ("membership.old_age_pension_base_monthly",
         direct.membership.old_age_pension_base_monthly,
         back.membership.old_age_pension_base_monthly),
    ]
    for name, a, b in exact:
        err = _max_rel(a, b)
        ok = err < 1e-9
        print(f"   {'OK ' if ok else 'FAIL'} {name:<44} {err:.3e}")
        if not ok:
            failures.append(f"{name} not exact ({err:.3e})")

    print("\n   Year-varying arrays (via anchor interpolation -- error expected):")
    interpolated = [
        ("demography.mortality_qx", direct.demography.mortality_qx,
         back.demography.mortality_qx),
        ("demography.tfr_path", direct.demography.tfr_path, back.demography.tfr_path),
        ("labour.inflation", direct.labour.inflation, back.labour.inflation),
        ("labour.real_wage_growth", direct.labour.real_wage_growth,
         back.labour.real_wage_growth),
        ("membership.coverage_rate", direct.membership.coverage_rate,
         back.membership.coverage_rate),
        ("membership.density", direct.membership.density, back.membership.density),
    ]
    for name, a, b in interpolated:
        print(f"        {name:<44} {_max_rel(a, b):.3e}")

    # ---- 2. marginal reconstruction -------------------------------------
    print("\n2. Marginal reconstruction (what you typed must survive)")
    print("-" * 70)
    c0, m0 = _marginals(np.asarray(direct.membership.active_base, float))
    c1, m1 = _marginals(np.asarray(back.membership.active_base, float))
    for label, a, b in (("head count by age and sex", c0, c1),
                        ("mean credit by age and sex", m0, m1)):
        err = _max_rel(a, b, floor=1e-6)
        ok = err < 1e-6
        print(f"   {'OK ' if ok else 'FAIL'} active   {label:<34} {err:.3e}")
        if not ok:
            failures.append(f"active {label} not preserved ({err:.3e})")

    c0, m0 = _marginals(np.asarray(direct.membership.inactive_base, float))
    c1, m1 = _marginals(np.asarray(back.membership.inactive_base, float))
    for label, a, b in (("head count by age and sex", c0, c1),
                        ("mean credit by age and sex", m0, m1)):
        err = _max_rel(a, b, floor=1e-6)
        ok = err < 1e-6
        print(f"   {'OK ' if ok else 'FAIL'} inactive {label:<34} {err:.3e}")
        if not ok:
            failures.append(f"inactive {label} not preserved ({err:.3e})")

    print("\n   NOTE: the cell-by-cell credit distribution is NOT preserved in")
    print("   marginal mode -- it is rebuilt from the mean you supply. Use")
    print("   detail mode if the SSO gives you the full joint distribution.")

    # ---- 3. detail mode --------------------------------------------------
    print("\n3. Detail mode (full 330 x 51 joint distribution)")
    print("-" * 70)
    write_input_workbook(direct, tmp / "d.xlsx", detail=True)
    back_d = read_input_workbook(tmp / "d.xlsx")
    for name, a, b in (("active_base", direct.membership.active_base,
                        back_d.membership.active_base),
                       ("inactive_base", direct.membership.inactive_base,
                        back_d.membership.inactive_base)):
        err = _max_rel(a, b, floor=1e-6)
        ok = err < 1e-9
        print(f"   {'OK ' if ok else 'FAIL'} membership.{name:<40} {err:.3e}")
        if not ok:
            failures.append(f"detail {name} not exact ({err:.3e})")

    # ---- 4. error handling ----------------------------------------------
    print("\n4. Error handling")
    print("-" * 70)
    from openpyxl import load_workbook

    wb = load_workbook(tmp / "m.xlsx")
    wb["Density_M"].cell(row=12, column=4).value = None
    wb.save(tmp / "blank.xlsx")
    try:
        read_input_workbook(tmp / "blank.xlsx")
        print("   FAIL blank cell was silently accepted")
        failures.append("blank cell not caught")
    except ValueError as exc:
        named = "Density_M" in str(exc) and "12" in str(exc)
        print(f"   {'OK ' if named else 'FAIL'} blank cell -> {exc}")
        if not named:
            failures.append("blank-cell error does not name the sheet and cell")

    wb = load_workbook(tmp / "m.xlsx")
    wb["Coverage_F"].cell(row=20, column=3).value = "n/a"
    wb.save(tmp / "text.xlsx")
    try:
        read_input_workbook(tmp / "text.xlsx")
        print("   FAIL text cell was silently accepted")
        failures.append("text cell not caught")
    except ValueError as exc:
        named = "Coverage_F" in str(exc)
        print(f"   {'OK ' if named else 'FAIL'} text cell  -> {exc}")
        if not named:
            failures.append("text-cell error does not name the sheet")

    # ---- 5. end-to-end impact -------------------------------------------
    print("\n5. Effect on the headline result")
    print("-" * 70)
    g_direct = run_valuation(direct).finance.general_average_premium
    g_excel = run_valuation(back).finance.general_average_premium
    diff_pp = abs(g_direct - g_excel) * 100
    print(f"   GAP, direct config    {g_direct:.4%}")
    print(f"   GAP, via the workbook {g_excel:.4%}")
    print(f"   difference            {diff_pp:.3f} pp  (anchor interpolation)")
    if diff_pp > 0.25:
        failures.append(f"anchor interpolation costs {diff_pp:.3f} pp on the GAP")
        print("   FAIL exceeds the 0.25 pp tolerance -- add anchor years.")
    else:
        print("   OK   within the 0.25 pp tolerance.")

    status = workbook_status(tmp / "m.xlsx")
    print(f"\n   _STATUS sheet lists {len(status)} sheets, "
          f"{int((status['status'] == 'TO REPLACE').sum())} marked TO REPLACE.")

    print("\n" + "=" * 70)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All workbook tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
