"""
Output layer -- ILO-style result tables, CSV and formatted Excel.
=================================================================

Reproduces the reporting structure of an ILO actuarial valuation report
(ILO/TF/THAILAND/R.40 Tables 3.12-3.21), restricted to the Old-Age branch.

===========================  =================================================
Sheet / file                 Contents
===========================  =================================================
``Summary``                  Headline indicators and the run's assumptions
``Demographic_Projection``   Contributors, pensioners, vesting, dependency
``Financial_Projection``     Cash inflows, outflows and the reserve
``Actuarial_Balance``        PAYG rate, reserve ratio, expenditure / GDP
``Population``               National population, births, deaths, OADR
``Labour_Market``            Labour force, employment, macro assumptions
``Credit_Distribution``      Active members by accumulated credit
``Active_By_Age``            Active members by sex and age
``Pensioners_By_Age``        Old-age pensioners by sex and age
``Scenarios``                Side-by-side scenario comparison
``Data_Inputs_Required``     The data-insertion checklist with status
``Validation``               Consistency-check report
===========================  =================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import AGE_MAX_INS, AGE_MIN_INS, Config
from .engine import ValuationResult
from .validation import Check


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def demographic_table(result: ValuationResult) -> pd.DataFrame:
    """Scheme demographic projection (ILO report Table 3.12 / 3.13 shape)."""
    return pd.DataFrame(
        {
            "active_members": result.active_total,
            "inactive_members": result.inactive_total,
            "contributors": result.contributors_total,
            "new_entrants": result.new_entrants_total,
            "old_age_pensioners": result.pensioners_total,
            "new_old_age_pensioners": result.new_pensioners,
            "old_age_grants_at_retirement": result.retirement_grants_n,
            "old_age_grants_on_death": result.death_grants_n,
            "mean_credit_years": result.mean_credit_active,
            "vested_share": result.vested_share_active,
            "demographic_dependency_ratio": result.finance.dependency_ratio,
            "system_replacement_ratio": result.finance.replacement_ratio,
        },
        index=pd.Index(result.years, name="year"),
    )


def financial_table(result: ValuationResult, unit: float = 1e6) -> pd.DataFrame:
    """Cash inflows, outflows and the reserve (ILO report Table 3.20 shape).

    Parameters
    ----------
    unit
        Divisor applied to monetary columns.  Default 1e6 = THB millions.
    """
    f = result.finance
    return pd.DataFrame(
        {
            "insurable_earnings": f.insurable_earnings / unit,
            "contribution_income": f.contribution_income / unit,
            "investment_income": f.investment_income / unit,
            "total_income": f.total_income / unit,
            "old_age_pension_outflow": result.pension_expenditure / unit,
            "retirement_grant_outflow": result.retirement_grant_expenditure / unit,
            "death_grant_outflow": result.death_grant_expenditure / unit,
            "guarantee_lump_outflow": result.guarantee_lump_expenditure / unit,
            "admin_expenditure": result.admin_expenditure / unit,
            "total_expenditure": f.total_expenditure / unit,
            "net_cash_flow": (f.total_income - f.total_expenditure) / unit,
            "reserve_end_of_year": f.reserve / unit,
            "avg_insurable_earnings_annual": result.avg_insurable_earnings,
            "avg_pension_annual": result.avg_pension,
        },
        index=pd.Index(result.years, name="year"),
    )


def actuarial_balance_table(result: ValuationResult) -> pd.DataFrame:
    """PAYG cost rate, funding ratios and the financing-system indicators."""
    f = result.finance
    return pd.DataFrame(
        {
            "payg_cost_rate": f.payg_rate,
            "contribution_rate": f.contribution_rate,
            "reserve_ratio_F_prev_over_S": f.reserve_ratio,
            "reserve_to_expenditure_F_over_S": f.reserve_to_expenditure,
            "balance_ratio": f.balance_ratio,
            "expenditure_pct_gdp": f.expenditure_pct_gdp,
            "reserve_pct_gdp": f.reserve_pct_gdp,
        },
        index=pd.Index(result.years, name="year"),
    )


def population_table(result: ValuationResult) -> pd.DataFrame:
    """National population projection outputs."""
    d = result.demography
    return pd.DataFrame(
        {
            "total_population": d.total,
            "population_15_64": d.working_age,
            "population_65_plus": d.population[:, 65:, :].sum(axis=(0, 1)),
            "births": d.births,
            "deaths": d.deaths,
            "net_migration": d.net_migration,
            "old_age_dependency_ratio": d.old_age_dependency_ratio,
        },
        index=pd.Index(result.years, name="year"),
    )


def labour_table(result: ValuationResult) -> pd.DataFrame:
    """Labour market and macroeconomic assumptions actually used."""
    l = result.labour
    return pd.DataFrame(
        {
            "labour_force": l.total_labour_force,
            "employed": l.total_employed,
            "aggregate_lfpr": l.aggregate_lfpr,
            "unemployment_rate": l.aggregate_unemployment,
            "inflation": l.inflation,
            "real_wage_growth": l.real_wage_growth,
            "nominal_wage_growth": l.nominal_wage_growth,
            "real_gdp_growth": l.real_gdp_growth,
            "nominal_interest_rate": l.nominal_interest_rate,
            "price_index": l.price_index,
            "wage_index": l.wage_index,
            "nominal_gdp_thb_mn": l.gdp_nominal / 1e6,
            "insurable_ceiling_annual_thb": result.earnings.ceiling_annual,
            "insurable_floor_annual_thb": result.earnings.floor_annual,
        },
        index=pd.Index(result.years, name="year"),
    )


def _selected_years(result: ValuationResult, step: int = 10) -> List[int]:
    """Indices of the years to show in wide cross-section tables."""
    n_t = len(result.years)
    positions = list(range(0, n_t, step))
    if positions[-1] != n_t - 1:
        positions.append(n_t - 1)
    return positions


def credit_distribution_table(result: ValuationResult, step: int = 10) -> pd.DataFrame:
    """Active members by accumulated credit, for selected years."""
    positions = _selected_years(result, step)
    data = {int(result.years[p]): result.credit_distribution[p] for p in positions}
    return pd.DataFrame(
        data,
        index=pd.Index(np.arange(result.credit_distribution.shape[1]),
                       name="credit_years"),
    )


def _by_age_table(values: np.ndarray, ages: np.ndarray, result: ValuationResult,
                  step: int) -> pd.DataFrame:
    positions = _selected_years(result, step)
    frames = []
    for sex_idx, sex in enumerate(("male", "female")):
        block = pd.DataFrame(
            {int(result.years[p]): values[sex_idx, :, p] for p in positions},
            index=pd.Index(ages, name="age"),
        )
        block.insert(0, "sex", sex)
        frames.append(block)
    return pd.concat(frames).reset_index().set_index(["sex", "age"]).sort_index()


def active_by_age_table(result: ValuationResult, step: int = 10) -> pd.DataFrame:
    """Active members by sex and single year of age, for selected years."""
    ages = np.arange(AGE_MIN_INS, AGE_MAX_INS + 1)
    return _by_age_table(result.active_by_sex_age, ages, result, step)


def pensioners_by_age_table(result: ValuationResult, step: int = 10) -> pd.DataFrame:
    """Old-age pensioners by sex and single year of age, for selected years."""
    ages = np.arange(result.pensioners_by_sex_age.shape[1])
    out = _by_age_table(result.pensioners_by_sex_age, ages, result, step)
    return out.loc[(slice(None), slice(40, 100)), :]


def summary_table(result: ValuationResult) -> pd.DataFrame:
    """Headline indicators plus the key assumptions, as a two-column sheet."""
    cfg, s, l = result.config, result.summary(), result.labour
    rows = [
        ("Scenario", s["scenario"]),
        ("Scope", "Old-Age Benefit Branch only (pension + lump-sum grant)"),
        ("Base year", s["base_year"]),
        ("Final projection year", s["final_year"]),
        ("", ""),
        ("--- Scheme design ---", ""),
        ("Normal retirement age (base year)", cfg.scheme.nra_for_year(0)),
        ("Vesting period (months)", cfg.scheme.vesting_months),
        ("Accrual: base rate", f"{cfg.scheme.accrual_base_rate:.1%}"),
        ("Accrual: increment per extra year", f"{cfg.scheme.accrual_increment_rate:.2%}"),
        ("Reference salary window (years)", cfg.scheme.final_salary_years),
        ("Reference salary revalued", cfg.scheme.revalue_past_salaries),
        ("Pension indexation", cfg.scheme.pension_indexation),
        ("Insurable earnings floor (THB/month)", cfg.scheme.min_insurable_earnings_monthly),
        ("Insurable earnings ceiling (THB/month)", cfg.scheme.max_insurable_earnings_monthly),
        ("Ceiling indexation", cfg.scheme.ceiling_indexation),
        ("Old-age contribution rate", f"{cfg.scheme.contribution_rate_total:.2%}"),
        ("Collection rate", cfg.scheme.collection_rate),
        ("Admin expense rate (% insurable earnings)", cfg.scheme.admin_expense_rate),
        ("Grant on pre-retirement death", cfg.scheme.pay_grant_on_pre_retirement_death),
        ("Pensioner death guarantee (months)", cfg.scheme.pensioner_death_guarantee_months),
        ("", ""),
        ("--- Ultimate economic assumptions ---", ""),
        ("Inflation", f"{l.inflation[-1]:.2%}"),
        ("Real wage growth", f"{l.real_wage_growth[-1]:.2%}"),
        ("Real GDP growth", f"{l.real_gdp_growth[-1]:.2%}"),
        ("Nominal investment return", f"{l.nominal_interest_rate[-1]:.2%}"),
        ("Ultimate TFR", f"{np.asarray(cfg.demography.tfr_path)[-1]:.2f}"),
        ("", ""),
        ("--- Results ---", ""),
        ("Initial reserve (THB mn)", cfg.finance.initial_reserve / 1e6),
        ("PAYG cost rate, year 25", f"{s['payg_rate_at_25y']:.2%}"),
        ("PAYG cost rate, final year", f"{s['payg_rate_final']:.2%}"),
        ("General Average Premium", f"{s['general_average_premium']:.2%}"),
        (f"Rate for reserve ratio {cfg.finance.target_reserve_ratio:g}",
         f"{s['rate_for_target_reserve_ratio']:.2%}"),
        (f"Rate for balance ratio {cfg.finance.target_balance_ratio:g}",
         f"{s['rate_for_target_balance_ratio']:.2%}"),
        ("Year of fund depletion", s["depletion_year"] or "not depleted"),
        ("Active members, final year", f"{s['active_final']:,.0f}"),
        ("Old-age pensioners, final year", f"{s['pensioners_final']:,.0f}"),
        ("Vested share of actives, final year", f"{s['vested_share_final']:.1%}"),
        ("Dependency ratio, final year", f"{s['dependency_ratio_final']:.3f}"),
        ("System replacement ratio, final year", f"{s['replacement_ratio_final']:.1%}"),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


# ---------------------------------------------------------------------------
# Data-insertion checklist
# ---------------------------------------------------------------------------

DATA_INPUT_CHECKLIST: List[tuple[str, str, str, str, str]] = [
    # (config path, description, required format / shape, unit, Thai source)
    ("demography.base_population", "National population at the valuation date",
     "ndarray (2, 101) [sex, age 0-100]", "persons", "NESDC / NSO"),
    ("demography.mortality_qx", "General-population mortality q(x,t)",
     "ndarray (2, 101, T)", "probability", "NESDC / UN WPP / OIC TMO"),
    ("demography.fertility_asfr_base", "Age-specific fertility rates, base year",
     "ndarray (101,)", "births per woman", "NSO vital registration"),
    ("demography.tfr_path", "Total fertility rate by year", "ndarray (T,)",
     "children per woman", "NESDC projection"),
    ("demography.net_migration_total", "Net migration by year", "ndarray (T,)",
     "persons/year", "NSO / Department of Employment"),
    ("demography.migration_age_profile", "Sex-age distribution of net migration",
     "ndarray (2, 101), sums to 1", "proportion", "NSO / DOE"),
    ("labour.lfpr", "Labour force participation rate", "ndarray (2, 101, T)",
     "proportion", "NSO Labour Force Survey"),
    ("labour.unemployment_rate", "Unemployment rate", "ndarray (2, 101, T)",
     "proportion", "NSO Labour Force Survey"),
    ("labour.inflation", "CPI inflation", "ndarray (T,)", "proportion/year",
     "MOC / BOT / IMF WEO"),
    ("labour.real_wage_growth", "Real increase in average insurable earnings",
     "ndarray (T,)", "proportion/year", "NESDC productivity / SSO wage data"),
    ("labour.real_gdp_growth", "Real GDP growth", "ndarray (T,)", "proportion/year",
     "NESDC / IMF WEO"),
    ("labour.base_gdp", "Nominal GDP in the base year", "float", "THB", "NESDC"),
    ("labour.real_investment_return", "Real return on scheme assets", "ndarray (T,)",
     "proportion/year", "SSO Investment Bureau / BOT yield curve"),
    ("scheme.min_insurable_earnings_monthly", "Statutory floor on the contribution base",
     "float", "THB/month", "SSO Act / ministerial regulation"),
    ("scheme.max_insurable_earnings_monthly", "Statutory ceiling on the contribution base",
     "float", "THB/month", "SSO Act / ministerial regulation"),
    ("scheme.contribution_rate_total",
     "Contribution rate allocated to the OLD-AGE branch (net of child allowance)",
     "float or ndarray (T,)", "proportion", "SSO Act / ILO branch allocation"),
    ("scheme.collection_rate", "Contribution collection / compliance rate",
     "float or ndarray (T,)", "proportion", "SSO financial statements"),
    ("scheme.nra_path", "Normal retirement age by year (OPTIONAL -- phased reform only)",
     "ndarray (T,) int", "age", "SSO Act / reform proposal"),
    ("scheme.admin_expense_rate", "Administrative expenditure",
     "float or ndarray (T,)", "% of insurable earnings", "SSO financial statements"),
    ("membership.active_base", "Active members by sex, income band, age, credit",
     "ndarray (2, 3, 55, 51)", "persons", "SSO member register"),
    ("membership.inactive_base", "Inactive members by sex, income, age, credit",
     "ndarray (2, 3, 55, 51)", "persons", "SSO member register"),
    ("membership.old_age_pensioners_base", "Old-age pensioners at the valuation date",
     "ndarray (2, 101)", "persons", "SSO benefit register"),
    ("membership.old_age_pension_base_monthly", "Average old-age pension in payment",
     "ndarray (2, 101)", "THB/month", "SSO benefit register"),
    ("membership.coverage_rate", "Insured / employed coverage rate",
     "ndarray (2, 55, T)", "proportion", "SSO insured count / NSO LFS employed"),
    ("membership.density", "Contribution density (contributors / active)",
     "ndarray (2, 55, T)", "proportion", "SSO contribution records"),
    ("membership.leaving_rate", "Probability of becoming inactive",
     "ndarray (2, 55, T)", "probability", "SSO member-register experience"),
    ("membership.retirement_rate", "Share of exits claiming an old-age benefit",
     "ndarray (2, 55, T)", "proportion", "SSO award experience"),
    ("membership.reentry_rate", "Re-entry probability from the inactive pool",
     "ndarray (2, 55, T)", "probability", "SSO member-register experience"),
    ("membership.insured_mortality_qx", "Mortality of the insured population",
     "ndarray (2, 101, T)", "probability", "SSO mortality experience / OIC TMO"),
    ("membership.new_entrant_share", "New entrants as a share of required inflow",
     "ndarray (55, T)", "proportion", "SSO registration statistics"),
    ("membership.salary_scale_monthly", "Average insurable earnings by band and age",
     "ndarray (2, 3, 55)", "THB/month", "SSO declared wages"),
    ("membership.income_level_shares", "Share of members in each income band",
     "ndarray (3,)", "proportion", "SSO earnings distribution"),
    ("finance.initial_reserve", "Old-age branch reserve at the valuation date",
     "float", "THB", "SSO audited financial statements"),
]

OPTIONAL_INPUTS = {"scheme.nra_path"}


def data_input_checklist() -> pd.DataFrame:
    """The full data-insertion guide as a DataFrame."""
    return pd.DataFrame(
        DATA_INPUT_CHECKLIST,
        columns=["config_path", "parameter_description", "required_format_shape",
                 "unit", "thai_source_agency"],
    )


def missing_inputs(cfg: Config) -> pd.DataFrame:
    """Report which documented inputs are still unset on a configuration.

    Parameters
    ----------
    cfg
        Configuration to audit.

    Returns
    -------
    pandas.DataFrame
        The checklist with a ``status`` column.
    """
    df = data_input_checklist().copy()
    statuses = []
    for path in df["config_path"]:
        block_name, _, field_name = path.partition(".")
        block = getattr(cfg, block_name, None)
        value = getattr(block, field_name, None) if block is not None else None
        if value is not None:
            statuses.append("SUPPLIED")
        else:
            statuses.append("OPTIONAL (unset)" if path in OPTIONAL_INPUTS else "MISSING")
    df["status"] = statuses
    return df


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _all_tables(result: ValuationResult) -> Dict[str, pd.DataFrame]:
    return {
        "Summary": summary_table(result),
        "Demographic_Projection": demographic_table(result),
        "Financial_Projection": financial_table(result),
        "Actuarial_Balance": actuarial_balance_table(result),
        "Population": population_table(result),
        "Labour_Market": labour_table(result),
        "Credit_Distribution": credit_distribution_table(result),
        "Active_By_Age": active_by_age_table(result),
        "Pensioners_By_Age": pensioners_by_age_table(result),
    }


def write_csv_bundle(result: ValuationResult, out_dir: str | Path) -> List[Path]:
    """Write every result table as a CSV file.

    Parameters
    ----------
    result
        Completed valuation.
    out_dir
        Destination directory; created if it does not exist.

    Returns
    -------
    list of Path
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = _all_tables(result)
    tables["Data_Inputs_Required"] = missing_inputs(result.config)

    written = []
    for name, df in tables.items():
        path = out / f"{name.lower()}.csv"
        df.to_csv(path)
        written.append(path)
    return written


def write_excel_report(
    result: ValuationResult,
    path: str | Path,
    scenario_comparison: Optional[pd.DataFrame] = None,
    checks: Optional[Sequence[Check]] = None,
) -> Path:
    """Write a formatted, ILO-style Excel workbook.

    Parameters
    ----------
    result
        Completed baseline valuation.
    path
        Destination ``.xlsx`` file.
    scenario_comparison
        Optional output of :func:`~ilo_pens_oldage.scenarios.compare`.
    checks
        Optional output of :func:`~ilo_pens_oldage.validation.run_all_checks`.

    Returns
    -------
    Path
    """
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sheets = _all_tables(result)
    if scenario_comparison is not None:
        sheets["Scenarios"] = scenario_comparison.reset_index()
    sheets["Data_Inputs_Required"] = missing_inputs(result.config)
    if checks is not None:
        sheets["Validation"] = pd.DataFrame(
            [{"check": c.name, "result": "PASS" if c.passed else "FAIL",
              "detail": c.detail, "max_error": c.max_error, "tolerance": c.tolerance}
             for c in checks]
        )

    plain = {"Summary", "Scenarios", "Validation", "Data_Inputs_Required"}

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=name not in plain)

        header_fill = PatternFill("solid", fgColor="1F3864")
        header_font = Font(color="FFFFFF", bold=True, size=10)
        body_font = Font(size=10)

        for name, df in sheets.items():
            ws = writer.sheets[name]
            ws.freeze_panes = "B2"
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center",
                                           wrap_text=True)

            for col_idx in range(1, ws.max_column + 1):
                letter = get_column_letter(col_idx)
                header = str(ws.cell(row=1, column=col_idx).value or "")
                ws.column_dimensions[letter].width = max(12, min(32, len(header) + 3))

                if name in plain:
                    continue
                low = header.lower()
                if any(k in low for k in ("rate", "ratio", "share", "pct", "growth",
                                          "inflation", "lfpr", "premium")):
                    fmt = "0.00%"
                elif any(k in low for k in ("index", "credit_years", "mean_credit")):
                    fmt = "0.000"
                else:
                    fmt = "#,##0"
                for row_idx in range(2, ws.max_row + 1):
                    cell = ws.cell(row=row_idx, column=col_idx)
                    cell.font = body_font
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = fmt
            ws.row_dimensions[1].height = 32

        ws = writer.sheets["Summary"]
        ws.column_dimensions["A"].width = 44
        ws.column_dimensions["B"].width = 26

    return path


__all__ = [
    "demographic_table", "financial_table", "actuarial_balance_table",
    "population_table", "labour_table", "credit_distribution_table",
    "active_by_age_table", "pensioners_by_age_table", "summary_table",
    "data_input_checklist", "missing_inputs", "write_csv_bundle",
    "write_excel_report", "DATA_INPUT_CHECKLIST",
]
