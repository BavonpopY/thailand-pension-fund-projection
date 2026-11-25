"""
Excel input workbook -- template writer, reader and status report.
==================================================================

**Old-Age Benefit Branch only.**  There are no invalidity-rate or
family-structure sheets, and ``PensionersBase`` carries only the old-age
pensioner stock -- those inputs are not needed and are not read.

Lets the whole model be driven from a single spreadsheet.  You never have to
open a ``.py`` file: fill in ``ILO_PENS_INPUTS.xlsx``, run one command, and the
results come back as another workbook.

The three entry points
----------------------
:func:`write_input_workbook`
    Create the template.  Every input cell is **pre-filled with the current
    dummy value and shaded amber**, meaning *placeholder -- to be replaced*.
    Each sheet carries its ``config`` path, unit and the Thai source agency in
    its header block.

:func:`read_input_workbook`
    Read a filled-in workbook back into a :class:`~ilo_pens_oldage.config.Config`,
    ready for :func:`~ilo_pens_oldage.engine.run_valuation`.

:func:`workbook_status`
    Report which sheets are still marked ``TO REPLACE``, so you can see at a
    glance how far the data-collection exercise has got.

Two design decisions that make the workbook fillable by hand
-------------------------------------------------------------
**Anchor years, not 101 columns.**  Nobody types a hundred columns of mortality
rates.  Every year-varying surface is specified at seventeen anchor years --
dense in the short term (0, 1, 2, 3, 4, 5, 7, 10) and sparse in the long term
(15, 20, 25, 30, 40, 50, 60, 80, 100), editable on the ``_ANCHORS`` sheet.  The
reader fills in the years between: log-linearly for mortality, so a constant
percentage annual improvement is reproduced exactly, and linearly for
everything else.  Values are held flat after the last anchor.

This is how long-term assumptions are actually set -- the IMF publishes a
5-7 year economic forecast and the actuary drifts to an ultimate rate beyond it
-- and it is how the ILO report itself presents them: Tables A2.5 and A2.9 give
mortality for 2013, 2038 and 2063 only.  With the default anchors the
approximation costs about 0.003 percentage points on the General Average
Premium, which ``test_workbook.py`` measures on every run.

**Marginal or detailed base membership.**  The active-member array is
``2 x 3 x 55 x 51`` -- 16,830 numbers.  The SSO will normally give you the two
*marginals* instead: members by age and sex, and average past contribution
years by age and sex (the shape of Tables A2.6 and A2.13).  In
``active_base_mode = "marginal"`` (the default) you fill in those four columns
and the reader spreads them over the credit axis with a discretised gamma
matching the stated mean.  Set the mode to ``"detail"`` and regenerate the
template with ``detail=True`` if you obtain the full joint distribution.

Sheet naming
------------
Sheets beginning with an underscore (``_README``, ``_STATUS``, ``_ANCHORS``) are
control sheets.  Data sheets suffixed ``_M`` / ``_F`` hold the male and female
halves of a ``[sex, ...]`` array.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np

from .config import (
    AGE_MAX_INS,
    AGE_MAX_POP,
    AGE_MIN_INS,
    Config,
    MONTHS_PER_YEAR,
    N_AGE_INS,
    N_AGE_POP,
    N_CREDIT,
    N_INC,
    N_SEX,
)

# ---------------------------------------------------------------------------
# Layout constants -- the reader depends on these, do not move the data block
# ---------------------------------------------------------------------------

TITLE_ROW = 1
PATH_ROW = 2
META_ROW = 3
NOTE_ROW = 4
HEADER_ROW = 5
DATA_START_ROW = 6
INDEX_COL = 1  # column A holds the row label
DATA_START_COL = 2  # column B is the first data column

PLACEHOLDER_RGB = "FFF2CC"  # amber: still a placeholder
VERIFIED_RGB = "E2EFDA"     # green: replaced with real data
HEADER_RGB = "1F3864"
CONTROL_RGB = "D9E1F2"

STATUS_VALUES = ("TO REPLACE", "IN PROGRESS", "VERIFIED")

DEFAULT_ANCHOR_OFFSETS: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100
)
"""Year offsets at which year-varying inputs are specified.

Dense in the short term, sparse in the long term -- which is how assumptions
are actually set: the IMF publishes a 5-7 year economic forecast, and beyond
that the actuary sets a slow drift to an ultimate rate.  Seventeen columns
stays comfortably fillable by hand while keeping the interpolation error on
the General Average Premium below about 0.1 percentage points.

Editable on the ``_ANCHORS`` sheet; add offsets wherever your assumptions bend.
"""

POP_AGES = np.arange(N_AGE_POP)
INS_AGES = np.arange(AGE_MIN_INS, AGE_MAX_INS + 1)


# ---------------------------------------------------------------------------
# Sheet specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SheetSpec:
    """Declarative description of one input sheet.

    Attributes
    ----------
    name
        Worksheet name.
    title
        Human-readable heading written into row 1.
    path
        Dotted ``Config`` path the sheet populates, e.g.
        ``"membership.density"``.  Purely documentary for composite sheets.
    unit
        Unit of the values, written into the header block.
    source
        Thai source agency, written into the header block.
    index
        ``"age"`` (0-100), ``"insurable_age"`` (15-69), ``"credit"`` (0-50),
        ``"anchor_year"`` or ``"label"``.
    columns
        Column headings.  ``"ANCHORS"`` is expanded to the anchor-year labels.
    note
        Extra guidance written into row 4.
    interp
        How the reader fills the years between anchors: ``"linear"`` (default)
        or ``"log"``.  Mortality improves geometrically, so a constant annual
        rate of improvement is a straight line in log space and a curve in
        level space -- linear interpolation across a 20-year anchor gap then
        systematically overstates mortality in between.  Log interpolation
        makes the common "q improves x% a year" assumption exact.
    """

    name: str
    title: str
    path: str
    unit: str
    source: str
    index: str
    columns: Sequence[str]
    note: str = ""
    interp: Literal["linear", "log"] = "linear"


def _sex_pair(base: str, title: str, path: str, unit: str, source: str,
              index: str, note: str = "",
              interp: Literal["linear", "log"] = "linear") -> List[SheetSpec]:
    """Two sheets, ``_M`` and ``_F``, for a ``[sex, ...]`` array."""
    return [
        SheetSpec(f"{base}_M", f"{title} -- MALES", path, unit, source, index,
                  ["ANCHORS"], note, interp),
        SheetSpec(f"{base}_F", f"{title} -- FEMALES", path, unit, source, index,
                  ["ANCHORS"], note, interp),
    ]


MATRIX_SHEETS: List[SheetSpec] = [
    SheetSpec(
        "Population_Base", "National population at the valuation date",
        "demography.base_population", "persons", "NESDC / NSO", "age",
        ["male", "female"],
        "Single years of age. If you only have 5-year groups, disaggregate "
        "(Sprague / Beers) BEFORE entering them here.",
    ),
    *_sex_pair("Mortality", "General-population mortality q(x,t)",
               "demography.mortality_qx", "probability 0-1",
               "NESDC / UN WPP / OIC Thai Mortality Table", "age",
               "One column per anchor year. Interpolated LOG-linearly between "
               "anchors (so a constant % annual improvement is exact) and held "
               "flat after the last anchor.", "log"),
    *_sex_pair("InsMortality", "Insured-population mortality q(x,t)",
               "membership.insured_mortality_qx", "probability 0-1",
               "SSO experience / OIC TMO with selection", "age",
               "Insured mortality is materially lighter than general-population "
               "mortality. Do not simply copy the Mortality sheets. "
               "Interpolated LOG-linearly between anchors.", "log"),
    SheetSpec(
        "Fertility", "Age-specific fertility rates, base year",
        "demography.fertility_asfr_base", "births per woman per year",
        "NSO vital registration", "age", ["asfr"],
        "Non-zero only at ages 15-49. The profile is rescaled each year to hit "
        "the TFR on the Assumptions_Year sheet, so only its SHAPE matters here.",
    ),
    SheetSpec(
        "Migration_Profile", "Sex-age distribution of net migration",
        "demography.migration_age_profile", "proportion (all cells sum to 1)",
        "NSO / Department of Employment", "age", ["male", "female"],
        "The reader renormalises to sum to 1, so relative weights are enough.",
    ),
    SheetSpec(
        "Assumptions_Year", "Demographic and economic assumptions by year",
        "demography.tfr_path / labour.*", "see each column",
        "NESDC / NSO / BOT / MOC / IMF / SSO Investment Bureau", "anchor_year",
        ["tfr", "net_migration_persons", "inflation", "real_wage_growth",
         "real_gdp_growth", "real_investment_return"],
        "Rates are proportions per year (0.02 = 2%), NOT percentages.",
    ),
    *_sex_pair("LFPR", "Labour force participation rate", "labour.lfpr",
               "proportion 0-1", "NSO Labour Force Survey", "age"),
    *_sex_pair("Unemployment", "Unemployment rate", "labour.unemployment_rate",
               "proportion 0-1", "NSO Labour Force Survey", "age"),
    *_sex_pair("Coverage", "Coverage rate (insured / employed)",
               "membership.coverage_rate", "proportion 0-1",
               "SSO insured count / NSO LFS employed", "insurable_age"),
    *_sex_pair("Density", "Contribution density (contributors / active)",
               "membership.density", "proportion 0-1",
               "SSO contribution records", "insurable_age",
               "Drives BOTH the contribution base and the speed of credit "
               "accrual, so it is one of the highest-leverage inputs here."),
    *_sex_pair("Leaving", "Probability of becoming inactive",
               "membership.leaving_rate", "probability 0-1",
               "SSO member-register experience", "insurable_age"),
    *_sex_pair("Retirement", "Share of exits claiming a retirement benefit",
               "membership.retirement_rate", "proportion 0-1",
               "SSO award statistics", "insurable_age",
               "Should be 0 below the normal retirement age (the SSO does not "
               "permit early retirement) and near 1 at and above it."),
    *_sex_pair("Reentry", "Re-entry probability from the inactive pool",
               "membership.reentry_rate", "probability 0-1",
               "SSO member-register experience", "insurable_age"),
    SheetSpec(
        "NewEntrantShare", "New entrants as a share of the required inflow",
        "membership.new_entrant_share", "proportion 0-1",
        "SSO registration statistics", "insurable_age", ["ANCHORS"],
        "The remainder of the inflow is drawn from the inactive pool as "
        "re-entrants, subject to the Reentry availability cap.",
    ),
    SheetSpec(
        "SalaryScale", "Average insurable earnings by income band and age",
        "membership.salary_scale_monthly", "THB per month (base year)",
        "SSO declared wages", "insurable_age",
        ["male_low", "male_medium", "male_high",
         "female_low", "female_medium", "female_high"],
        "Bands are the mean of the bottom-30 / middle-40 / top-30 percentiles "
        "of the earnings distribution. Enter GROSS wages -- the statutory floor "
        "and ceiling are applied by the model, not here.",
    ),
    SheetSpec(
        "IncomeShares", "Share of members in each income band",
        "membership.income_level_shares", "proportion (sums to 1)",
        "SSO earnings distribution", "label", ["share"],
        "Structurally 0.30 / 0.40 / 0.30 under the percentile definition above.",
    ),
    SheetSpec(
        "ActiveBase_Marginal", "Active members at the valuation date (marginals)",
        "membership.active_base", "persons / years", "SSO member register",
        "insurable_age",
        ["male_members", "female_members", "male_mean_credit", "female_mean_credit"],
        "Used when RunControl!active_base_mode = 'marginal'. The reader spreads "
        "the member counts over the 0-50 credit axis using a discretised gamma "
        "matching the mean credit you enter.",
    ),
    SheetSpec(
        "InactiveBase_Marginal", "Inactive members at the valuation date (marginals)",
        "membership.inactive_base", "persons / years", "SSO member register",
        "insurable_age",
        ["male_members", "female_members", "male_mean_credit", "female_mean_credit"],
        "Registered members with no contribution in the base year. Their average "
        "past service is typically far below the active population's.",
    ),
    SheetSpec(
        "PensionersBase", "Old-age pensions in payment at the valuation date",
        "membership.old_age_pensioners_base / old_age_pension_base_monthly",
        "persons / THB per month", "SSO benefit register", "age",
        ["oldage_n_M", "oldage_n_F", "oldage_amt_M", "oldage_amt_F"],
        "The ``amt`` columns are the AVERAGE monthly pension per pensioner in "
        "that age cell, not the cell total. The SSO paid its first old-age "
        "pensions in 2014, so this stock is young and growing fast.",
    ),
]

DETAIL_SHEETS: List[SheetSpec] = [
    SheetSpec(
        "ActiveBase_Detail", "Active members by sex, income band, age and credit",
        "membership.active_base", "persons", "SSO member register", "credit_block",
        [str(c) for c in range(N_CREDIT)],
        "Used when RunControl!active_base_mode = 'detail'. 330 rows "
        "(2 sexes x 3 income bands x 55 ages) by 51 credit columns.",
    ),
    SheetSpec(
        "InactiveBase_Detail", "Inactive members by sex, income band, age and credit",
        "membership.inactive_base", "persons", "SSO member register", "credit_block",
        [str(c) for c in range(N_CREDIT)],
        "Used when RunControl!active_base_mode = 'detail'.",
    ),
]


# ---------------------------------------------------------------------------
# Scalar sheets
# ---------------------------------------------------------------------------

RUN_CONTROL_FIELDS = [
    ("base_year", "run.base_year", "calendar year",
     "Valuation date is 31 December of this year."),
    ("n_years", "run.n_years", "years",
     "Length of the projection; maximum 100 (ILO-PENS limit)."),
    ("title", "run.title", "text", "Printed on the Summary sheet."),
    ("scenario_name", "scenario_name", "text", "Label for this run."),
    ("active_base_mode", "-", "marginal | detail",
     "'marginal' reads the *_Marginal sheets; 'detail' reads the *_Detail sheets."),
]

SCHEME_FIELDS = [
    ("normal_retirement_age", "years", "SSO Act", "Statutory. 55 at present."),
    ("vesting_months", "months", "SSO Act", "Statutory. 180 months."),
    ("accrual_base_rate", "proportion", "SSO Act",
     "Statutory. 20% of the reference salary for the first 180 months."),
    ("accrual_increment_rate", "proportion", "SSO Act",
     "Statutory. +1.5% for each further 12 months of contribution."),
    ("final_salary_years", "years", "SSO Act",
     "Statutory. Reference salary = average of the final 60 months."),
    ("revalue_past_salaries", "TRUE / FALSE", "SSO Act",
     "FALSE reproduces the SSO's nominal average. TRUE tests the reform."),
    ("pension_indexation", "price|wage|fixed|none", "Assumption",
     "The SSO Act has NO cost-of-living clause; 'price' is the ILO assumption. "
     "Run 'none' as a scenario."),
    ("pension_indexation_fixed_rate", "proportion", "Assumption",
     "Used only when pension_indexation = 'fixed'."),
    ("min_insurable_earnings_monthly", "THB/month", "Ministerial regulation",
     "Statutory floor on the contribution base."),
    ("max_insurable_earnings_monthly", "THB/month", "Ministerial regulation",
     "Statutory ceiling. THB 15,000 at present."),
    ("ceiling_indexation", "price|wage|fixed|none", "Assumption",
     "'none' (frozen ceiling) is the realistic status quo and raises the cost rate."),
    ("floor_indexation", "price|wage|fixed|none", "Assumption", ""),
    ("contribution_rate_total", "proportion", "SSO Act / ILO branch allocation",
     "OLD-AGE share only. The statutory 7% funds old-age AND child allowance; "
     "ILO/TF/THAILAND/R.40 puts child allowance at 0.65%, leaving ~6.35%."),
    ("contribution_rate_employee", "proportion", "SSO Act",
     "Matters for the old-age grant refund rule below 12 months."),
    ("contribution_rate_employer", "proportion", "SSO Act", ""),
    ("contribution_rate_government", "proportion", "SSO Act", ""),
    ("collection_rate", "proportion", "SSO financial statements",
     "Contributions actually collected / contributions legally due."),
    ("grant_min_months_for_employer_share", "months", "SSO Act",
     "Below this, the old-age grant refunds only the employee's own share."),
    ("pensioner_death_guarantee_months", "months", "SSO Act", "60 months."),
    ("pensioner_death_lump_sum_multiple", "x monthly pension", "SSO Act", "10x."),
    ("min_pension_monthly", "THB/month or blank", "SSO Act",
     "Leave BLANK for no statutory minimum (the current position)."),
    ("max_pension_monthly", "THB/month or blank", "SSO Act", "Leave BLANK for none."),
    ("pay_grant_on_pre_retirement_death", "TRUE / FALSE", "SSO Act",
     "Refund of contributions when a member dies before vesting. An OLD-AGE "
     "lump sum funded from this branch, so TRUE keeps it in scope. FALSE "
     "understates branch expenditure."),
    ("admin_expense_rate", "proportion of insurable earnings",
     "SSO financial statements", "0.47% is the 2009-13 SSO all-branch average."),
]

FINANCE_FIELDS = [
    ("initial_reserve", "THB", "SSO audited financial statements",
     "Reserve of the branch being valued, at the valuation date."),
    ("gap_horizon_years", "years or blank", "Actuary's judgement",
     "Blank = the full projection period."),
    ("target_reserve_ratio", "multiple of annual expenditure", "Fund policy",
     "a0 = F(t-1) / S(t) at the end of the funding horizon."),
    ("target_balance_ratio", "-", "Fund policy",
     "b0 = 1 reproduces Thullen's scaled premium."),
]

LABOUR_SCALAR_FIELDS = [
    ("base_gdp", "THB", "NESDC", "Nominal GDP in the base year."),
]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _anchor_offsets(n_years: int) -> List[int]:
    """Anchor-year offsets that fit inside the projection horizon."""
    offsets = [o for o in DEFAULT_ANCHOR_OFFSETS if o <= n_years]
    if offsets[-1] != n_years:
        offsets.append(n_years)
    return offsets


def _index_values(
    kind: str,
    offsets: Optional[Sequence[int]] = None,
    base_year: int = 0,
) -> tuple[str, np.ndarray | List[str]]:
    """Row-label heading and values for an index kind.

    Parameters
    ----------
    kind
        One of ``"age"``, ``"insurable_age"``, ``"credit"``, ``"label"``,
        ``"anchor_year"`` or ``"credit_block"``.
    offsets, base_year
        Required only for ``"anchor_year"``, where the row labels are the
        calendar years of the anchors.
    """
    if kind == "age":
        return "age", POP_AGES
    if kind == "insurable_age":
        return "age", INS_AGES
    if kind == "credit":
        return "credit_years", np.arange(N_CREDIT)
    if kind == "label":
        return "band", ["low", "medium", "high"]
    if kind == "anchor_year":
        if offsets is None:
            raise ValueError("anchor_year index requires the anchor offsets.")
        return "year", [str(base_year + o) for o in offsets]
    if kind == "credit_block":
        labels = [
            f"{s}|{b}|{a}"
            for s in ("M", "F")
            for b in ("low", "medium", "high")
            for a in INS_AGES
        ]
        return "sex|band|age", labels
    raise ValueError(f"Unknown index kind {kind!r}")


def _collect_sheet_data(cfg: Config, spec: SheetSpec, offsets: List[int]) -> np.ndarray:
    """Extract the 2-D block of numbers a sheet should contain."""
    d, l, m = cfg.demography, cfg.labour, cfg.membership
    name = spec.name
    idx = np.asarray(offsets)

    def yr(arr: np.ndarray, sex: Optional[int] = None) -> np.ndarray:
        """Slice a ``[..., year]`` array at the anchor years."""
        a = np.asarray(arr, dtype=float)
        if sex is not None:
            a = a[sex]
        return a[:, idx]

    sex = 0 if name.endswith("_M") else (1 if name.endswith("_F") else None)

    if name == "Population_Base":
        return np.asarray(d.base_population, float).T
    if name.startswith("Mortality"):
        return yr(d.mortality_qx, sex)
    if name.startswith("InsMortality"):
        return yr(m.insured_mortality_qx, sex)
    if name == "Fertility":
        return np.asarray(d.fertility_asfr_base, float)[:, None]
    if name == "Migration_Profile":
        return np.asarray(d.migration_age_profile, float).T
    if name == "Assumptions_Year":
        return np.column_stack([
            np.asarray(d.tfr_path, float)[idx],
            np.asarray(d.net_migration_total, float)[idx],
            np.asarray(l.inflation, float)[idx],
            np.asarray(l.real_wage_growth, float)[idx],
            np.asarray(l.real_gdp_growth, float)[idx],
            np.asarray(l.real_investment_return, float)[idx],
        ])
    if name.startswith("LFPR"):
        return yr(l.lfpr, sex)
    if name.startswith("Unemployment"):
        return yr(l.unemployment_rate, sex)
    if name.startswith("Coverage"):
        return yr(m.coverage_rate, sex)
    if name.startswith("Density"):
        return yr(m.density, sex)
    if name.startswith("Leaving"):
        return yr(m.leaving_rate, sex)
    if name.startswith("Retirement"):
        return yr(m.retirement_rate, sex)
    if name.startswith("Reentry"):
        return yr(m.reentry_rate, sex)
    if name == "NewEntrantShare":
        return np.asarray(m.new_entrant_share, float)[:, idx]
    if name == "SalaryScale":
        s = np.asarray(m.salary_scale_monthly, float)  # (2, 3, 55)
        return np.column_stack([s[0, 0], s[0, 1], s[0, 2], s[1, 0], s[1, 1], s[1, 2]])
    if name == "IncomeShares":
        return np.asarray(m.income_level_shares, float)[:, None]
    if name in ("ActiveBase_Marginal", "InactiveBase_Marginal"):
        arr = np.asarray(
            m.active_base if name.startswith("Active") else m.inactive_base, float
        )
        counts = arr.sum(axis=(1, 3))  # (2, 55)
        credits = np.arange(N_CREDIT, dtype=float)
        weighted = (arr * credits[None, None, None, :]).sum(axis=(1, 3))
        mean = np.divide(weighted, counts, out=np.zeros_like(counts), where=counts > 0)
        return np.column_stack([counts[0], counts[1], mean[0], mean[1]])
    if name in ("ActiveBase_Detail", "InactiveBase_Detail"):
        arr = np.asarray(
            m.active_base if name.startswith("Active") else m.inactive_base, float
        )
        return arr.reshape(N_SEX * N_INC * N_AGE_INS, N_CREDIT)
    if name == "PensionersBase":
        n = np.asarray(m.old_age_pensioners_base, float)
        a = np.asarray(m.old_age_pension_base_monthly, float)
        return np.column_stack([n[0], n[1], a[0], a[1]])
    raise ValueError(f"No data extractor for sheet {name!r}")


def write_input_workbook(
    cfg: Config,
    path: str | Path,
    detail: bool = False,
) -> Path:
    """Write the fillable input workbook.

    Parameters
    ----------
    cfg
        A fully populated configuration -- normally the dummy one from
        :func:`~ilo_pens_oldage.dummy_data.build_dummy_config`.  Its values become
        the amber placeholders.
    path
        Destination ``.xlsx`` file.
    detail
        Also write the 330 x 51 ``*_Detail`` membership sheets.  Only needed if
        the SSO gives you the full joint distribution of age and credit.

    Returns
    -------
    Path
        The file written.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    offsets = _anchor_offsets(cfg.run.n_years)
    base_year = cfg.run.base_year

    wb = Workbook()
    wb.remove(wb.active)

    head_fill = PatternFill("solid", fgColor=HEADER_RGB)
    head_font = Font(color="FFFFFF", bold=True, size=10)
    ph_fill = PatternFill("solid", fgColor=PLACEHOLDER_RGB)
    ctl_fill = PatternFill("solid", fgColor=CONTROL_RGB)
    small = Font(size=9, italic=True, color="555555")
    body = Font(size=10)

    specs = list(MATRIX_SHEETS) + (list(DETAIL_SHEETS) if detail else [])

    # ---- _README -------------------------------------------------------
    ws = wb.create_sheet("_README")
    readme = [
        ("ILO-PENS Thailand -- INPUT WORKBOOK", ""),
        ("", ""),
        ("HOW TO USE", ""),
        ("1.", "Every amber cell holds a DUMMY placeholder value. Replace it with "
               "real data."),
        ("2.", "Each sheet's header block gives the config path, the unit, and the "
               "Thai source agency."),
        ("3.", "When a sheet is done, set its status on the _STATUS sheet to "
               "VERIFIED, and optionally clear the amber fill."),
        ("4.", "Run:  python run_from_excel.py --inputs <this file> --out results"),
        ("", ""),
        ("COLOUR KEY", ""),
        ("Amber", "Placeholder -- dummy value, must be replaced."),
        ("Blue", "Control sheet -- edit with care; the reader depends on it."),
        ("", ""),
        ("ANCHOR YEARS", ""),
        ("", "Year-varying inputs are entered at the anchor years listed on the "
             "_ANCHORS sheet, not for all 101 years."),
        ("", "The reader interpolates LINEARLY between anchors and holds the last "
             "anchor flat thereafter."),
        ("", "This matches how the ILO report itself presents assumptions "
             "(e.g. mortality for 2013, 2038 and 2063 only)."),
        ("", ""),
        ("BASE MEMBERSHIP", ""),
        ("", "RunControl!active_base_mode selects how the base membership is read."),
        ("marginal", "Fill ActiveBase_Marginal / InactiveBase_Marginal: members by "
                     "age and sex, plus average past contribution years by age and "
                     "sex. The reader spreads these over the 0-50 credit axis with "
                     "a discretised gamma matching your stated mean. This is the "
                     "usual case."),
        ("detail", "Fill ActiveBase_Detail / InactiveBase_Detail: the full "
                   "330 x 51 joint distribution. Regenerate the template with "
                   "--detail to get these sheets."),
        ("", ""),
        ("WHAT THE MODEL DOES NOT READ FROM HERE", ""),
        ("", "Nothing. Every exogenous input the engine uses is on one of these "
             "sheets -- there are no hidden constants in the code."),
        ("", ""),
        ("IMPORTANT", ""),
        ("", "Do not rename sheets, insert or delete rows above row 6, or move "
             "column A. The reader locates the data block by position."),
        ("", "Do not change the number of rows in a data block -- the age and "
             "credit axes are fixed by the model geometry."),
    ]
    for r, (a, b) in enumerate(readme, start=1):
        ws.cell(row=r, column=1, value=a).font = Font(bold=True, size=10)
        ws.cell(row=r, column=2, value=b).alignment = Alignment(wrap_text=True,
                                                                vertical="top")
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 104
    ws["A1"].font = Font(bold=True, size=14)

    # ---- _ANCHORS ------------------------------------------------------
    ws = wb.create_sheet("_ANCHORS")
    for c, h in enumerate(["anchor_index", "year_offset", "calendar_year"], start=1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.fill = head_fill
        cell.font = head_font
    for r, off in enumerate(offsets, start=2):
        ws.cell(row=r, column=1, value=r - 1)
        ws.cell(row=r, column=2, value=off).fill = ctl_fill
        ws.cell(row=r, column=3, value=base_year + off)
    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 16
    ws.cell(row=len(offsets) + 3, column=1,
            value="Edit year_offset to change where assumptions are anchored. "
                  "Offsets must be ascending and within 0..n_years.").font = small

    # ---- scalar sheets --------------------------------------------------
    def _scalar_sheet(sheet_name: str, title: str, fields, values: Dict[str, object],
                      with_source: bool = True) -> None:
        ws = wb.create_sheet(sheet_name)
        ws.cell(row=TITLE_ROW, column=1, value=title).font = Font(bold=True, size=12)
        headers = ["parameter", "value", "unit", "source", "note"]
        for c, h in enumerate(headers, start=1):
            cell = ws.cell(row=HEADER_ROW, column=c, value=h)
            cell.fill = head_fill
            cell.font = head_font
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
        for r, field in enumerate(fields, start=DATA_START_ROW):
            key = field[0]
            ws.cell(row=r, column=1, value=key).font = body
            v = values.get(key, None)
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, np.ndarray):
                v = None  # array-valued overrides are not editable here
            cell = ws.cell(row=r, column=2, value=v)
            cell.fill = ph_fill
            cell.font = body
            for c, text in enumerate(field[1:], start=3):
                ws.cell(row=r, column=c, value=text).font = small
        widths = [38, 20, 26, 30, 78]
        for c, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(c)].width = w

    _scalar_sheet("RunControl", "Run control", RUN_CONTROL_FIELDS, {
        "base_year": cfg.run.base_year,
        "n_years": cfg.run.n_years,
        "title": cfg.run.title,
        "scenario_name": cfg.scenario_name,
        "active_base_mode": "detail" if detail else "marginal",
    })
    _scalar_sheet("Scheme", "Statutory scheme parameters", SCHEME_FIELDS,
                  {f[0]: getattr(cfg.scheme, f[0], None) for f in SCHEME_FIELDS})
    _scalar_sheet("Finance", "Fund and financing parameters",
                  FINANCE_FIELDS + LABOUR_SCALAR_FIELDS,
                  {**{f[0]: getattr(cfg.finance, f[0], None) for f in FINANCE_FIELDS},
                   **{f[0]: getattr(cfg.labour, f[0], None)
                      for f in LABOUR_SCALAR_FIELDS}})

    # ---- matrix sheets --------------------------------------------------
    anchor_labels = [f"{base_year + o}" for o in offsets]

    for spec in specs:
        ws = wb.create_sheet(spec.name)
        ws.cell(row=TITLE_ROW, column=1, value=spec.title).font = Font(bold=True,
                                                                       size=12)
        ws.cell(row=PATH_ROW, column=1, value=f"config: {spec.path}").font = small
        ws.cell(row=META_ROW, column=1,
                value=f"unit: {spec.unit}    |    source: {spec.source}").font = small
        if spec.note:
            ws.cell(row=NOTE_ROW, column=1, value=spec.note).font = small

        index_name, index_vals = _index_values(spec.index, offsets, base_year)
        cols = anchor_labels if list(spec.columns) == ["ANCHORS"] else list(spec.columns)

        cell = ws.cell(row=HEADER_ROW, column=INDEX_COL, value=index_name)
        cell.fill = head_fill
        cell.font = head_font
        for c, label in enumerate(cols, start=DATA_START_COL):
            cell = ws.cell(row=HEADER_ROW, column=c, value=label)
            cell.fill = head_fill
            cell.font = head_font
            cell.alignment = Alignment(horizontal="center", wrap_text=True)

        data = np.atleast_2d(_collect_sheet_data(cfg, spec, offsets))
        if data.shape[1] != len(cols):
            raise RuntimeError(
                f"Sheet {spec.name}: {data.shape[1]} data columns but "
                f"{len(cols)} headings."
            )

        for r, label in enumerate(index_vals):
            ws.cell(row=DATA_START_ROW + r, column=INDEX_COL,
                    value=label if isinstance(label, str) else int(label)).font = body
            for c in range(data.shape[1]):
                cell = ws.cell(row=DATA_START_ROW + r, column=DATA_START_COL + c,
                               value=float(data[r, c]))
                cell.fill = ph_fill
                cell.font = body
                cell.number_format = _number_format(spec.unit)

        ws.freeze_panes = ws.cell(row=DATA_START_ROW, column=DATA_START_COL)
        ws.column_dimensions["A"].width = 18 if spec.index != "credit_block" else 22
        for c in range(DATA_START_COL, DATA_START_COL + len(cols)):
            ws.column_dimensions[get_column_letter(c)].width = 13
        ws.row_dimensions[NOTE_ROW].height = 28
        ws.row_dimensions[HEADER_ROW].height = 26

    # ---- _STATUS --------------------------------------------------------
    ws = wb.create_sheet("_STATUS")
    headers = ["sheet", "config_path", "status", "source_agency", "note"]
    for c, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.fill = head_fill
        cell.font = head_font
    rows = [("RunControl", "run.*", "SSO / actuary", ""),
            ("Scheme", "scheme.*", "SSO Act / regulations", ""),
            ("Finance", "finance.* / labour.base_gdp", "SSO / NESDC", "")]
    rows += [(s.name, s.path, s.source, s.note[:120]) for s in specs]
    for r, (name, cpath, source, note) in enumerate(rows, start=2):
        ws.cell(row=r, column=1, value=name).font = body
        ws.cell(row=r, column=2, value=cpath).font = small
        cell = ws.cell(row=r, column=3, value="TO REPLACE")
        cell.fill = ph_fill
        cell.font = body
        ws.cell(row=r, column=4, value=source).font = small
        ws.cell(row=r, column=5, value=note).font = small

    dv = DataValidation(type="list", formula1='"' + ",".join(STATUS_VALUES) + '"',
                        allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"C2:C{len(rows) + 1}")
    for c, w in enumerate([26, 34, 16, 34, 70], start=1):
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.freeze_panes = "A2"

    # Put the control sheets first.
    order = ["_README", "_STATUS", "_ANCHORS", "RunControl", "Scheme", "Finance"]
    wb._sheets.sort(key=lambda s: order.index(s.title) if s.title in order
                    else len(order) + specs.index(
                        next(x for x in specs if x.name == s.title)))

    wb.save(path)
    return path


def _number_format(unit: str) -> str:
    """Pick a sensible Excel number format from a unit string."""
    u = unit.lower()
    if "probability" in u or "proportion" in u:
        return "0.000000"
    if "thb" in u:
        return "#,##0"
    if "person" in u:
        return "#,##0"
    return "0.0000"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _read_block(ws, n_rows: int, n_cols: int, sheet_name: str) -> np.ndarray:
    """Read the fixed data block out of a matrix sheet.

    Raises
    ------
    ValueError
        If a cell is blank or non-numeric, naming the sheet and cell so you can
        go straight to it.
    """
    from openpyxl.utils import get_column_letter

    out = np.zeros((n_rows, n_cols), dtype=float)
    for r in range(n_rows):
        for c in range(n_cols):
            cell = ws.cell(row=DATA_START_ROW + r, column=DATA_START_COL + c)
            v = cell.value
            if v is None or v == "":
                raise ValueError(
                    f"{sheet_name}!{get_column_letter(DATA_START_COL + c)}"
                    f"{DATA_START_ROW + r} is blank -- every data cell must hold a "
                    "number."
                )
            try:
                out[r, c] = float(v)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{sheet_name}!{get_column_letter(DATA_START_COL + c)}"
                    f"{DATA_START_ROW + r} contains {v!r}, which is not a number."
                ) from None
    return out


def _read_scalars(ws, sheet_name: str) -> Dict[str, object]:
    """Read a key/value scalar sheet into a dict."""
    out: Dict[str, object] = {}
    r = DATA_START_ROW
    while True:
        key = ws.cell(row=r, column=1).value
        if key is None or str(key).strip() == "":
            break
        out[str(key).strip()] = ws.cell(row=r, column=2).value
        r += 1
    return out


def _coerce(value, kind: str, key: str, sheet: str):
    """Convert a spreadsheet value to the type the config field expects."""
    if kind == "optional_float":
        if value is None or value == "":
            return None
        return float(value)
    if kind == "optional_int":
        if value is None or value == "":
            return None
        return int(value)
    if value is None or value == "":
        raise ValueError(f"{sheet}: parameter {key!r} is blank but is required.")
    if kind == "float":
        return float(value)
    if kind == "int":
        return int(value)
    if kind == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().upper() in {"TRUE", "YES", "Y", "1"}
    if kind == "str":
        return str(value).strip()
    raise ValueError(f"Unknown coercion kind {kind!r}")


SCHEME_TYPES: Dict[str, str] = {
    "normal_retirement_age": "int",
    "vesting_months": "int",
    "accrual_base_rate": "float",
    "accrual_increment_rate": "float",
    "final_salary_years": "int",
    "revalue_past_salaries": "bool",
    "pension_indexation": "str",
    "pension_indexation_fixed_rate": "float",
    "min_insurable_earnings_monthly": "float",
    "max_insurable_earnings_monthly": "float",
    "ceiling_indexation": "str",
    "floor_indexation": "str",
    "contribution_rate_total": "float",
    "contribution_rate_employee": "float",
    "contribution_rate_employer": "float",
    "contribution_rate_government": "float",
    "collection_rate": "float",
    "grant_min_months_for_employer_share": "int",
    "pensioner_death_guarantee_months": "int",
    "pensioner_death_lump_sum_multiple": "float",
    "min_pension_monthly": "optional_float",
    "max_pension_monthly": "optional_float",
    "pay_grant_on_pre_retirement_death": "bool",
    "admin_expense_rate": "float",
}

FINANCE_TYPES: Dict[str, str] = {
    "initial_reserve": "float",
    "gap_horizon_years": "optional_int",
    "target_reserve_ratio": "float",
    "target_balance_ratio": "float",
}


def _interpolate_years(
    block: np.ndarray,
    offsets: Sequence[int],
    n_t: int,
    mode: str = "linear",
) -> np.ndarray:
    """Expand an anchor-year block to every projection year.

    Parameters
    ----------
    block
        Values at the anchor years, shape ``(n_rows, n_anchors)``.
    offsets
        Anchor-year offsets from the base year, ascending.
    n_t
        Number of projection points.
    mode
        ``"linear"`` or ``"log"``.  Log interpolation is applied to strictly
        positive rows only; a row containing a zero or negative value falls
        back to linear, since the logarithm is undefined there.

    Returns
    -------
    np.ndarray
        Shape ``(n_rows, n_t)``.  Interpolated between anchors and held flat
        after the last anchor.
    """
    x = np.asarray(offsets, dtype=float)
    xi = np.arange(n_t, dtype=float)
    out = np.empty((block.shape[0], n_t), dtype=float)
    for r in range(block.shape[0]):
        row = block[r]
        if mode == "log" and np.all(row > 0):
            out[r] = np.exp(np.interp(xi, x, np.log(row)))
        else:
            out[r] = np.interp(xi, x, row)
    return out


def read_input_workbook(path: str | Path) -> Config:
    """Read a filled-in input workbook into a :class:`Config`.

    Parameters
    ----------
    path
        Path to the workbook produced by :func:`write_input_workbook`.

    Returns
    -------
    Config
        Fully populated and ready for
        :func:`~ilo_pens_oldage.engine.run_valuation`.

    Raises
    ------
    ValueError
        On a blank or non-numeric data cell, a missing sheet, or an
        inconsistent anchor-year list -- always naming the sheet and cell.
    """
    from openpyxl import load_workbook

    from .dummy_data import build_credit_distribution

    path = Path(path)
    wb = load_workbook(path, data_only=True)

    def sheet(name: str):
        if name not in wb.sheetnames:
            raise ValueError(f"Workbook is missing the required sheet {name!r}.")
        return wb[name]

    # -- anchors ----------------------------------------------------------
    ws = sheet("_ANCHORS")
    offsets: List[int] = []
    r = 2
    while ws.cell(row=r, column=2).value is not None:
        offsets.append(int(ws.cell(row=r, column=2).value))
        r += 1
    if not offsets or offsets != sorted(offsets) or len(set(offsets)) != len(offsets):
        raise ValueError("_ANCHORS!B: year_offset must be strictly ascending.")

    # -- run control ------------------------------------------------------
    cfg = Config()
    run_vals = _read_scalars(sheet("RunControl"), "RunControl")
    cfg.run.base_year = _coerce(run_vals.get("base_year"), "int", "base_year",
                                "RunControl")
    cfg.run.n_years = _coerce(run_vals.get("n_years"), "int", "n_years", "RunControl")
    cfg.run.title = _coerce(run_vals.get("title"), "str", "title", "RunControl")
    cfg.scenario_name = _coerce(run_vals.get("scenario_name"), "str",
                                "scenario_name", "RunControl")
    mode = _coerce(run_vals.get("active_base_mode"), "str", "active_base_mode",
                   "RunControl").lower()
    if mode not in {"marginal", "detail"}:
        raise ValueError("RunControl!active_base_mode must be 'marginal' or 'detail'.")

    n_t = cfg.run.n_t
    if max(offsets) > cfg.run.n_years:
        raise ValueError(
            f"_ANCHORS: largest year_offset ({max(offsets)}) exceeds "
            f"RunControl!n_years ({cfg.run.n_years})."
        )
    n_anchor = len(offsets)

    # -- scheme -----------------------------------------------------------
    scheme_vals = _read_scalars(sheet("Scheme"), "Scheme")
    for key, kind in SCHEME_TYPES.items():
        if key in scheme_vals:
            setattr(cfg.scheme, key, _coerce(scheme_vals[key], kind, key, "Scheme"))

    # -- finance ----------------------------------------------------------
    fin_vals = _read_scalars(sheet("Finance"), "Finance")
    for key, kind in FINANCE_TYPES.items():
        if key in fin_vals:
            setattr(cfg.finance, key, _coerce(fin_vals[key], kind, key, "Finance"))
    if "base_gdp" in fin_vals:
        cfg.labour.base_gdp = _coerce(fin_vals["base_gdp"], "float", "base_gdp",
                                      "Finance")

    d, l, m = cfg.demography, cfg.labour, cfg.membership

    def matrix(name: str, n_rows: int, n_cols: int) -> np.ndarray:
        return _read_block(sheet(name), n_rows, n_cols, name)

    interp_modes = {spec.name: spec.interp
                    for spec in list(MATRIX_SHEETS) + list(DETAIL_SHEETS)}

    def surface(base: str, n_rows: int) -> np.ndarray:
        """Read a ``_M`` / ``_F`` anchor-year pair into a ``(2, n_rows, n_t)`` array."""
        out = np.empty((N_SEX, n_rows, n_t), dtype=float)
        for s, suffix in enumerate(("_M", "_F")):
            name = base + suffix
            block = matrix(name, n_rows, n_anchor)
            out[s] = _interpolate_years(block, offsets, n_t,
                                        interp_modes.get(name, "linear"))
        return out

    # -- demography -------------------------------------------------------
    d.base_population = matrix("Population_Base", N_AGE_POP, 2).T.copy()
    d.mortality_qx = surface("Mortality", N_AGE_POP)
    d.fertility_asfr_base = matrix("Fertility", N_AGE_POP, 1)[:, 0]
    d.migration_age_profile = matrix("Migration_Profile", N_AGE_POP, 2).T.copy()
    total = d.migration_age_profile.sum()
    if total <= 0:
        raise ValueError("Migration_Profile: all weights are zero.")
    d.migration_age_profile = d.migration_age_profile / total

    assumptions = matrix("Assumptions_Year", n_anchor, 6)
    year_series = _interpolate_years(assumptions.T, offsets, n_t)
    d.tfr_path = year_series[0]
    d.net_migration_total = year_series[1]
    l.inflation = year_series[2]
    l.real_wage_growth = year_series[3]
    l.real_gdp_growth = year_series[4]
    l.real_investment_return = year_series[5]

    # -- labour -----------------------------------------------------------
    l.lfpr = surface("LFPR", N_AGE_POP)
    l.unemployment_rate = surface("Unemployment", N_AGE_POP)

    # -- membership rate surfaces ------------------------------------------
    m.insured_mortality_qx = surface("InsMortality", N_AGE_POP)
    m.coverage_rate = surface("Coverage", N_AGE_INS)
    m.density = surface("Density", N_AGE_INS)
    m.leaving_rate = surface("Leaving", N_AGE_INS)
    m.retirement_rate = surface("Retirement", N_AGE_INS)
    m.reentry_rate = surface("Reentry", N_AGE_INS)
    m.new_entrant_share = _interpolate_years(
        matrix("NewEntrantShare", N_AGE_INS, n_anchor), offsets, n_t
    )

    # -- earnings ----------------------------------------------------------
    sal = matrix("SalaryScale", N_AGE_INS, 6)
    m.salary_scale_monthly = np.stack([sal[:, 0:3].T, sal[:, 3:6].T])  # (2, 3, 55)
    shares = matrix("IncomeShares", N_INC, 1)[:, 0]
    if shares.sum() <= 0:
        raise ValueError("IncomeShares: shares must sum to a positive number.")
    m.income_level_shares = shares / shares.sum()

    # -- base membership ---------------------------------------------------
    if mode == "detail":
        m.active_base = matrix("ActiveBase_Detail", N_SEX * N_INC * N_AGE_INS,
                               N_CREDIT).reshape(N_SEX, N_INC, N_AGE_INS, N_CREDIT)
        m.inactive_base = matrix("InactiveBase_Detail", N_SEX * N_INC * N_AGE_INS,
                                 N_CREDIT).reshape(N_SEX, N_INC, N_AGE_INS, N_CREDIT)
    else:
        def from_marginal(sheet_name: str) -> np.ndarray:
            block = matrix(sheet_name, N_AGE_INS, 4)
            out = np.zeros((N_SEX, N_INC, N_AGE_INS, N_CREDIT), dtype=float)
            for s in range(N_SEX):
                by_credit = build_credit_distribution(block[:, s], block[:, 2 + s])
                for j in range(N_INC):
                    out[s, j] = by_credit * m.income_level_shares[j]
            return out

        m.active_base = from_marginal("ActiveBase_Marginal")
        m.inactive_base = from_marginal("InactiveBase_Marginal")

    # -- old-age pensioners in payment -------------------------------------
    pens = matrix("PensionersBase", N_AGE_POP, 4)
    m.old_age_pensioners_base = pens[:, 0:2].T.copy()
    m.old_age_pension_base_monthly = pens[:, 2:4].T.copy()

    # -- derived ------------------------------------------------------------
    cfg.finance.nominal_interest_rate = (
        (1.0 + l.real_investment_return) * (1.0 + l.inflation) - 1.0
    )

    wb.close()
    return cfg


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------


def workbook_status(path: str | Path, check_fills: bool = False):
    """Report how far the data-collection exercise has got.

    Parameters
    ----------
    path
        Path to the input workbook.
    check_fills
        Also count the cells on each sheet that still carry the amber
        placeholder fill.  Accurate only if you clear the fill as you replace
        values; it is slower, so it is off by default.

    Returns
    -------
    pandas.DataFrame
        One row per sheet: ``sheet``, ``config_path``, ``status``,
        ``source_agency`` and, with ``check_fills``, ``placeholder_cells``.
    """
    import pandas as pd
    from openpyxl import load_workbook

    wb = load_workbook(Path(path), data_only=True)
    ws = wb["_STATUS"]
    rows = []
    r = 2
    while ws.cell(row=r, column=1).value is not None:
        rows.append({
            "sheet": ws.cell(row=r, column=1).value,
            "config_path": ws.cell(row=r, column=2).value,
            "status": ws.cell(row=r, column=3).value or "TO REPLACE",
            "source_agency": ws.cell(row=r, column=4).value,
        })
        r += 1
    df = pd.DataFrame(rows)

    if check_fills:
        counts = []
        for name in df["sheet"]:
            n = 0
            if name in wb.sheetnames:
                for row in wb[name].iter_rows(min_row=DATA_START_ROW,
                                              min_col=DATA_START_COL):
                    for cell in row:
                        fill = cell.fill
                        rgb = getattr(getattr(fill, "fgColor", None), "rgb", None)
                        if rgb and str(rgb).upper().endswith(PLACEHOLDER_RGB):
                            n += 1
            counts.append(n)
        df["placeholder_cells"] = counts

    wb.close()
    return df


__all__ = [
    "write_input_workbook",
    "read_input_workbook",
    "workbook_status",
    "SheetSpec",
    "MATRIX_SHEETS",
    "DETAIL_SHEETS",
    "DEFAULT_ANCHOR_OFFSETS",
    "STATUS_VALUES",
]
