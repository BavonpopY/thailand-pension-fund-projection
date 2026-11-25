"""
ilo_pens_oldage -- ILO Pension Model (ILO-PENS), Old-Age Benefit Branch.
=========================================================================

A modular implementation of the ILO's long-term social-security pension
projection methodology, parameterised for Thailand's Social Security Office
(SSO) Article 33/39 scheme and **restricted to the Old-Age Benefit Branch**.

Scope
-----
Two benefits are valued:

* the **Old-Age Monthly Pension** -- 20% of the final 60-month average wage at
  180 months of contributions, plus 1.5% for each further 12 months;
* the **Old-Age Lump-Sum Grant** -- a refund of contributions where the
  180-month vesting condition is not met.

Two associated old-age lump sums are also valued, because they are funded from
this branch: the refund on death before vesting, and the 10x monthly pension
paid when a pensioner dies inside the 60-month guarantee period.

Invalidity, survivors', orphans' and all short-term benefits are **out of
scope** -- those code paths do not exist, they are not merely disabled.

References
----------
*   International Labour Office (2018). *The ILO Pension Model: A Technical
    Guide*. Public Finance, Actuarial and Statistics Unit, Social Protection
    Department, Geneva. ISBN 978-92-2-031102-8.
*   International Labour Office (2016). *Thailand: Report to the Government --
    Actuarial Valuation of Thailand Social Security Scheme administered by the
    Social Security Office as of 31 December 2013*. ILO/TF/THAILAND/R.40.

Modules
-------
=====================  ====================================================
:mod:`.config`         All inputs and the ``[DATA INPUT REQUIRED]`` markers
:mod:`.dummy_data`     Placeholder generators (replace with real data)
:mod:`.demography`     Module 1 -- cohort-component population projection
:mod:`.labour`         Module 2 -- labour force and macroeconomic frame
:mod:`.earnings`       Salary surface, ``FINSAL()`` and ``BALANCE()``
:mod:`.actives`        Module 3 -- credit-distribution membership engine
:mod:`.benefits`       Module 4 -- vesting, pension formula, pensioner roll
:mod:`.finance`        Module 5 -- fund recursion and financing systems
:mod:`.scenarios`      Module 6 -- sensitivity and reform scenario engine
:mod:`.engine`         Orchestration (``ILOPENSION()`` / ``Projection()``)
:mod:`.outputs`        ILO-style result tables, CSV and Excel export
:mod:`.workbook_io`    Excel input template writer, reader, status report
:mod:`.validation`     Fourteen mathematical and actuarial consistency checks
=====================  ====================================================

Quick start
-----------
>>> from ilo_pens_oldage import Config, build_dummy_config, run_valuation
>>> cfg = build_dummy_config(Config())
>>> result = run_valuation(cfg)
>>> round(result.finance.general_average_premium, 4)  # doctest: +SKIP
"""

from .actives import ActiveFlows, ActiveState, step_active_population
from .benefits import BenefitAwards, PensionerState, old_age_pension_amount
from .config import (
    Config,
    DemographyConfig,
    FinanceConfig,
    LabourConfig,
    MembershipConfig,
    RunConfig,
    SchemeConfig,
)
from .demography import DemographyResult, project_population
from .dummy_data import build_credit_distribution, build_dummy_config
from .earnings import EarningsSurface, build_earnings_surface
from .engine import ValuationResult, run_valuation
from .finance import FinancialProjection, project_fund
from .labour import LabourResult, project_labour
from .outputs import missing_inputs, write_csv_bundle, write_excel_report
from .scenarios import Scenario, ScenarioSet, compare, default_scenarios
from .validation import format_report, run_all_checks

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # configuration
    "Config", "RunConfig", "DemographyConfig", "LabourConfig", "SchemeConfig",
    "MembershipConfig", "FinanceConfig",
    "build_dummy_config", "build_credit_distribution",
    # modules
    "project_population", "DemographyResult",
    "project_labour", "LabourResult",
    "build_earnings_surface", "EarningsSurface",
    "ActiveState", "ActiveFlows", "step_active_population",
    "PensionerState", "BenefitAwards", "old_age_pension_amount",
    "project_fund", "FinancialProjection",
    "Scenario", "ScenarioSet", "compare", "default_scenarios",
    # orchestration and output
    "run_valuation", "ValuationResult",
    "write_excel_report", "write_csv_bundle", "missing_inputs",
    "run_all_checks", "format_report",
]
