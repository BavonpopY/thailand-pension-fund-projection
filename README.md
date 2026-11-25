# Thailand Old-Age Pension Fund: Sustainability Projection

Senior project in Insurance (Actuarial Science), Chulalongkorn University, 2026. Team of 2.

**Question:** how long can the Old-Age Benefit Branch of Thailand's Social Security Fund (Articles 33 and 39) keep paying under current rules, and which reforms would keep it solvent?

**What we built:** a Python implementation of the ILO pension projection method (ILO-PENS), limited to the old-age branch. An Excel input workbook lets users change assumptions without editing code. The base year is 2024 and the projection runs 100 years (2025–2124). The project analysed 37 sensitivity and reform scenarios.

**Key results (baseline, national data):**

- Under current rules, the old-age fund is projected to run out in **2062**.
- The 100-year general average premium (GAP) is **11.81%** of insurable wages, about **1.86×** the 6.35% currently allocated to the old-age branch.
- Combined reforms keep the fund solvent over the full 100-year horizon. These pair a retirement age of 65 with higher contribution rates.

**Skills shown:** actuarial projection (demography, membership, benefits, financing), Python (NumPy, pandas, openpyxl), Excel-driven model inputs, automated model checks, scenario analysis.

> **Data:** `ILO_PENS_OLDAGE_INPUTS.xlsx` holds the national inputs, compiled from public sources (NESDC, NSO, SSO, OIC, ILO, UN). `results/` holds the baseline valuation (CSV and Excel). Without the workbook, the model also runs on its built-in placeholder data.

---

## Model documentation: ilo_pens_oldage — ILO Pension Model, Old-Age Benefit Branch

A modular implementation of the ILO's long-term pension projection methodology,
parameterised for Thailand's Social Security Office (SSO) Article 33/39 scheme and
**restricted to the Old-Age Benefit Branch**. Runs out of the box on annotated dummy
data; every exogenous input is flagged so it can be swapped for a real national
dataset without touching the model logic.

**Thai-language documentation** — the Excel workflow, the data-insertion tables and
the workflow explanation — is in
[`DATA_INSERTION_GUIDE_TH.md`](DATA_INSERTION_GUIDE_TH.md).

## Scope

**In scope — two benefits:**

| Benefit | Rule |
|---|---|
| Old-Age Monthly Pension (บำนาญชราภาพ) | 20% of the final 60-month average wage at 180 months of contributions, **+1.5% per further 12 months**. Payable from the NRA (55). |
| Old-Age Lump-Sum Grant (บำเหน็จชราภาพ) | Refund of contributions where the 180-month condition is not met. Under 12 months: employee share only. 12–179 months: employee + employer, with interest. |

**Also valued**, because both are old-age lump sums funded from this branch:

* refund of accumulated contributions on **death before vesting**
  (`scheme.pay_grant_on_pre_retirement_death`, default on);
* **10× the monthly pension** when a pensioner dies inside the 60-month guarantee.

**Out of scope:** invalidity, survivors', orphans', and all short-term benefits
(sickness, maternity, child allowance, funeral, unemployment). These are not toggled
off — the code paths do not exist.

**Contribution rate defaults to 6.35%, not 7%.** The statutory 7% (employer 3% +
employee 3% + government 1%) funds the "two benefits type" branch, which is old-age
*and* child allowance. ILO/TF/THAILAND/R.40 assesses the child-allowance cost at
0.65%, leaving ~6.35% for old age. Entering the full 7% would overstate old-age
income by about 10%.

### One consequence of dropping invalidity

Members who would have left the active population as invalids now remain exposed to
the old-age decrements, so a share of them eventually vest and draw an old-age
pension. This is the correct treatment for a stand-alone old-age valuation — it is
how the liability behaves once invalidity is financed separately — but it means
old-age costs here run marginally above the old-age slice of a combined
multi-branch run.

## Install

```bash
pip install -r requirements.txt
```

## Excel-driven workflow (no Python editing)

```bash
python run_from_excel.py --create ILO_PENS_OLDAGE_INPUTS.xlsx
#   ... fill it in ...
python run_from_excel.py --status ILO_PENS_OLDAGE_INPUTS.xlsx
python run_from_excel.py --inputs ILO_PENS_OLDAGE_INPUTS.xlsx --out results
```

`--create` writes a 31-sheet workbook in which **every input cell already holds the
dummy value, shaded amber** — meaning *placeholder, replace me*. Each sheet's header
block carries its `config` path, unit and Thai source agency. Mark each sheet
`VERIFIED` on `_STATUS` as you finish it.

The model runs at every stage of completion. Nothing is ever silently defaulted: a
blank or non-numeric cell stops the run and names the sheet and cell
(`Density_M!D12 is blank`).

Two things make the workbook fillable by hand:

* **Anchor years, not 101 columns.** Year-varying inputs are entered at 17 anchor
  years — dense early (0,1,2,3,4,5,7,10), sparse later (15…100) — and the reader
  fills the gaps: log-linearly for mortality, so a constant % annual improvement is
  exact, linearly otherwise. Costs 0.004 pp on the GAP, measured on every test run.
* **Marginal base membership.** The active array is 16,830 numbers. In the default
  `marginal` mode you enter members by age × sex and *average past contribution
  years* by age × sex — the shape of ILO Tables A2.6 and A2.13 — and the reader
  spreads them over the credit axis with a gamma whose scale is solved by bisection
  so your stated mean is reproduced exactly. Use `--create --detail` for the full
  330 × 51 joint distribution.

Verify the plumbing with `python test_workbook.py`.

## Python workflow

```bash
python run_model.py                     # baseline, checks, CSV + Excel
python run_model.py --scenarios         # + 18 sensitivity / reform scenarios
python run_model.py --years 60 --contribution-rate 0.09
```

A 100-year projection with the full scenario battery takes about five seconds.

```python
from ilo_pens_oldage import Config, build_dummy_config, run_valuation

cfg = build_dummy_config(Config())
cfg.scheme.vesting_months = 120        # any statutory parameter
result = run_valuation(cfg)
print(result.summary())
```

## Package layout

| File | Role |
|---|---|
| `config.py` | Every input, with `# [DATA INPUT REQUIRED: … \| shape \| unit]` markers |
| `dummy_data.py` | Placeholder generators — **replace with real data readers** |
| `demography.py` | **Module 1** — cohort-component national population projection |
| `labour.py` | **Module 2** — labour force, employment, macroeconomic indices |
| `earnings.py` | Salary surface; `FINSAL()` reference salary; `BALANCE()` accumulations |
| `actives.py` | **Module 3** — `[sex × income × age × credit]` membership engine |
| `benefits.py` | **Module 4** — vesting, pension formula, grant, pensioner roll-forward |
| `finance.py` | **Module 5** — fund recursion, PAYG rate, GAP, target-ratio rates |
| `scenarios.py` | **Module 6** — declarative sensitivity and reform engine |
| `engine.py` | Orchestration (`ILOPENSION()` / `Projection()`) |
| `outputs.py` | ILO-style tables, CSV bundle, formatted Excel workbook |
| `workbook_io.py` | Excel input template writer, reader and status report |
| `validation.py` | Fourteen mathematical and actuarial consistency checks |

Entry points: `run_from_excel.py`, `run_model.py`, `test_workbook.py`.

## Array conventions

| Symbol | Meaning | Size |
|---|---|---|
| `s` | sex — 0 male, 1 female | 2 |
| `j` | income band — low / medium / high (ILO-PENS `JKC`) | 3 |
| `a` | national age, 0–100, top group open-ended | 101 |
| `x` | insurable age, 15–69 | 55 |
| `c` | accumulated credit in years, 0–50 | 51 |
| `t` | projection year, 0 = base year | ≤ 101 |

The insured population is a dense `(2, 3, 55, 51)` array. Carrying the **full credit
distribution** rather than an average past-service per cell is what lets the
180-month vesting condition and the tiered accrual rate be applied exactly. For a
scheme as young as the SSO's, the share of members crossing the vesting threshold is
the single biggest cost driver — the `Vesting cut to 120 months` scenario moves the
GAP by +8.6 pp, more than any other lever in the battery.

## Actuarial verification

`python run_model.py` runs fourteen checks and exits non-zero if any fails. The
report also goes to the `Validation` sheet.

| Check | Identity |
|---|---|
| `population_balance` | `P(t+1) = P(t) − D + B + M` |
| `fund_recursion` | `F(t) − F(t−1) = I(t) − S(t)` |
| `income_identity` | `I(t) = P(t) + R(t)` |
| `payg_identity` | `C_PAYG(t) · G(t) = S(t)` |
| `payg_decomposition` | `C_PAYG = d · r` on the pension component |
| `expenditure_components` | pension + grants + guarantee + admin = `S(t)` |
| `old_age_scope` | only old-age streams present; no out-of-scope leakage |
| `gap_balances_the_fund` | the GAP zeroes the discounted equation of value |
| `credit_axis_conservation` | accrual preserves head count **and** mean credit |
| `non_negative_stocks` | no negative members, pensioners or amounts |
| `coverage_target_tracking` | actives track `Covrate × Employed` |
| `vesting_threshold` | **no pension payable below 180 months** |
| `monotone_eligibility` | pension non-decreasing in credit |
| `replacement_rate_bounds` | 20% at 15 years, 72.5% at 50 years |

## Reading the results

* **After the depletion year the reserve goes negative** and investment income turns
  negative with it. The recursion is deliberately continued rather than truncated,
  because the level-premium and target-ratio formulas require a linear reserve path.
  Past depletion the meaningful statement is that the required contribution rate
  becomes the PAYG cost rate, reported year by year.
* **The GAP is independent of the legal contribution rate** — it is the rate the
  scheme *requires*. A contribution-rate scenario therefore moves the depletion year
  but not the GAP.

## Replacing the dummy data

1. Open `DATA_INSERTION_GUIDE_TH.md` or the `Data_Inputs_Required` sheet for the full
   parameter × shape × unit × source-agency table.
2. Fill the Excel workbook, or assign arrays directly in Python
   (`cfg.membership.active_base = read_sso_register(...)`).
3. `build_dummy_config(cfg)` fills only what is still `None`, so real and placeholder
   data coexist while collection is under way; `missing_inputs(cfg)` reports the rest.
