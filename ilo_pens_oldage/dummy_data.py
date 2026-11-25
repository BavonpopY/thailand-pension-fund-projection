"""
Dummy-data generators -- OLD-AGE BRANCH -- REPLACE WITH REAL NATIONAL DATASETS.
==============================================================================

Every function here manufactures an input array with the exact shape, dtype and
unit that :mod:`ilo_pens_oldage.config` documents, so that the whole pipeline runs
out of the box.  The numbers are *plausible* -- loosely shaped on the published
Thai figures in ILO/TF/THAILAND/R.40 -- but they are **not** a valuation.

Replacement strategy
--------------------
For each ``make_*`` function below there is a matching
``# [DATA INPUT REQUIRED: ...]`` annotation in :mod:`ilo_pens_oldage.config`.  To go
live, write a reader that returns an array of the same shape and assign it to
the corresponding config field; you do not need to touch any other module.

A convenience wrapper :func:`build_dummy_config` populates a whole
:class:`~ilo_pens_oldage.config.Config` in one call.
"""

from __future__ import annotations

import numpy as np

from .config import (
    AGE_MAX_INS,
    AGE_MAX_POP,
    AGE_MIN_INS,
    Config,
    N_AGE_INS,
    N_AGE_POP,
    N_CREDIT,
    N_INC,
    N_SEX,
    CREDIT_MAX,
    MONTHS_PER_YEAR,
)

# ---------------------------------------------------------------------------
# Small shape helpers
# ---------------------------------------------------------------------------

POP_AGES = np.arange(N_AGE_POP)  # 0..100
INS_AGES = np.arange(AGE_MIN_INS, AGE_MAX_INS + 1)  # 15..69


def _expand_5yr(values_by_group: dict[tuple[int, int], float], fill: float = 0.0) -> np.ndarray:
    """Expand a {(lo, hi): value} 5-year-group mapping to single ages 0..100."""
    out = np.full(N_AGE_POP, fill, dtype=float)
    for (lo, hi), value in values_by_group.items():
        out[lo : hi + 1] = value
    return out


def _smooth(arr: np.ndarray, passes: int = 2) -> np.ndarray:
    """Cheap 3-point moving-average smoother along the last axis."""
    out = np.asarray(arr, dtype=float).copy()
    for _ in range(passes):
        padded = np.concatenate([out[..., :1], out, out[..., -1:]], axis=-1)
        out = (padded[..., :-2] + padded[..., 1:-1] + padded[..., 2:]) / 3.0
    return out


def _ramp(n_t: int, start: float, ultimate: float, transition_years: int = 15) -> np.ndarray:
    """Linear ramp from ``start`` to ``ultimate`` then flat, shape ``(n_t,)``."""
    path = np.full(n_t, float(ultimate))
    k = min(transition_years, n_t)
    path[:k] = np.linspace(start, ultimate, k)
    return path


# ---------------------------------------------------------------------------
# 1. Demography
# ---------------------------------------------------------------------------


def make_base_population() -> np.ndarray:
    """National population by sex and single year of age.

    Shape ``(2, 101)``, unit persons.  Shaped on the Thai age structure
    (ILO/TF/THAILAND/R.40 Table A2.2): a post-transition pyramid with a bulge
    around ages 35-55 and a narrow base.
    """
    # Population per single year of age within each 5-year band (thousands).
    male_5yr = {
        (0, 4): 1_928_390, (5, 9): 2_056_107, (10, 14): 2_264_717,
        (15, 19): 2_411_064, (20, 24): 2_383_406, (25, 29): 2_341_773,
        (30, 34): 2_537_870, (35, 39): 2_712_679, (40, 44): 2_765_492,
        (45, 49): 2_724_272, (50, 54): 2_388_850, (55, 59): 1_979_908,
        (60, 64): 1_505_707, (65, 69): 1_047_368, (70, 74): 758_487,
        (75, 79): 529_031, (80, 84): 326_094, (85, 89): 134_985,
        (90, 100): 39_360,
    }
    female_5yr = {
        (0, 4): 1_841_714, (5, 9): 1_968_808, (10, 14): 2_188_093,
        (15, 19): 2_354_285, (20, 24): 2_352_888, (25, 29): 2_338_139,
        (30, 34): 2_581_860, (35, 39): 2_805_306, (40, 44): 2_898_635,
        (45, 49): 2_895_394, (50, 54): 2_568_923, (55, 59): 2_164_711,
        (60, 64): 1_684_580, (65, 69): 1_218_295, (70, 74): 938_394,
        (75, 79): 714_922, (80, 84): 483_023, (85, 89): 217_301,
        (90, 100): 71_409,
    }

    pop = np.zeros((N_SEX, N_AGE_POP), dtype=float)
    for s, groups in enumerate((male_5yr, female_5yr)):
        for (lo, hi), total in groups.items():
            width = hi - lo + 1
            pop[s, lo : hi + 1] = total / width
    return _smooth(pop, passes=2)


def make_mortality_qx(n_t: int, improvement: float = 0.010) -> np.ndarray:
    """General-population mortality surface q(x, t).

    Shape ``(2, 101, n_t)``, unit probability.  A Gompertz-Makeham base with an
    infant hump, calibrated so that base-year life expectancy at birth is
    roughly 71 (male) / 78 (female), then improved geometrically.
    """
    a = np.array([0.00035, 0.00018])          # Makeham constant, by sex
    b = np.array([0.000042, 0.000020])        # Gompertz level
    c = np.array([0.0925, 0.0955])            # Gompertz slope
    infant = np.array([0.0105, 0.0087])       # q0

    ages = POP_AGES[None, :]
    qx0 = a[:, None] + b[:, None] * np.exp(c[:, None] * ages)
    qx0[:, 0] = infant
    qx0[:, 1:5] = infant[:, None] * np.array([0.12, 0.07, 0.05, 0.04])[None, :]
    qx0 = np.clip(qx0, 1e-6, 0.999)
    qx0[:, AGE_MAX_POP] = 1.0  # absorbing terminal age

    improvement_factor = (1.0 - improvement) ** np.arange(n_t)
    qx = qx0[:, :, None] * improvement_factor[None, None, :]
    qx[:, AGE_MAX_POP, :] = 1.0
    return np.clip(qx, 1e-8, 1.0)


def make_insured_mortality_qx(general_qx: np.ndarray, selection: float = 0.85) -> np.ndarray:
    """Insured-population mortality: a flat selection discount on general q(x,t).

    Shape ``(2, 101, n_t)``.  The 2013 Thai valuation used an insured life
    expectancy at 20 of 56.6 (male) vs 53.0 for the general population, i.e.
    materially lighter mortality.
    """
    qx = np.asarray(general_qx, dtype=float) * float(selection)
    qx[:, AGE_MAX_POP, :] = 1.0
    return np.clip(qx, 1e-8, 1.0)


def make_fertility_asfr() -> np.ndarray:
    """Age-specific fertility rates in the base year, shape ``(101,)``.

    Shaped on ILO/TF/THAILAND/R.40 Table A2.3 (TFR 1.60 in 2013), spread to
    single ages within each 5-year band.
    """
    asfr_5yr = {
        (15, 19): 0.04660, (20, 24): 0.08409, (25, 29): 0.08588,
        (30, 34): 0.06526, (35, 39): 0.03034, (40, 44): 0.00745,
        (45, 49): 0.00038,
    }
    return _smooth(_expand_5yr(asfr_5yr), passes=1)


def make_tfr_path(n_t: int, start: float = 1.30, ultimate: float = 1.20) -> np.ndarray:
    """Total fertility rate by year, shape ``(n_t,)``.

    NOTE the defaults are BELOW the 1.60 -> 1.50 used in the 2013 ILO report,
    reflecting the sharp subsequent fall in Thai fertility.  This is exactly
    the kind of assumption to re-derive from current NESDC projections.
    """
    return _ramp(n_t, start, ultimate, transition_years=20)


def make_migration_age_profile() -> np.ndarray:
    """Share of annual net migration by sex and age, shape ``(2, 101)``, sums to 1."""
    profile = np.zeros((N_SEX, N_AGE_POP), dtype=float)
    working = np.exp(-0.5 * ((POP_AGES - 27.0) / 8.0) ** 2)
    working[POP_AGES < 15] = 0.0
    working[POP_AGES > 60] = 0.0
    profile[0] = working * 0.55
    profile[1] = working * 0.45
    total = profile.sum()
    return profile / total if total > 0 else profile


def make_net_migration(n_t: int, annual: float = 40_000.0) -> np.ndarray:
    """Net migration by year, shape ``(n_t,)``, unit persons.

    The 2013 ILO Thailand valuation held this constant at +40,000 p.a.
    """
    return np.full(n_t, float(annual))


# ---------------------------------------------------------------------------
# 2. Labour force and macroeconomy
# ---------------------------------------------------------------------------


def make_lfpr(n_t: int) -> np.ndarray:
    """Labour force participation rate by sex, age, year, shape ``(2, 101, n_t)``.

    Shaped on ILO/TF/THAILAND/R.40 Figure 2.6: male peak ~96% at 30-50,
    female peak ~85%, both declining sharply after 55.  Held constant over the
    projection (the aggregate rate still falls, through ageing alone).
    """
    ages = POP_AGES.astype(float)

    def profile(peak: float, rise_mid: float, fall_mid: float, fall_slope: float) -> np.ndarray:
        rise = 1.0 / (1.0 + np.exp(-(ages - rise_mid) / 2.0))
        fall = 1.0 / (1.0 + np.exp((ages - fall_mid) / fall_slope))
        p = peak * rise * fall
        p[ages < AGE_MIN_INS] = 0.0
        p[ages > 79] = 0.0
        return p

    lfpr = np.zeros((N_SEX, N_AGE_POP), dtype=float)
    lfpr[0] = profile(peak=0.965, rise_mid=19.5, fall_mid=62.0, fall_slope=6.0)
    lfpr[1] = profile(peak=0.850, rise_mid=20.5, fall_mid=58.0, fall_slope=6.5)
    return np.repeat(lfpr[:, :, None], n_t, axis=2)


def make_unemployment_rate(n_t: int) -> np.ndarray:
    """Unemployment rate by sex, age, year, shape ``(2, 101, n_t)``.

    Thailand's measured unemployment is extremely low (0.7% in 2013) and
    strongly concentrated in the youngest ages.
    """
    ages = POP_AGES.astype(float)
    base = 0.004 + 0.045 * np.exp(-0.5 * ((ages - 20.0) / 4.5) ** 2)
    base[ages < AGE_MIN_INS] = 0.0
    u = np.zeros((N_SEX, N_AGE_POP), dtype=float)
    u[0] = base
    u[1] = base * 1.10
    return np.repeat(u[:, :, None], n_t, axis=2)


def make_inflation(n_t: int, start: float = 0.012, ultimate: float = 0.020) -> np.ndarray:
    """CPI inflation by year, shape ``(n_t,)``.  Ultimate = BOT target midpoint."""
    return _ramp(n_t, start, ultimate, transition_years=10)


def make_real_wage_growth(n_t: int, start: float = 0.015, ultimate: float = 0.020) -> np.ndarray:
    """Real growth of average insurable earnings by year, shape ``(n_t,)``."""
    return _ramp(n_t, start, ultimate, transition_years=12)


def make_real_gdp_growth(n_t: int, start: float = 0.028, ultimate: float = 0.018) -> np.ndarray:
    """Real GDP growth by year, shape ``(n_t,)``."""
    return _ramp(n_t, start, ultimate, transition_years=25)


def make_real_investment_return(n_t: int, ultimate: float = 0.020) -> np.ndarray:
    """Real return on scheme assets by year, shape ``(n_t,)``.

    The 2013 ILO Thailand valuation used a constant 2.0% real return.
    """
    return np.full(n_t, float(ultimate))


# ---------------------------------------------------------------------------
# 3. Scheme membership
# ---------------------------------------------------------------------------


def make_coverage_rate(n_t: int, start_peak: float = 0.58, ultimate_peak: float = 0.72) -> np.ndarray:
    """Insured / employed coverage rate by sex, insurable age, year.

    Shape ``(2, 55, n_t)``.  Shaped on ILO/TF/THAILAND/R.40 Figure A2.1:
    a sharp rise to a peak in the early 20s, then a long decline as informal
    and self-employed work takes over at older ages.  Coverage deepens over the
    first 40 years of projection, then is held flat.
    """
    ages = INS_AGES.astype(float)
    shape = np.exp(-0.5 * ((ages - 24.0) / 4.0) ** 2) * 0.55 + np.exp(
        -0.5 * ((ages - 33.0) / 13.0) ** 2
    ) * 0.62
    shape = shape / shape.max()
    shape[ages > 62] *= np.linspace(1.0, 0.12, int((ages > 62).sum()))

    deepening = np.ones(n_t)
    k = min(40, n_t)
    deepening[:k] = np.linspace(1.0, ultimate_peak / start_peak, k)
    deepening[k:] = ultimate_peak / start_peak

    cov = np.zeros((N_SEX, N_AGE_INS, n_t), dtype=float)
    cov[0] = start_peak * shape[:, None] * deepening[None, :]
    cov[1] = start_peak * 1.06 * shape[:, None] * deepening[None, :]
    return np.clip(cov, 0.0, 0.98)


def make_density(n_t: int) -> np.ndarray:
    """Contribution density by sex, insurable age, year, shape ``(2, 55, n_t)``.

    ILO/TF/THAILAND/R.40 Table A2.12: ~50% at 15-19 rising to ~93-95% at 50+.
    """
    dens_5yr = {
        (15, 19): (0.508, 0.493), (20, 24): (0.691, 0.726),
        (25, 29): (0.845, 0.876), (30, 34): (0.891, 0.915),
        (35, 39): (0.908, 0.927), (40, 44): (0.918, 0.935),
        (45, 49): (0.925, 0.940), (50, 54): (0.930, 0.946),
        (55, 59): (0.914, 0.928), (60, 64): (0.923, 0.940),
        (65, 69): (0.950, 0.963),
    }
    dens = np.zeros((N_SEX, N_AGE_INS), dtype=float)
    for (lo, hi), (m, f) in dens_5yr.items():
        idx = slice(lo - AGE_MIN_INS, hi - AGE_MIN_INS + 1)
        dens[0, idx] = m
        dens[1, idx] = f
    dens = _smooth(dens, passes=1)
    return np.repeat(dens[:, :, None], n_t, axis=2)


def make_leaving_rate(n_t: int, nra: int = 55) -> np.ndarray:
    """Probability of becoming inactive during the year.

    Shape ``(2, 55, n_t)``.  High churn at young ages, a trough in mid-career,
    then a sharp spike at the normal retirement age.
    """
    ages = INS_AGES.astype(float)
    base = 0.055 + 0.16 * np.exp(-0.5 * ((ages - 19.0) / 4.0) ** 2)
    base += 0.020 * np.exp(-0.5 * ((ages - 50.0) / 9.0) ** 2)
    # Retirement spike: most exits at/after the NRA are retirements.
    spike = np.where(ages >= nra, 0.16 + 0.035 * (ages - nra), 0.0)
    spike = np.clip(spike, 0.0, 0.80)
    rate = np.clip(base + spike, 0.0, 0.95)
    rate[ages >= AGE_MAX_INS] = 1.0  # forced exit at the top coverage age

    out = np.zeros((N_SEX, N_AGE_INS), dtype=float)
    out[0] = rate
    out[1] = rate * 1.08
    return np.repeat(np.clip(out, 0.0, 1.0)[:, :, None], n_t, axis=2)


def make_retirement_rate(n_t: int, nra: int = 55) -> np.ndarray:
    """Share of exits from the active population that claim a retirement benefit.

    Shape ``(2, 55, n_t)``.  Zero below the NRA (the SSO does not permit early
    retirement) then rising to 1 -- ILO-PENS ``RACT = DOUT * Retrate(x,t)``.
    """
    ages = INS_AGES.astype(float)
    rate = np.zeros(N_AGE_INS, dtype=float)
    at_or_after = ages >= nra
    rate[at_or_after] = np.clip(0.55 + 0.09 * (ages[at_or_after] - nra), 0.0, 1.0)
    rate[ages >= AGE_MAX_INS] = 1.0
    return np.repeat(np.repeat(rate[None, :], N_SEX, axis=0)[:, :, None], n_t, axis=2)


def make_leaving_rate_from_nra_path(nra_path: np.ndarray) -> np.ndarray:
    """Leaving-rate surface whose retirement spike tracks a year-varying NRA.

    Use this alongside ``SchemeConfig.nra_path`` in a retirement-age reform
    scenario, so that exit behaviour moves with the statutory age instead of
    staying anchored at 55.

    Parameters
    ----------
    nra_path
        Normal retirement age by projection year, shape ``(n_t,)``.

    Returns
    -------
    np.ndarray
        Shape ``(2, 55, n_t)``.
    """
    nra_path = np.asarray(nra_path)
    n_t = len(nra_path)
    out = np.zeros((N_SEX, N_AGE_INS, n_t), dtype=float)
    for t, nra in enumerate(nra_path):
        out[:, :, t] = make_leaving_rate(1, nra=int(nra))[:, :, 0]
    return out


def make_retirement_rate_from_nra_path(nra_path: np.ndarray) -> np.ndarray:
    """Retirement-rate surface whose onset tracks a year-varying NRA.

    Parameters
    ----------
    nra_path
        Normal retirement age by projection year, shape ``(n_t,)``.

    Returns
    -------
    np.ndarray
        Shape ``(2, 55, n_t)``.
    """
    nra_path = np.asarray(nra_path)
    n_t = len(nra_path)
    out = np.zeros((N_SEX, N_AGE_INS, n_t), dtype=float)
    for t, nra in enumerate(nra_path):
        out[:, :, t] = make_retirement_rate(1, nra=int(nra))[:, :, 0]
    return out


def make_reentry_rate(n_t: int) -> np.ndarray:
    """Probability an inactive member re-enters the active population.

    Shape ``(2, 55, n_t)``.  Re-entry is common in the 20s-40s and collapses
    near retirement.
    """
    ages = INS_AGES.astype(float)
    rate = 0.22 * np.exp(-0.5 * ((ages - 30.0) / 12.0) ** 2)
    rate[ages >= 55] = 0.0
    out = np.repeat(rate[None, :], N_SEX, axis=0)
    return np.repeat(out[:, :, None], n_t, axis=2)


def make_new_entrant_share(n_t: int) -> np.ndarray:
    """Share of the required inflow that is new (rather than re-) entrants.

    Shape ``(55, n_t)``.  Almost all inflow at the youngest ages is genuinely
    new; at older ages it is dominated by re-entry from the inactive pool.
    """
    ages = INS_AGES.astype(float)
    share = np.clip(1.0 - (ages - AGE_MIN_INS) / 28.0, 0.05, 1.0)
    return np.repeat(share[:, None], n_t, axis=1)


def make_salary_scale_monthly() -> np.ndarray:
    """Average monthly insurable earnings by sex, income band and age.

    Shape ``(2, 3, 55)``, unit THB/month.  The all-band average is shaped on
    ILO/TF/THAILAND/R.40 Table A2.11 (rising from ~THB 6,200 at 15-19 to a peak
    ~THB 10,100 at 30-34, then flat/declining), uprated to a base-year level,
    and split into the ILO-PENS low / medium / high percentile bands.
    """
    avg_5yr = {
        (15, 19): (6_347, 6_073), (20, 24): (8_039, 8_113),
        (25, 29): (9_695, 9_706), (30, 34): (10_232, 10_014),
        (35, 39): (10_242, 9_703), (40, 44): (10_178, 9_437),
        (45, 49): (9_955, 9_125), (50, 54): (9_635, 8_802),
        (55, 59): (9_228, 8_366), (60, 64): (8_252, 7_518),
        (65, 69): (7_561, 7_130),
    }
    uprate = 1.55  # 2013 -> base-year nominal wage level; replace with real data

    avg = np.zeros((N_SEX, N_AGE_INS), dtype=float)
    for (lo, hi), (m, f) in avg_5yr.items():
        idx = slice(lo - AGE_MIN_INS, hi - AGE_MIN_INS + 1)
        avg[0, idx] = m * uprate
        avg[1, idx] = f * uprate
    avg = _smooth(avg, passes=1)

    # Band multipliers: mean of the bottom-30 / middle-40 / top-30 percentiles
    # of a lognormal earnings distribution with sigma ~ 0.55.
    band_multiplier = np.array([0.58, 0.92, 1.55])
    sal = avg[:, None, :] * band_multiplier[None, :, None]
    return sal


def make_income_level_shares() -> np.ndarray:
    """Share of members in each income band, shape ``(3,)``.

    Structural, by the ILO-PENS percentile definition of the bands.
    """
    return np.array([0.30, 0.40, 0.30], dtype=float)


def _gamma_pdf_on_grid(scale: float, shape_k: float, max_credit: int) -> np.ndarray:
    """Discretised, truncated gamma density over credits ``0 .. max_credit``."""
    credits = np.arange(N_CREDIT, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pdf = (shape_k - 1.0) * np.log(np.maximum(credits, 1e-9)) - credits / scale
    pdf = np.exp(log_pdf - np.nanmax(log_pdf))
    if N_CREDIT > 1:
        pdf[0] = pdf[1]  # the gamma is singular at 0; hold the neighbouring value
    pdf[max_credit + 1:] = 0.0
    total = pdf.sum()
    return pdf / total if total > 0 else pdf


def build_credit_distribution(
    members_by_age: np.ndarray,
    mean_credit_by_age: np.ndarray,
    dispersion: float = 0.65,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> np.ndarray:
    """Spread a marginal member count over the accumulated-credit axis.

    Use this when you have SSO member counts by age (the shape of Table A2.6)
    and *average* past contribution years by age (Table A2.13), but not the
    full joint distribution.

    Credit within an age cell is modelled as a discretised gamma truncated at
    ``min(CREDIT_MAX, age - AGE_MIN_INS)`` -- you cannot have more credit than
    years since the minimum age of coverage.  Truncation biases the mean
    downwards, so the gamma's scale is solved by bisection until the
    **realised** mean of the truncated, discretised distribution matches the
    mean you supplied.  The realised mean is monotone increasing in the scale,
    so the bisection always converges.

    Parameters
    ----------
    members_by_age
        Member counts, shape ``(N_AGE_INS,)``.
    mean_credit_by_age
        Average accumulated credit in years, shape ``(N_AGE_INS,)``.
    dispersion
        Coefficient of variation of the credit distribution within an age cell.
        Only affects the *spread*; the mean is matched exactly regardless.
    tol
        Convergence tolerance on the realised mean, in years of credit.
    max_iter
        Maximum bisection steps.

    Returns
    -------
    np.ndarray
        Shape ``(N_AGE_INS, N_CREDIT)``.  Each row sums to ``members_by_age[i]``
        and has mean credit equal to ``mean_credit_by_age[i]``, except where
        that mean is physically impossible (it exceeds ``age - 15``), in which
        case it is clamped to the maximum attainable and all mass sits at the
        top attainable credit.

    Notes
    -----
    The *marginals* you supply are reproduced exactly; the shape within a cell
    is an assumption.  If the SSO gives you the full joint distribution, load
    it directly into ``membership.active_base`` instead.
    """
    members_by_age = np.asarray(members_by_age, dtype=float)
    mean_credit_by_age = np.asarray(mean_credit_by_age, dtype=float)
    out = np.zeros((N_AGE_INS, N_CREDIT), dtype=float)
    credits = np.arange(N_CREDIT, dtype=float)
    shape_k = max(1.0 / (dispersion**2), 0.5)

    for i, (n_members, target) in enumerate(zip(members_by_age, mean_credit_by_age)):
        if n_members <= 0:
            continue
        max_credit = int(min(CREDIT_MAX, i))  # age index i == age - AGE_MIN_INS
        if max_credit <= 0 or target <= 0:
            out[i, 0] = n_members
            continue
        if target >= max_credit:
            out[i, max_credit] = n_members  # mean is unattainable; clamp
            continue

        # Bisect on the gamma scale until the truncated mean hits the target.
        lo, hi = 1e-6, max(target, 1.0) * 50.0
        pdf = _gamma_pdf_on_grid(hi, shape_k, max_credit)
        if float(pdf @ credits) < target:  # target above the attainable ceiling
            out[i, max_credit] = n_members
            continue

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            pdf = _gamma_pdf_on_grid(mid, shape_k, max_credit)
            realised = float(pdf @ credits)
            if abs(realised - target) <= tol:
                break
            if realised < target:
                lo = mid
            else:
                hi = mid
        out[i] = n_members * pdf
    return out


def make_active_base() -> np.ndarray:
    """Base-year active members by sex, income, age, credit.

    Shape ``(2, 3, 55, 51)``, unit persons.  Built from the Table A2.6 age x sex
    marginal and the Table A2.13 average-past-service profile, scaled up to a
    base-year total of ~13 million insured.
    """
    members_5yr = {
        (15, 19): (264_228, 226_793), (20, 24): (894_545, 984_653),
        (25, 29): (1_109_136, 1_236_808), (30, 34): (1_148_192, 1_208_599),
        (35, 39): (940_851, 944_457), (40, 44): (728_150, 729_303),
        (45, 49): (555_243, 542_971), (50, 54): (367_455, 343_183),
        (55, 59): (199_931, 163_636), (60, 64): (75_374, 57_126),
        (65, 69): (23_613, 16_854),
    }
    credit_5yr = {
        (15, 19): (1.3, 1.3), (20, 24): (2.4, 2.5), (25, 29): (4.8, 5.3),
        (30, 34): (8.0, 8.9), (35, 39): (9.9, 10.7), (40, 44): (10.4, 11.0),
        (45, 49): (10.5, 11.0), (50, 54): (10.6, 11.1), (55, 59): (10.7, 11.1),
        (60, 64): (12.0, 12.0), (65, 69): (13.4, 13.3),
    }
    growth_since_2013 = 1.03  # scheme grew modestly; replace with SSO actuals

    members = np.zeros((N_SEX, N_AGE_INS), dtype=float)
    credits = np.zeros((N_SEX, N_AGE_INS), dtype=float)
    for (lo, hi), (m, f) in members_5yr.items():
        idx = slice(lo - AGE_MIN_INS, hi - AGE_MIN_INS + 1)
        width = hi - lo + 1
        members[0, idx] = m / width * growth_since_2013
        members[1, idx] = f / width * growth_since_2013
    for (lo, hi), (m, f) in credit_5yr.items():
        idx = slice(lo - AGE_MIN_INS, hi - AGE_MIN_INS + 1)
        credits[0, idx] = m
        credits[1, idx] = f

    shares = make_income_level_shares()
    active = np.zeros((N_SEX, N_INC, N_AGE_INS, N_CREDIT), dtype=float)
    for s in range(N_SEX):
        by_credit = build_credit_distribution(members[s], credits[s])
        for j in range(N_INC):
            active[s, j] = by_credit * shares[j]
    return active


def make_inactive_base(active_base: np.ndarray, ratio: float = 0.32) -> np.ndarray:
    """Base-year inactive members, shape ``(2, 3, 55, 51)``.

    Head count is taken as a fraction of the active stock; average past service
    follows ILO/TF/THAILAND/R.40 Table A2.14, which shows 1-4 years for the
    inactive population -- far below the active average, because members who
    stop contributing tend to be the ones who never accumulated much.

    Built through :func:`build_credit_distribution` so the credit axis respects
    the ``credit <= age - 15`` constraint and the stated mean is reproduced
    exactly.
    """
    mean_5yr = {
        (15, 19): 1.1, (20, 24): 1.4, (25, 29): 2.3, (30, 34): 3.4,
        (35, 39): 3.8, (40, 44): 3.6, (45, 49): 3.5, (50, 54): 3.3,
        (55, 59): 2.2, (60, 64): 2.0, (65, 69): 2.1,
    }
    mean_credit = np.zeros(N_AGE_INS, dtype=float)
    for (lo, hi), value in mean_5yr.items():
        mean_credit[lo - AGE_MIN_INS : hi - AGE_MIN_INS + 1] = value

    counts = active_base.sum(axis=(1, 3)) * ratio  # (2, 55)
    shares = make_income_level_shares()

    inactive = np.zeros_like(active_base)
    for s in range(N_SEX):
        by_credit = build_credit_distribution(counts[s], mean_credit)
        for j in range(N_INC):
            inactive[s, j] = by_credit * shares[j]
    return inactive


def make_pensioners_base() -> dict[str, np.ndarray]:
    """Base-year old-age pensioner stock and average pension.

    Returns a dict with keys ``old_age_n`` (persons) and ``old_age_amt``
    (THB per month), both shape ``(2, 101)``.

    The SSO paid its first old-age pensions in 2014, so a base year a decade
    later carries a young, fast-growing pensioner stock -- modelled here as a
    cohort just above the normal retirement age.
    """
    ages = POP_AGES.astype(float)

    old_n = np.zeros((N_SEX, N_AGE_POP), dtype=float)
    old_shape = np.exp(-0.5 * ((ages - 60.0) / 4.5) ** 2)
    old_shape[ages < 55] = 0.0
    old_n[0] = old_shape / old_shape.sum() * 620_000.0
    old_n[1] = old_shape / old_shape.sum() * 560_000.0

    old_amt = np.zeros((N_SEX, N_AGE_POP), dtype=float)
    old_amt[0] = np.where(old_n[0] > 0, 3_200.0, 0.0)
    old_amt[1] = np.where(old_n[1] > 0, 3_050.0, 0.0)

    return {"old_age_n": old_n, "old_age_amt": old_amt}


# ---------------------------------------------------------------------------
# 4. One-call config builder
# ---------------------------------------------------------------------------


def build_dummy_config(cfg: Config | None = None) -> Config:
    """Populate every ``None`` input on a :class:`Config` with dummy data.

    Fields already set by the caller are left untouched, so you can supply real
    data for some blocks and let the rest fall back to placeholders while a
    data-collection exercise is still running.

    Parameters
    ----------
    cfg
        Configuration to fill.  A fresh default :class:`Config` if omitted.

    Returns
    -------
    Config
        The same object, with all array inputs populated.
    """
    cfg = cfg or Config()
    n_t = cfg.run.n_t
    rng = np.random.default_rng(cfg.run.random_seed)  # noqa: F841  (reserved for stochastic variants)

    d, l, m, s, f = cfg.demography, cfg.labour, cfg.membership, cfg.scheme, cfg.finance

    # -- Demography --------------------------------------------------------
    if d.base_population is None:
        d.base_population = make_base_population()
    if d.mortality_qx is None:
        d.mortality_qx = make_mortality_qx(n_t, d.mortality_improvement)
    if d.fertility_asfr_base is None:
        d.fertility_asfr_base = make_fertility_asfr()
    if d.tfr_path is None:
        d.tfr_path = make_tfr_path(n_t)
    if d.net_migration_total is None:
        d.net_migration_total = make_net_migration(n_t)
    if d.migration_age_profile is None:
        d.migration_age_profile = make_migration_age_profile()

    # -- Labour / macro ----------------------------------------------------
    if l.lfpr is None:
        l.lfpr = make_lfpr(n_t)
    if l.unemployment_rate is None:
        l.unemployment_rate = make_unemployment_rate(n_t)
    if l.inflation is None:
        l.inflation = make_inflation(n_t, ultimate=l.ultimate_inflation)
    if l.real_wage_growth is None:
        l.real_wage_growth = make_real_wage_growth(n_t, ultimate=l.ultimate_real_wage_growth)
    if l.real_gdp_growth is None:
        l.real_gdp_growth = make_real_gdp_growth(n_t, ultimate=l.ultimate_real_gdp_growth)
    if l.real_investment_return is None:
        l.real_investment_return = make_real_investment_return(
            n_t, ultimate=l.ultimate_real_investment_return
        )

    # -- Membership --------------------------------------------------------
    if m.insured_mortality_qx is None:
        m.insured_mortality_qx = make_insured_mortality_qx(d.mortality_qx)
    if m.coverage_rate is None:
        m.coverage_rate = make_coverage_rate(n_t)
    if m.density is None:
        m.density = make_density(n_t)
    if m.leaving_rate is None:
        m.leaving_rate = make_leaving_rate(n_t, nra=s.normal_retirement_age)
    if m.retirement_rate is None:
        m.retirement_rate = make_retirement_rate(n_t, nra=s.normal_retirement_age)
    if m.reentry_rate is None:
        m.reentry_rate = make_reentry_rate(n_t)
    if m.new_entrant_share is None:
        m.new_entrant_share = make_new_entrant_share(n_t)
    if m.salary_scale_monthly is None:
        m.salary_scale_monthly = make_salary_scale_monthly()
    if m.income_level_shares is None:
        m.income_level_shares = make_income_level_shares()
    if m.active_base is None:
        m.active_base = make_active_base()
    if m.inactive_base is None:
        m.inactive_base = make_inactive_base(m.active_base)

    pens = make_pensioners_base()
    if m.old_age_pensioners_base is None:
        m.old_age_pensioners_base = pens["old_age_n"]
    if m.old_age_pension_base_monthly is None:
        m.old_age_pension_base_monthly = pens["old_age_amt"]


    # -- Finance -----------------------------------------------------------
    if f.nominal_interest_rate is None:
        f.nominal_interest_rate = (1.0 + l.real_investment_return) * (1.0 + l.inflation) - 1.0

    return cfg


__all__ = [name for name in dir() if name.startswith(("make_", "build_"))]
