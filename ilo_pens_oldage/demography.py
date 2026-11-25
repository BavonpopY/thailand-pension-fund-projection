"""
Module 1 -- Demographic projection (cohort-component method).
=============================================================

Implements the standard ILO-POP cohort-component recursion that feeds the rest
of ILO-PENS:

.. math::
    P_{a+1,\\,t+1} = P_{a,t}\\,\\bigl(1 - q_{a,t}\\bigr) + M_{a,t}

with

*   births added at age 0 from age-specific fertility rates scaled to the
    assumed total fertility rate,
*   an open-ended terminal age group at :data:`~ilo_pens_th.config.AGE_MAX_POP`,
*   net migration distributed over a fixed sex/age profile.

Migration is applied at mid-year (half exposed to mortality), which is the
convention that keeps the annual population balance identity exact:

.. math::
    P_{t+1} = P_t - D_t + B_t + M_t
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AGE_MAX_POP, Config, N_AGE_POP, N_SEX

# Standard fertile age range used to rescale the ASFR profile to the target TFR.
FERTILE_AGE_MIN = 15
FERTILE_AGE_MAX = 49


@dataclass
class DemographyResult:
    """Output of the national population projection.

    Attributes
    ----------
    population
        Population by sex, age and year, shape ``(2, 101, n_t)``, persons.
    births
        Total births by year, shape ``(n_t,)``.
    deaths
        Total deaths by year, shape ``(n_t,)``.  ``deaths[t]`` are deaths
        occurring between year ``t-1`` and year ``t``; ``deaths[0] = 0``.
    net_migration
        Net migration by year, shape ``(n_t,)``.
    old_age_dependency_ratio
        Population 65+ divided by population 15-64, shape ``(n_t,)``.
    """

    population: np.ndarray
    births: np.ndarray
    deaths: np.ndarray
    net_migration: np.ndarray
    old_age_dependency_ratio: np.ndarray

    @property
    def total(self) -> np.ndarray:
        """Total population by year, shape ``(n_t,)``."""
        return self.population.sum(axis=(0, 1))

    @property
    def working_age(self) -> np.ndarray:
        """Population aged 15-64 by year, shape ``(n_t,)``."""
        return self.population[:, 15:65, :].sum(axis=(0, 1))


def _scale_asfr_to_tfr(asfr_base: np.ndarray, tfr_target: float) -> np.ndarray:
    """Rescale an age-specific fertility profile to hit a target TFR.

    Parameters
    ----------
    asfr_base
        Base-year ASFR by single age, shape ``(101,)``.
    tfr_target
        Desired total fertility rate.

    Returns
    -------
    np.ndarray
        Scaled ASFR, shape ``(101,)``.
    """
    base_tfr = asfr_base[FERTILE_AGE_MIN : FERTILE_AGE_MAX + 1].sum()
    if base_tfr <= 0:
        return np.zeros_like(asfr_base)
    return asfr_base * (tfr_target / base_tfr)


def project_population(cfg: Config) -> DemographyResult:
    """Run the cohort-component national population projection.

    Parameters
    ----------
    cfg
        Fully populated configuration (see
        :func:`ilo_pens_th.dummy_data.build_dummy_config`).

    Returns
    -------
    DemographyResult

    Notes
    -----
    The recursion is exact in the sense that
    ``P[t+1].sum() == P[t].sum() - deaths[t+1] + births[t+1] + migration[t+1]``
    up to floating-point error; :func:`ilo_pens_th.validation.check_population_balance`
    asserts this.
    """
    d = cfg.demography
    n_t = cfg.run.n_t

    pop = np.zeros((N_SEX, N_AGE_POP, n_t), dtype=float)
    births = np.zeros(n_t, dtype=float)
    deaths = np.zeros(n_t, dtype=float)
    migration = np.zeros(n_t, dtype=float)

    pop[:, :, 0] = np.asarray(d.base_population, dtype=float)

    qx = np.asarray(d.mortality_qx, dtype=float)
    asfr_base = np.asarray(d.fertility_asfr_base, dtype=float)
    tfr = np.asarray(d.tfr_path, dtype=float)
    mig_total = np.asarray(d.net_migration_total, dtype=float)
    mig_profile = np.asarray(d.migration_age_profile, dtype=float)

    male_share = d.sex_ratio_at_birth / (1.0 + d.sex_ratio_at_birth)

    for t in range(n_t - 1):
        current = pop[:, :, t]
        q_now = qx[:, :, t]

        # -- Mid-year net migration ----------------------------------------
        # Half the flow arrives before the mortality exposure (and is therefore
        # subject to it) and half after; this is the convention that keeps the
        # annual balance identity exact.
        mig = mig_total[t + 1] * mig_profile
        exposed = current + 0.5 * mig

        # -- Survivors age one year ---------------------------------------
        survivors = exposed * (1.0 - q_now) + 0.5 * mig
        died = exposed * q_now

        nxt = np.zeros_like(current)
        nxt[:, 1:AGE_MAX_POP] = survivors[:, 0 : AGE_MAX_POP - 1]
        # Open-ended terminal group absorbs survivors of the top two ages.
        nxt[:, AGE_MAX_POP] = survivors[:, AGE_MAX_POP - 1] + survivors[:, AGE_MAX_POP]

        # -- Births ---------------------------------------------------------
        asfr = _scale_asfr_to_tfr(asfr_base, float(tfr[t + 1]))
        women = exposed[1]  # female exposure
        born = float((asfr * women).sum())
        infant_q = q_now[:, 0]
        nxt[0, 0] = born * male_share * (1.0 - 0.5 * infant_q[0])
        nxt[1, 0] = born * (1.0 - male_share) * (1.0 - 0.5 * infant_q[1])
        infant_deaths = born - nxt[0, 0] - nxt[1, 0]

        pop[:, :, t + 1] = nxt
        births[t + 1] = born
        deaths[t + 1] = float(died.sum()) + infant_deaths
        migration[t + 1] = float(mig.sum())

    pop_65p = pop[:, 65:, :].sum(axis=(0, 1))
    pop_15_64 = pop[:, 15:65, :].sum(axis=(0, 1))
    oadr = np.divide(pop_65p, pop_15_64, out=np.zeros_like(pop_65p), where=pop_15_64 > 0)

    return DemographyResult(
        population=pop,
        births=births,
        deaths=deaths,
        net_migration=migration,
        old_age_dependency_ratio=oadr,
    )


__all__ = ["DemographyResult", "project_population"]
