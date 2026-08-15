from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy

from src.variables import Variable


QUANTILES = np.linspace(0.05, 0.95, 19)

N_SAMPLES_DEFLECTION_VAR = 10**4


@dataclass(frozen=True, kw_only=True)
class NormalParams:
    mu: float
    sigma: float


@dataclass(frozen=True, kw_only=True)
class InverseGammaParams:
    """Shape/scale parameters for a SciPy/PyMC inverse gamma distribution."""
    alpha: float
    """shape"""
    beta: float
    """scale"""


@dataclass(frozen=True, kw_only=True)
class DeflectionParams:
    """Normal distribution having mean 0 and unkwown variance that follows an
    inverse gamma distribution."""
    mean: float = 0
    inverse_gamma_params: InverseGammaParams


@dataclass(kw_only=True)
class UniformParams:
    min: float
    max: float


@dataclass(frozen=True, kw_only=True)
class GLMParams:
    """Parameters for Kruschke-style GLM."""
    baseline: NormalParams
    """Latent scale."""
    deflection: DeflectionParams
    """Latent scale."""
    sd_observed: UniformParams
    """Observed scale; Kruschke's `ySigma`."""


def inv_gamma_alpha_beta_from_mode_std(
    mode: float,
    std: float,
) -> InverseGammaParams:
    """Compute alpha (shape) and beta (scale) parameters for a SciPy/PyMC
    inverse gamma distribution.Based on Kruschke DBDA2e p.238.
    
    Parameters
    ----------
    `mode` : float
        Mode of observed data.
    `std` : float
        Standard deviation of the observed data.

    Returns
    -------
    InverseGammaParams with values cast to `np.float32`.
    """
    rate = (mode + (mode**2 + 4 * std**2)**0.5) / (2 * std**2)
    shape = 1 + mode * rate
    return InverseGammaParams(
        alpha=np.float32(shape),
        beta=np.float32(1 / rate)
    )


def inv_gamma_alpha_beta_from_mean_var(
    mean: float,
    variance: float
) -> InverseGammaParams:
    """Compute alpha (shape) and beta (scale) parameters for a SciPy/PyMC
    inverse gamma distribution.

    Parameters
    ----------
    `mean` : float
        Mean of the inverse gamma distribution.
    `variance` : float
        Variance of the inverse gamma distribution.

    Returns
    -------
    InverseGammaParams with values cast to `np.float32`.
    """
    alpha = (mean / variance) ** 2 + 2
    beta = mean * (alpha - 1)
    return InverseGammaParams(alpha=np.float32(alpha), beta=np.float32(beta))


def set_prior_dbda(
    prior_data_path: Path,
    variable: Variable,
    link: Callable = lambda x: x,
) -> GLMParams:
    """
    Based on Kruschke (DBDA2e, p560): noncommittal on the appropriate scales.

    Parameters
    ----------
    `prior_data_path` : Points to CSV.

    `column` : Key into dataframe read from CSV.
    """
    df = pd.read_csv(prior_data_path)
    assert variable.column_name in df.columns
    
    obs = df[variable.column_name].to_numpy()
    obs_latent: np.ma.MaskedArray = np.ma.masked_invalid(link(obs))
    obs_latent_std = obs_latent.std()

    inverse_gamma_params = inv_gamma_alpha_beta_from_mode_std(
        mode=obs_latent_std/2,
        std=obs_latent_std*2,
    )

    normal_params = NormalParams(
        mu=np.float32(obs_latent.mean()),  # appropriate scale
        sigma=np.float32(obs_latent_std * 5),  # noncommittal
    )

    glm_params = GLMParams(
        baseline=normal_params,
        sd_observed=UniformParams(  # appropriate scale, noncommittal
            min=np.float32(obs_latent_std/100),
            max=np.float32(obs_latent_std*10),
        ),
        deflection=DeflectionParams(inverse_gamma_params=inverse_gamma_params)
    )

    return glm_params


def plot_prior_dbda(
    prior_data_path: Path,
    glm_params: GLMParams,
    variable: Variable,
    savedir: Path,
    experiment_name: str,
    link: Callable = lambda x: x,
):
    df = pd.read_csv(prior_data_path)
    assert variable.column_name in df.columns
    
    normal_params: NormalParams = glm_params.baseline
    _samples = scipy.stats.norm(
        loc=normal_params.mu,
        scale=normal_params.sigma
    ).rvs(size=1_000)
    x_pdf_intercept = np.linspace(_samples.min(), _samples.max(), 100)
    
    inverse_gamma_params: InverseGammaParams = glm_params.deflection.inverse_gamma_params
    _samples = scipy.stats.invgamma(
        a=inverse_gamma_params.alpha,
        scale=inverse_gamma_params.beta
    ).rvs(size=1_000)
    x_pdf_deflection = np.linspace(_samples.min(), 0.9 * _samples.max(), 100)
    
    _plot_dbda_prior(
        data=np.ma.masked_invalid(link(df[variable.column_name].to_numpy())),
        x_pdf_intercept=x_pdf_intercept,
        y_pdf_intercept=scipy.stats.norm.pdf(
            x_pdf_intercept, 
            loc=glm_params.baseline.mu,
            scale=glm_params.baseline.sigma
        ),
        x_pdf_deflection=x_pdf_deflection,
        y_pdf_deflection=scipy.stats.invgamma.pdf(
            x_pdf_deflection,
            a=glm_params.deflection.inverse_gamma_params.alpha,
            scale=glm_params.deflection.inverse_gamma_params.beta,
        ),
        param_name=variable.semantic_name,
        savedir=savedir,
        experiment_name=experiment_name,
        filename=f"prior {variable.semantic_name}.png",
    )


def _plot_dbda_prior( 
    data: np.ndarray,
    x_pdf_intercept: np.ndarray,
    y_pdf_intercept: np.ndarray,
    x_pdf_deflection: np.ndarray,
    y_pdf_deflection: np.ndarray,
    param_name: str,
    savedir: Path,
    experiment_name: str,
    filename: str,
):
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))
    title = f"Prior for GLM Predictor of {param_name}: {experiment_name}"
    fig.suptitle(title)

    ax[0].set_title("Prior Distribution of Values for Intercept Term")
    ax[0].set_xlabel("Value of Intercept Term")
    ax[0].set_ylabel("Prior Density")
    ax[0].hist(data, bins="auto", density=True, label="Data", color="gray")
    ax[0].plot(x_pdf_intercept, y_pdf_intercept, label="Prior PDF", c="k")
    ax[0].legend()

    ax[1].set_title(f"Prior Distribution of Variance for Deflection Terms")
    ax[1].set_xlabel(f"Value of Variance")
    ax[1].set_ylabel("Prior Density")
    ax[1].plot(x_pdf_deflection, y_pdf_deflection, label="Prior PDF", c="k")
    ax[1].legend()

    fig.tight_layout()
    
    savepath = Path(savedir, filename)
    os.makedirs(savedir, exist_ok=True)
    print(f"Prior plot for {param_name} saved to {savepath}")
    plt.savefig(savepath)
    plt.close()


def plot_informative_prior(
    obs_latent: np.ndarray,
    x_pdf_intercept: np.ndarray,
    y_pdf_intercept: np.ndarray,
    deflection_samples: np.ndarray,
    x_pdf_deflection: np.ndarray,
    y_pdf_deflection: np.ndarray,
    param_name: str,
    savedir: Path,
    filename: str,
):
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    fig.suptitle(f"Prior for GLM Predictor of {param_name}")

    ax[0, 0].set_title("Distribution of Values for Intercept Term")
    ax[0, 0].set_xlabel("Value of Intercept Term")
    ax[0, 0].set_ylabel("Prior Density")
    ax[0, 0].hist(obs_latent, bins="auto", density=True, label="Data", color="gray")
    ax[0, 0].plot(x_pdf_intercept, y_pdf_intercept, label="Prior PDF", c="k")
    ax[0, 0].legend()

    ax[1, 0].set_title(f"Samples of Variance of Deflection")
    ax[1, 0].set_xlabel(f"Value of Variance")
    ax[1, 0].set_ylabel("Prior Density")
    ax[1, 0].hist(
        deflection_samples, bins="auto", density=True, label="Data", color="gray"
    )
    ax[1, 0].plot(x_pdf_deflection, y_pdf_deflection, label="Prior PDF", c="k")
    ax[1, 0].legend()

    fig.tight_layout()
    savepath = Path(savedir, filename)
    print(f"Prior plot for {param_name} saved to {savepath}")
    plt.savefig(savepath)
    plt.close()


def fit_ppf(quantiles, scores, candidate_ppf, params_guess):
    fit_params_ppf, _ = scipy.optimize.curve_fit(
        candidate_ppf, xdata=quantiles, ydata=scores, p0=params_guess
    )
    return fit_params_ppf


def set_prior_ecdf(
    prior_data_path: Path,
    column: str,
    rng:np.random.Generator,
    inv_link: Callable = lambda x: x,
) -> GLMParams:
    ...
