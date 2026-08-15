"""
Analysis classes for the KNife study.

This module contains classes for setting up and running analyses on the KNife dataset.
Classes help create output directories, define experimental variables, and manage MCMC sampling.

Classes
-------
Analysis
    Sets up the output directory (`savedir`) and defines semantics of the experimental variables.

McmcAnalysis
    Sets MCMC sampler parameters. Provides an interface for running MCMC sampling
    and analyzing the posterior samples.

Notes
-----
TODO
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Callable

import arviz as az
import numpy as np
from src.mcmc_config import McmcConfig

class Analysis(ABC):
    """
    Base class for all analyses in this module.
    Subclassed by `McmcAnalysis` and `PosteriorPredictiveAnalysis`.
    All outputs will be saved into the directory `savedir`, which is created
    if necessary.

    Notes
    -----
    All paths/filenames are set in `config.yaml`.
    """

    def __init__(self, savedir: Path):
        self.savedir = savedir
        os.makedirs(savedir, exist_ok=True)


class McmcAnalysis(Analysis):
    """
    Base class for MCMC analysis.
    Subclassed by `PcAnalysis` and `HssmAnalysis`.
    Sets MCMC sampler defaults.

    Abstract Methods
    -------
    `run_mcmc`
        Run MCMC sampling. Creates a `.nc` file holding chains and a `.csv` file holding MCMC diagnostic information.
    
    `_glm_samples`
        Construct MCMC samples for the output of the GLM predictor.
    
    `analyze_posterior`
        Analyze experimental effects and summarize the posterior on GLM parameters.

    Notes
    -----
    All paths/filenames are set in a config YAML.

    A subclass must define its own `self.inv_link_np`.
    """

    def __init__(
        self, 
        savedir: Path, 
        rng: np.random.Generator,
        mcmc_config: McmcConfig
    ):
        super().__init__(savedir)
        self.rng = rng
        self.mcmc_config = mcmc_config
        self.inv_link_np: Callable = None  # Subclass must define

    @abstractmethod
    def _glm_samples(self, idata_posterior: az.InferenceData) -> np.ndarray:
        """
        Construct MCMC samples for the output of `self`'s general linear model.
        """
        pass

    @abstractmethod
    def run_mcmc(
        self,
        output_mcmc_chains_filename: str, 
        output_mcmc_summary_filename: str
    ) -> None:
        """
        Creates `.nc` containing MCMC chains, to be loaded with ArViz.
        """
        pass

    @abstractmethod
    def analyze_posterior(
        self,
        mcmc_chains_filename: str,
        output_posterior_summary_filename: str,
        output_effects_analysis_filename: str,
    ):
        """
        Analyze experimental effects and summarize the posterior on GLM parameters.

        This method performs an analysis of the experimental effects
        on GLM parameters. It also computes a summary of the posterior distribution
        for these parameters.

        Parameters
        ----------
        mcmc_chains_filename : str
            Name of file containing MCMC samples of GLM parameters.
        output_posterior_summary_filename : str
            Name of file that will contain summary of the posterior distribution.
        output_effects_analysis_filename : str
            Name of file that will contain experimental effects analysis.

        Returns
        -------
        None

        Notes
        -----
        All filenames are set in `config.yaml`.
        """
        pass
