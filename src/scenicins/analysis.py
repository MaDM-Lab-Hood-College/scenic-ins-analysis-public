from collections import OrderedDict
from enum import StrEnum, auto
from pathlib import Path
from typing import Callable, Dict

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import scipy

from src.mcmc_config import McmcConfig
from src.analysis import McmcAnalysis
from src.effects_comparisons import (
    effect_comparison_rows,
    joint_level_scatterplot,
    posterior_summary_rows,
)
from src.prior import GLMParams, plot_prior_dbda, set_prior_dbda
from src.variables import VariableBundle, setup_variables

pytensor.config.floatX = "float32"
pytensor.config.blas__ldflags = "-llapack -lblas -lcblas"


class AnalysisType(StrEnum):
    PC = auto()


class ScenicInsAnalysis(McmcAnalysis):

    analysis_type: AnalysisType | None = None
    inv_link_np: Callable | None = None
    link_np: Callable | None = None

    def __init__(
        self,
        savedir: Path,
        rng: np.random.Generator,
        mcmc_config: McmcConfig,
        prior_params: GLMParams,
        data_path: Path,
        variables: VariableBundle,
    ):
        super().__init__(savedir, rng, mcmc_config)
        self.mcmc_config = mcmc_config
        self.prior_params = prior_params
        self.data_path = data_path
        assert len(variables.predictors) == 2
        self.variables = variables

    @classmethod
    def from_config(cls, config: Dict) -> "ScenicInsAnalysis":
        assert cls.analysis_type
        assert cls.link_np
        
        rng = np.random.default_rng(config["seed"])
        savedir = Path(config["rootdir"], config["savedir"])
        data_path = Path(
            config["rootdir"],
            config["input_paths"][f"{cls.analysis_type}_data"]
        )
        
        variables = setup_variables(
            variable_config=config["variables"],
            data_path=data_path,
            analysis_type=cls.analysis_type,
        )
        
        prior_params = set_prior_dbda(
            prior_data_path=data_path,
            variable=variables.response,
            link=cls.link_np,
        )
        plot_prior_dbda(
            prior_data_path=data_path,
            glm_params=prior_params,
            variable=variables.response,
            savedir=savedir,
            experiment_name=config["experiment_name"],
            link=cls.link_np,
        )

        return cls(
            savedir=savedir,
            rng=rng,
            mcmc_config=McmcConfig(
                cores=config["mcmc_config"]["cores"], 
                chains=config["mcmc_config"]["chains"], 
                tune=config["mcmc_config"]["tune"], 
                draws=config["mcmc_config"]["draws"]
            ),
            prior_params=prior_params,
            data_path=data_path,
            variables=variables,
        )

    def analyze_posterior(
        self,
        mcmc_chains_filename: str,
        output_posterior_summary_filename: str,
        output_effects_analysis_filename: str,
        experiment_name: str,
    ):
        assert type(self).inv_link_np, f"Subclass must define inv_link_np"
        params = self._glm_samples(
            az.from_netcdf(
                str(Path(self.savedir, mcmc_chains_filename))
            ).posterior
        )
        params = type(self).inv_link_np(params)
        self._summarize_posterior(params, output_posterior_summary_filename)      
        self._analyze_effects(
            params=params,
            output_effects_analysis_filename=output_effects_analysis_filename,
            experiment_name=experiment_name,
        )

    def _glm_samples(
        self,
        idata_posterior: az.InferenceData
    ) -> np.ndarray:
        """
        Construct MCMC samples for the output of a general linear model having 
        two predictors and a participant ID term.

        This function combines MCMC samples of terms, marginalizing out participant ID.

        ***Does not apply inverse link function.***

        Parameters
        ----------
        `idata_posterior` : Its `values` has keys `"mu0"`, `"mu1"`, `"mu2"`, `"mu3"`, `"mu1mu2"`, guaranteed by the model defined in ' `self.run_mcmc`.

        Returns
        -------
        `samples_unlinked` : ndarray, shape `(c*d, l1, l2)`, where `c` is the number of chains, `d` the number of draws, `l1` the number of levels of predictor 1, and `l2` the number of levels of predictor 2. Output of the general linear model with participant ID marginalized out.
        """
        p0, p1, p2, p3, p1p2 = [
            idata_posterior[k].values 
            for k in ("mu0", "mu1", "mu2", "mu3", "mu1mu2")
        ]
        samples = (
            # dimensions are chain, draw, predictor_1, predictor_2, pid
            p0[:, :, np.newaxis, np.newaxis, np.newaxis]  # broadcast over all
            + p1[:, :, :, np.newaxis, np.newaxis]  # broadcast over predictor_1/pid
            + p2[:, :, np.newaxis, :, np.newaxis]  # broadcast over predictor_2/pid
            + p3[:, :, np.newaxis, np.newaxis, :]  # broadcast over predictor1/predictor_2
            + p1p2[:, :, :, :, np.newaxis]  # broadcast over pid
        )
        samples = samples.mean(axis=-1)  # marginalize out pid
        return samples.reshape(-1, *(samples.shape[2:]))

    def _summarize_posterior(
        self, 
        params: np.ndarray,
        output_posterior_summary_filename: str
    ):

        x1 = self.variables.predictors[0]
        x2 = self.variables.predictors[1]
        postsum_rows = posterior_summary_rows(
            param_name=self.variables.response.semantic_name, 
            joint_level_data=OrderedDict(
                {
                    f"{x1.levels[0]}:{x2.levels[0]}": params[:, 0, 0],
                    f"{x1.levels[0]}:{x2.levels[1]}": params[:, 0, 1],
                    f"{x1.levels[1]}:{x2.levels[0]}": params[:, 1, 0],
                    f"{x1.levels[1]}:{x2.levels[1]}": params[:, 1, 1],
                }
            )
        )
        postsum_path = Path(self.savedir, output_posterior_summary_filename)
        print(f"Summary of posterior: {postsum_path}")
        pd.DataFrame(postsum_rows).to_csv(postsum_path, index=False)


    def _analyze_effects(
            self,
            params: np.ndarray,
            output_effects_analysis_filename: str,
            experiment_name: str,
    ):
        params_raw = np.copy(params)
        params = np.round(params, 3)

        x1 = self.variables.predictors[0]
        x2 = self.variables.predictors[1]
        response = self.variables.response

        # effects comparisons
        
        x1_level1 = params[:, 0, :].mean(-1)  # marginalize x2
        x1_level2 = params[:, 1, :].mean(-1)
        x2_level1 = params[:, :, 0].mean(-1)  # marginalize x1
        x2_level2 = params[:, :, 1].mean(-1)
        
        x1x2_level11 = params[:, 0, 0]
        x1x2_level12 = params[:, 0, 1]
        x1x2_level21 = params[:, 1, 0]
        x1x2_level22 = params[:, 1, 1]

        posteff_rows = effect_comparison_rows(
            param_name=self.variables.response.semantic_name, 
            comparisons_data=OrderedDict({
                x1.column_name: x1_level1 - x1_level2,
                x2.column_name: x2_level1 - x2_level2,
                f"{x2.column_name} @ {x1.levels[0]}": x1x2_level11 - x1x2_level12,
                f"{x2.column_name} @ {x1.levels[1]}": x1x2_level21 - x1x2_level22,
                f"{x1.column_name} vs {x2.column_name}": (x1x2_level11 - x1x2_level12) - (x1x2_level21 - x1x2_level22),
                # TODO condition on StCl
                f"{x1.column_name} @ {x2.levels[0]}": x1x2_level11 - x1x2_level21,
                f"{x1.column_name} @ {x2.levels[1]}": x1x2_level12 - x1x2_level22,
                f"{x2.column_name} vs {x1.column_name}": (x1x2_level11 - x1x2_level21) - (x1x2_level12 - x1x2_level22),
                
            })
        )

        # joint level visualization
        
        draw = np.arange(params_raw.shape[0])
        
        x1x2_level11_df = pd.DataFrame()
        x1x2_level11_df[response.column_name] = params_raw[:, 0, 0]
        x1x2_level11_df["draw"] = draw
        x1x2_level11_df[x1.column_name] = x1.levels[0]
        x1x2_level11_df[x2.column_name] = x2.levels[0]
        
        x1x2_level12_df = pd.DataFrame()
        x1x2_level12_df[response.column_name] = params_raw[:, 0, 1]
        x1x2_level12_df["draw"] = draw
        x1x2_level12_df[x1.column_name] = x1.levels[0]
        x1x2_level12_df[x2.column_name] = x2.levels[1]
        
        x1x2_level21_df = pd.DataFrame()
        x1x2_level21_df[response.column_name] = params_raw[:, 1, 0]
        x1x2_level21_df["draw"] = draw
        x1x2_level21_df[x1.column_name] = x1.levels[1]
        x1x2_level21_df[x2.column_name] = x2.levels[0]
        
        x1x2_level22_df = pd.DataFrame()
        x1x2_level22_df[response.column_name] = params_raw[:, 1, 1]
        x1x2_level22_df["draw"] = draw
        x1x2_level22_df[x1.column_name] = x1.levels[1]
        x1x2_level22_df[x2.column_name] = x2.levels[1]

        joint_level_scatterplot(
            param_name=self.variables.response.semantic_name,
            values=pd.concat(
                (x1x2_level11_df, x1x2_level12_df, x1x2_level21_df, x1x2_level22_df),
                ignore_index=True,
            ),
            axes=self.variables.predictors[1],
            hue=self.variables.predictors[0],
            response=self.variables.response,
            experiment_name=experiment_name,
            savedir=self.savedir
        )

        posteff_path = Path(self.savedir, output_effects_analysis_filename)
        print(f"Experimental effects analysis: {posteff_path}")
        pd.DataFrame(posteff_rows).to_csv(posteff_path, index=False)


class PcAnalysis(ScenicInsAnalysis):
    """
    Analysis of mean proportion correct using a beta likelihood and
    normal priors on GLM terms.
    """

    analysis_type = AnalysisType.PC
    inv_link_np  = scipy.special.expit
    link_np = scipy.special.logit

    def __init__(
        self,
        savedir: Path,
        rng: np.random.Generator,
        mcmc_config: McmcConfig,
        prior_params: GLMParams,
        data_path: Path,
        variables: VariableBundle,
    ):
        super().__init__(savedir, rng, mcmc_config, prior_params, data_path, variables)

    @classmethod
    def from_config(cls, config: Dict) -> "PcAnalysis":
        analysis = super().from_config(config)
        analysis.prior_params.sd_observed.max = min(
            0.49, analysis.prior_params.sd_observed.max
        )
        return analysis
    
    def run_mcmc(
        self,
        output_mcmc_chains_filename: str,
        output_mcmc_summary_filename: str
    ):
        df = pd.read_csv(self.data_path)

        predictor_1 = self.variables.predictors[0]  # Ins
        level_to_idx = {k: v for v, k in enumerate(predictor_1.levels)}
        x1 = np.vectorize(lambda x: level_to_idx[x])(df[predictor_1.column_name].values)
        n_x1_levels = len(predictor_1.levels)
        
        predictor_2 = self.variables.predictors[1]  # Cond
        level_to_idx = {k: v for v, k in enumerate(predictor_2.levels)}
        x2 = np.vectorize(lambda x: level_to_idx[x])(df[predictor_2.column_name].values)
        n_x2_levels = len(predictor_2.levels)

        pid = df[self.variables.pid.column_name].to_numpy()
        n_pid = len(self.variables.pid.levels)

        # ceiling effects in pix experiment
        y = df[self.variables.response.column_name].to_numpy() - 0.0001

        with pm.Model() as model:

            # baseline
            mu0 = pm.Normal(
                name="mu0",
                mu=self.prior_params.baseline.mu,
                sigma=self.prior_params.baseline.sigma,
            )

            # main effects
            mu1 = pm.Normal(
                name="mu1",
                mu=0,
                sigma=pm.math.sqrt(
                    pm.InverseGamma(
                        name="mu1_var",
                        alpha=self.prior_params.deflection.inverse_gamma_params.alpha,
                        beta=self.prior_params.deflection.inverse_gamma_params.beta,
                    )
                ),
                shape=n_x1_levels,
            )
            
            mu2 = pm.Normal(
                name="mu2",
                mu=0,
                sigma=pm.math.sqrt(
                    pm.InverseGamma(
                        name="mu2_var",
                        alpha=self.prior_params.deflection.inverse_gamma_params.alpha,
                        beta=self.prior_params.deflection.inverse_gamma_params.beta,
                    )
                ),
                shape=n_x2_levels,
            )

            mu3 = pm.Normal(
                name="mu3",
                mu=0,
                sigma=pm.math.sqrt(
                    pm.InverseGamma(
                        name="mu3_var",
                        alpha=self.prior_params.deflection.inverse_gamma_params.alpha,
                        beta=self.prior_params.deflection.inverse_gamma_params.beta,
                    )
                ),
                shape=n_pid,
            )
            
            # two-way interactions
            mu1mu2 = pm.Normal(
                name="mu1mu2",
                mu=0,
                sigma=pm.math.sqrt(
                    pm.InverseGamma(
                    name="mu1mu2_var",
                    alpha=self.prior_params.deflection.inverse_gamma_params.alpha,
                    beta=self.prior_params.deflection.inverse_gamma_params.beta,
                )
                ),
                shape=(n_x1_levels, n_x2_levels),
            )

            # general linear model with logistic link function
            mu_latent = mu0 + mu1[x1] + mu2[x2] + mu3[pid] + mu1mu2[x1, x2]
            mu = pm.math.invlogit(mu_latent)

            # likelihood
            pc_observed_stdev = pm.Uniform(
                "pc_obs_stdev",
                lower=self.prior_params.sd_observed.min,
                upper=self.prior_params.sd_observed.max,
            )
            kappa_plus_1 = mu * (1 - mu) / pc_observed_stdev**2
            pc_obs = pm.Beta(
                name="pc_obs",
                alpha=mu * kappa_plus_1,
                beta=(1 - mu) * kappa_plus_1,
                observed=y
            )

            idata = pm.sample(
                cores=self.mcmc_config.cores,
                chains=self.mcmc_config.chains,
                tune=self.mcmc_config.tune,
                draws=self.mcmc_config.draws,
                nuts_sampler="pymc",
                random_seed=self.rng,
                return_inferencedata=True,
            )

        idata.to_netcdf(Path(self.savedir, output_mcmc_chains_filename))

        # summarize chains for parameters of interest
        for vn in ("mu0", "mu1", "mu2", "mu3", "mu1mu2", "pc_obs_stdev"):
            az.plot_trace(data=idata, var_names=(vn,))
            plt.title(f"Traces for {vn}")
            plt.savefig(Path(self.savedir, f"trace-{vn}.png"))
            plt.close()
        summary = pm.stats.summary(idata, hdi_prob=0.95)
        summary.to_csv(Path(self.savedir, output_mcmc_summary_filename))
