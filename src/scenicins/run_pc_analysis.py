from pathlib import Path

import yaml

from src.scenicins.analysis import PcAnalysis


if __name__ == "__main__":
    
    for config_path in (
        Path("config", "experiment1.yaml"),
        Path("config", "experiment2.yaml"),
    ):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        experiment_name = config["experiment_name"]

        print(f"********** p(c) analysis: {experiment_name} **********")

        pc_analysis = PcAnalysis.from_config(config)

        pc_config = config["pc_analysis"]
        mcmc_chains_filename = pc_config["mcmc_chains"]
        pc_analysis.run_mcmc(
            output_mcmc_chains_filename=mcmc_chains_filename,
            output_mcmc_summary_filename=pc_config["mcmc_summary"]
        )
        pc_analysis.analyze_posterior(
            mcmc_chains_filename=mcmc_chains_filename,
            output_posterior_summary_filename=pc_config["posterior_summary"],
            output_effects_analysis_filename=pc_config["posterior_effects"],
            experiment_name=experiment_name,
        )
