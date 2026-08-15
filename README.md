# scenic-ins-analysis-public

A Python library supporting Bayesian data analysis for the paper 
"Speed and Accuracy Instructions Invert Effects of Stimulus Class on 2AFC Recognition"
by [M. B. Moreland](https://github.com/moreland-hood) 
and [J. M. Dusel](https://johnmdusel.github.io/).


To build a container for running analyses, execute these commands in a terminal session inside this folder:
```bash
./scripts/generate_dotenv.sh
cd .devcontainer
docker build -t scenic-ins-data-analysis .
```

You can also open this repository in a VS Code devcontainer.

To run the analyses, execute this command
```bash
./scripts/run_scenic-ins_pc_analysis.sh
```
Output files will be generated inside `analysis_results/`.

# Repo contents

After running the Docker setup and the analysis script.
```
.
├── .devcontainer
│ ├── .env
│ ├── devcontainer.json
│ ├── Dockerfile
│ └── requirements.txt
├── .gitignore
├── analysis_results
│ ├── Experiment1
│ │ ├── JointLevel-p(c)-axes=Cond-hue=Ins.png
│ │ ├── mcmc_chains_pc.nc
│ │ ├── mcmc_summary_pc.csv
│ │ ├── posterior_effects_pc.csv
│ │ ├── posterior_summary_pc.csv
│ │ ├── prior p(c).png
│ │ ├── trace-mu0.png
│ │ ├── trace-mu1.png
│ │ ├── trace-mu1mu2.png
│ │ ├── trace-mu2.png
│ │ ├── trace-mu3.png
│ │ └── trace-pc_obs_stdev.png
│ └── Experiment2
│     ├── JointLevel-p(c)-axes=Cond-hue=Ins.png
│     ├── mcmc_chains_pc.nc
│     ├── mcmc_summary_pc.csv
│     ├── posterior_effects_pc.csv
│     ├── posterior_summary_pc.csv
│     ├── prior p(c).png
│     ├── trace-mu0.png
│     ├── trace-mu1.png
│     ├── trace-mu1mu2.png
│     ├── trace-mu2.png
│     ├── trace-mu3.png
│     └── trace-pc_obs_stdev.png
├── config
│ ├── experiment1.yaml
│ └── experiment2.yaml
├── data (release pending)
│ ├── experiment1.csv
│ └── experiment2.csv
├── LICENSE
├── README.md
├── scripts
│ ├── generate_dotenv.sh
│ └── run_scenic-ins_pc_analysis.sh
└── src
    ├── analysis.py
    ├── effects_comparisons.py
    ├── mcmc_config.py
    ├── prior.py
    ├── scenicins
    │ ├── analysis.py
    │ └── run_pc_analysis.py
    ├── summary_stats.py
    └── variables.py
```

Folder `config/` contains YAML configuration files. A configuration file is what distinguishes one analysis run from another. It holds the paths to data CSVs, output folders, and other settings. 

Folder `scripts` contain the shell scripts that you'll run to do the analyses. These scripts simply execute the appropriate Python script inside the Docker container.

Folder `src` contains the Python scripts that define the various analyses.

