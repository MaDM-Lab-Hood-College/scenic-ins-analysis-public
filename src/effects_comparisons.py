"""
Provides functions that create rows for a posterior summary dataframe, or a 
posterior effects comparisons dataframe.
"""

from collections import OrderedDict
from pathlib import Path
from statistics import mode
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.summary_stats import hdi, mean_standardized_difference
from src.variables import Variable


def posterior_summary_rows(
        param_name: str,
        joint_level_data: OrderedDict[str, np.ndarray]
) -> List[Dict]:
    rows = [
        {"parameter": f"{param_name} LL"},
        {"parameter": f"{param_name} Mode"},
        {"parameter": f"{param_name} UL"}
    ]
    for descr, values in joint_level_data.items():
        joint_level_hdi = hdi(values)
        rows[0].update({descr: joint_level_hdi[0]})
        rows[1].update({descr: mode(values)})
        rows[2].update({descr: joint_level_hdi[1]})
    return rows

def effect_comparison_rows(
        param_name: str,
        comparisons_data: OrderedDict[str, np.ndarray]
) -> List[Dict]:
    rows = []
    for descr, comparison in comparisons_data.items():
        comparison_hdi = hdi(comparison)
        rows.append(
            {
                "Parameter": param_name,
                "Comparison": descr,
                "LL": comparison_hdi[0],
                "UL": comparison_hdi[1],
                "Prob >= 0": (comparison >= 0).mean(),
                "Prob < 0": (comparison < 0).mean(),
                "Mean": comparison.mean(),
                "Median": np.median(comparison),
                "Mode": mode(comparison),
                "MSD": mean_standardized_difference(comparison),
            }
        )
    return rows


def joint_level_scatterplot(
    param_name: str, 
    values: pd.DataFrame,
    axes: Variable,
    hue: Variable,
    response: Variable,
    experiment_name: str,
    savedir: Path
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    xlabel, ylabel = values[axes.column_name].unique()
    x = values[values[axes.column_name]==xlabel][response.column_name]
    y = values[values[axes.column_name]==ylabel][response.column_name]
    xy_range = min(min(x), min(y)), max(max(x), max(y))

    divline_pts = np.linspace(*xy_range, 10)

    g = sns.jointplot(
        data=values.pivot(
            index=["draw", hue.column_name],
            columns=axes.column_name,
            values=response.column_name,
        ).reset_index(),
        x=xlabel,
        y=ylabel,
        xlim=xy_range,
        ylim=xy_range,
        hue=hue.column_name,
        palette=["k", "gray"],
        s=5,
        alpha=0.5,
        # ratio=4, space=0,
        marginal_ticks=False,
    )
    g.set_axis_labels(xlabel=xlabel, ylabel=ylabel)
    title1 = f"MCMC Samples of Mean {response.semantic_name}: {experiment_name}"
    g.figure.suptitle(t=f"{title1}", fontsize=11)
    g.ax_joint.plot(divline_pts, divline_pts, ls="-", lw=0.5, c="k")
    plt.legend(markerscale=2, title=hue.semantic_name)
    plt.tight_layout()
    filename = Path(
        savedir, f"./JointLevel-{param_name}-axes={axes.column_name}-hue={hue.column_name}.png"
    )
    print(f"Saving joint level scatterplot: {filename}")
    plt.savefig(filename)
    plt.close()
