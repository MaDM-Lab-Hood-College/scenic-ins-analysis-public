from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True, kw_only=True)
class Variable:
    column_name: str
    semantic_name: str
    levels: Tuple[str, ...] | None = None


@dataclass(frozen=True, kw_only=True)
class VariableBundle:
    predictors: Tuple[Variable, ...]
    pid: Variable
    """Will be marginalized out."""
    response: Variable
    """No levels."""


def setup_variables(
    variable_config: Dict,
    data_path: Path,
    analysis_type: str,
) -> VariableBundle:
    """
    Parameters
    ----------
    `variable_config` : Read from a config file. Example:
        ```
        predictors:
            column: [Ins, Cond]
            semantic: [Instructions, "Stimulus Class"]
        pid:
            column: participant_id
            semantic: "Participant ID"
        response:
            rt:
                column: rt
                semantic: "RT (sec)"
            pc:
                column: pc
                semantic: "Proportion Correct"
        ```

    `data_path` : Levels are inferred from these data.

    `analysis_type` : Used to determine the response variable.
    """
    for key in ("predictors", "pid"):
        assert key in variable_config
        assert "column" in variable_config[key]
        assert "semantic" in variable_config[key]
    assert "response" in variable_config
    assert "column" in variable_config["response"][analysis_type]
    assert "semantic" in variable_config["response"][analysis_type]
    data = pd.read_csv(data_path)
    return VariableBundle(
        predictors=[
            Variable(
                column_name=column,
                semantic_name=semantic,
                levels=tuple(sorted(data[column].unique()))
            )
            for column, semantic in zip(
                variable_config["predictors"]["column"],
                variable_config["predictors"]["semantic"]
            )
        ],
        pid=Variable(
            column_name=variable_config["pid"]["column"],
            semantic_name=variable_config["pid"]["semantic"],
            levels=tuple(sorted(data[variable_config["pid"]["column"]].unique())),
        ),
        response = Variable(
            column_name=variable_config["response"][analysis_type]["column"],
            semantic_name=variable_config["response"][analysis_type]["semantic"]
        ),
    )
