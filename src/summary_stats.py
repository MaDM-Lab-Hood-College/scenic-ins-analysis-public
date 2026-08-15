from typing import Tuple

import numpy as np


def hdi(
    samples: np.typing.NDArray, hdi_mass: float = 0.95
) -> Tuple[float, Tuple[float, float]]:
    """
    Estimated highest density interval from a unimodal distribution.

    :return: Estimated (lower_limit, upper_limit) of highest density interval
    """
    samples = np.sort(samples)
    window_width = int(hdi_mass * len(samples))
    widths = samples[window_width:] - samples[:-window_width]
    min_interval_idx_lo = int(np.argmin(widths))
    return samples[min_interval_idx_lo], samples[min_interval_idx_lo + window_width]



def mean_standardized_difference(param_diffs: np.typing.NDArray[np.float32]) -> float:
    standardized_diffs = param_diffs / param_diffs.std()
    msd = standardized_diffs.mean()
    return msd
