"""
alps_ensemble.py
-----------------
Final ALPS-index computation for each ROI strategy, plus an ensemble comparison
across strategies that produces an "agreement score" -- something none of the
existing pipelines (alps.sh, manual-ROI papers) report by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from purity_roi import MultiCenterROI


def _mean_in_mask(data: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        raise ValueError("Empty ROI mask -- cannot compute mean.")
    return float(data[mask].mean())


def alps_from_masks(Dxx: np.ndarray, Dyy: np.ndarray, Dzz: np.ndarray,
                     proj_mask: np.ndarray, assoc_mask: np.ndarray) -> float:
    """Standard ALPS formula (Taoka et al., 2017), applied to one pair of masks:
    ALPS = mean(Dxx_proj, Dxx_assoc) / mean(Dyy_proj, Dzz_assoc)
    """
    x_proj = _mean_in_mask(Dxx, proj_mask)
    x_assoc = _mean_in_mask(Dxx, assoc_mask)
    y_proj = _mean_in_mask(Dyy, proj_mask)
    z_assoc = _mean_in_mask(Dzz, assoc_mask)
    numerator = (x_proj + x_assoc) / 2.0
    denominator = (y_proj + z_assoc) / 2.0
    return numerator / denominator


@dataclass
class MultiCenterALPSResult:
    pooled_alps: float           # all 5 ROIs' voxels pooled, formula applied once
    per_pair_alps: List[float]   # ALPS computed separately per (proj_i, assoc_i) pair
    std_across_pairs: float      # built-in uncertainty estimate (NaN if <2 pairs)


def alps_from_multi_center(Dxx: np.ndarray, Dyy: np.ndarray, Dzz: np.ndarray,
                            proj_rois: MultiCenterROI,
                            assoc_rois: MultiCenterROI) -> MultiCenterALPSResult:
    # Pooled estimate: merge all voxels from all N projection ROIs into one set
    # (same for association), then apply the formula once -> a more stable mean.
    pooled = alps_from_masks(Dxx, Dyy, Dzz, proj_rois.union_mask, assoc_rois.union_mask)

    # Per-pair estimate: pair up center i (projection) with center i (association),
    # in the order they were found, as a measure of *local* consistency / spread.
    n_pairs = min(len(proj_rois.masks), len(assoc_rois.masks))
    per_pair: List[float] = []
    for i in range(n_pairs):
        try:
            per_pair.append(
                alps_from_masks(Dxx, Dyy, Dzz, proj_rois.masks[i], assoc_rois.masks[i])
            )
        except ValueError:
            continue

    std = float(np.std(per_pair)) if len(per_pair) > 1 else float("nan")
    return MultiCenterALPSResult(pooled_alps=pooled, per_pair_alps=per_pair, std_across_pairs=std)


@dataclass
class EnsembleResult:
    values: Dict[str, float]
    mean: float
    coefficient_of_variation: float   # std / mean, across strategies
    flag: str


def compare_strategies(values: Dict[str, float], cv_threshold: float = 0.10) -> EnsembleResult:
    """Compare ALPS-index estimates obtained from different ROI strategies
    (atlas-based, fixed-sphere, purity-based multi-center).

    A high coefficient of variation across methods means the methods *disagree* --
    that disagreement is itself useful information: it flags subjects/scans where
    the ALPS-index is method-dependent and should be interpreted cautiously,
    rather than silently reporting a single number as if it were unambiguous.
    """
    arr = np.array(list(values.values()), dtype=float)
    mean = float(np.mean(arr))
    cv = float(np.std(arr) / mean) if mean != 0 else float("nan")
    flag = "LOW_AGREEMENT" if cv > cv_threshold else "OK"
    return EnsembleResult(values=values, mean=mean, coefficient_of_variation=cv, flag=flag)