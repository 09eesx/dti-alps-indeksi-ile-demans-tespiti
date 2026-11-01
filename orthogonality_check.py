"""
orthogonality_check.py
-----------------------
Quantitative replacement for visual (color-FA) ROI verification.

Instead of eyeballing the RGB-FA map to confirm that the projection-fiber ROI
and the association-fiber ROI are sampling two genuinely different fiber
populations, this checks it numerically: the dominant diffusion-tensor
orientation inside each 5mm ROI should be close to perpendicular (~90 degrees)
to the other's. It does not need to be exactly 90 -- a tolerance band is
allowed, since real fiber crossings/partial-voluming are never perfect.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def principal_direction(evecs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Representative principal-eigenvector direction inside `mask`.

    DTI eigenvectors have a sign ambiguity (v and -v describe the same axis),
    so a plain vector mean would partially cancel out even for a tightly
    clustered bundle. Averaging the sign-invariant outer product v1 @ v1^T
    over all masked voxels and taking that averaged tensor's dominant
    eigenvector recovers the cluster axis regardless of per-voxel sign flips.
    """
    v1 = evecs[mask][..., :, 0]
    if v1.shape[0] == 0:
        raise ValueError("Empty ROI mask -- cannot estimate a direction.")
    dyadic = np.einsum("ni,nj->nij", v1, v1).mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(dyadic)
    return eigvecs[:, np.argmax(eigvals)]


def angle_between_axes(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees, in [0, 90], between two undirected axes (lines)."""
    cos_angle = np.clip(np.abs(np.dot(v1, v2)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


@dataclass
class OrthogonalityResult:
    angle_deg: float
    deviation_from_90_deg: float
    dir_a: np.ndarray
    dir_b: np.ndarray
    passed: bool


def check_roi_orthogonality(
    evecs: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    tol_deg: float = 30.0,
) -> OrthogonalityResult:
    """Check that ROI A's and ROI B's dominant fiber orientations are close
    to perpendicular, within `tol_deg` of 90 degrees (not required to be
    exactly 90).
    """
    dir_a = principal_direction(evecs, mask_a)
    dir_b = principal_direction(evecs, mask_b)
    angle = angle_between_axes(dir_a, dir_b)
    deviation = abs(angle - 90.0)
    return OrthogonalityResult(
        angle_deg=angle,
        deviation_from_90_deg=deviation,
        dir_a=dir_a,
        dir_b=dir_b,
        passed=deviation <= tol_deg,
    )


# Fixed anatomical reference axis for the third ("perpendicular") direction:
# left-right, perpendicular to both the projection (S-I, ~Z) and association
# (A-P, ~Y) fiber directions -- the direction the ALPS method attributes to
# the perivascular space itself. Unlike proj/assoc, this is not measured from
# a third ROI's tensors; it is the fixed axis both measured directions are
# expected to be perpendicular to.
AXIS_PERPENDICULAR = np.array([1.0, 0.0, 0.0], dtype=np.float32)


@dataclass
class TriadOrthogonalityResult:
    angle_proj_assoc_deg: float
    angle_proj_perp_deg: float
    angle_assoc_perp_deg: float
    max_deviation_from_90_deg: float
    passed: bool


def check_triad_orthogonality(
    evecs: np.ndarray,
    proj_mask: np.ndarray,
    assoc_mask: np.ndarray,
    perp_axis: np.ndarray = AXIS_PERPENDICULAR,
    low_deg: float = 78.0,
    high_deg: float = 90.0,
) -> TriadOrthogonalityResult:
    """Full three-way QC: projection ROI's direction, association ROI's
    direction, and the fixed perpendicular (left-right) axis should all sit
    close to a mutually orthogonal triad -- i.e. every one of the three
    pairwise angles should fall in [low_deg, high_deg] (default 78-90),
    not just the projection-vs-association angle.

    This also catches a failure mode the pairwise check alone would miss:
    a projection or association ROI that has drifted onto tissue running
    left-right (e.g. crossing fibers, partial-voluming) would still show
    ~90 degrees against the *other* ROI by coincidence, but would fail
    against the fixed left-right axis.
    """
    dir_proj = principal_direction(evecs, proj_mask)
    dir_assoc = principal_direction(evecs, assoc_mask)

    angle_proj_assoc = angle_between_axes(dir_proj, dir_assoc)
    angle_proj_perp = angle_between_axes(dir_proj, perp_axis)
    angle_assoc_perp = angle_between_axes(dir_assoc, perp_axis)

    deviations = [abs(a - 90.0) for a in (angle_proj_assoc, angle_proj_perp, angle_assoc_perp)]
    max_deviation = max(deviations)
    tol_deg = 90.0 - low_deg

    return TriadOrthogonalityResult(
        angle_proj_assoc_deg=angle_proj_assoc,
        angle_proj_perp_deg=angle_proj_perp,
        angle_assoc_perp_deg=angle_assoc_perp,
        max_deviation_from_90_deg=max_deviation,
        passed=max_deviation <= tol_deg,
    )
