"""
test_orthogonality_check.py
----------------------------
Validates the angle-based ROI QC mechanism (orthogonality_check.py) using
synthetic/random tensors, before trusting it on real subject data.

No real DWI data or DIPY tensor fit is needed here: we build a fake `evecs`
volume directly (shape (X, Y, Z, 3, 3), only column 0 -- the principal
eigenvector -- populated), place two 5mm-radius spherical ROIs in it (reusing
the same sphere_mask() the real pipeline uses), and control what direction
each ROI's tensors point in. That lets us check the mechanism against known
ground truth: two ROIs built with perpendicular axes must pass, two ROIs
built with parallel axes must fail, and the check must survive DTI's
eigenvector sign ambiguity.

Run: python test_orthogonality_check.py
"""
from __future__ import annotations

import numpy as np

from orthogonality_check import check_roi_orthogonality, principal_direction, angle_between_axes
from purity_roi import sphere_mask, AXIS_Y, AXIS_Z

VOXEL_SIZE_MM = (2.0, 2.0, 2.0)
ROI_RADIUS_MM = 5.0
VOLUME_SHAPE = (32, 32, 32)


def _perpendicular_axis(axis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Any unit vector perpendicular to `axis` (used as a rotation axis)."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = np.cross(axis, helper)
    return perp / np.linalg.norm(perp)


def sample_directions_near_axis(
    axis: np.ndarray, n: int, dispersion_deg: float, rng: np.random.Generator
) -> np.ndarray:
    """n unit vectors clustered around `axis`, each tilted by a random angle
    in [0, dispersion_deg] around a random perpendicular rotation axis --
    i.e. a crude but adequate stand-in for "principal eigenvectors of real
    tensors that are approximately, but not perfectly, aligned with `axis`".
    Signs are randomly flipped to mimic DTI's eigenvector sign ambiguity.
    """
    axis = axis / np.linalg.norm(axis)
    out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        tilt = np.radians(rng.uniform(0.0, dispersion_deg))
        rot_axis = _perpendicular_axis(axis, rng)
        # rotate `rot_axis` itself by a random angle around `axis`, so the tilt
        # direction is uniformly distributed around the cone, not fixed
        spin = np.radians(rng.uniform(0.0, 360.0))
        rot_axis = (
            rot_axis * np.cos(spin)
            + np.cross(axis, rot_axis) * np.sin(spin)
            + axis * np.dot(axis, rot_axis) * (1 - np.cos(spin))
        )
        # Rodrigues' rotation formula: rotate `axis` by `tilt` around `rot_axis`
        v = (
            axis * np.cos(tilt)
            + np.cross(rot_axis, axis) * np.sin(tilt)
            + rot_axis * np.dot(rot_axis, axis) * (1 - np.cos(tilt))
        )
        v = v / np.linalg.norm(v)
        if rng.random() < 0.5:
            v = -v
        out[i] = v
    return out


def sample_random_directions(n: int, rng: np.random.Generator) -> np.ndarray:
    """n uniformly random unit vectors on the sphere (fully unaligned tensors)."""
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def build_evecs_volume(mask_a, dirs_a, mask_b, dirs_b) -> np.ndarray:
    evecs = np.zeros(VOLUME_SHAPE + (3, 3), dtype=np.float64)
    evecs[mask_a, :, 0] = dirs_a
    evecs[mask_b, :, 0] = dirs_b
    return evecs


def make_two_rois():
    radius_vox = np.array(ROI_RADIUS_MM) / np.array(VOXEL_SIZE_MM)
    mask_a = sphere_mask(VOLUME_SHAPE, (10, 16, 16), radius_vox)
    mask_b = sphere_mask(VOLUME_SHAPE, (22, 16, 16), radius_vox)
    assert not np.any(mask_a & mask_b), "ROIs must not overlap"
    return mask_a, mask_b


def test_perpendicular_axes_pass():
    """Projection ROI ~ z-axis, association ROI ~ y-axis (the real ALPS setup):
    should be recognized as close to 90 degrees and pass."""
    rng = np.random.default_rng(0)
    mask_a, mask_b = make_two_rois()
    dirs_a = sample_directions_near_axis(AXIS_Z, int(mask_a.sum()), dispersion_deg=15.0, rng=rng)
    dirs_b = sample_directions_near_axis(AXIS_Y, int(mask_b.sum()), dispersion_deg=15.0, rng=rng)
    evecs = build_evecs_volume(mask_a, dirs_a, mask_b, dirs_b)

    result = check_roi_orthogonality(evecs, mask_a, mask_b, tol_deg=30.0)
    print(f"[perpendicular] angle={result.angle_deg:.2f} deg, "
          f"deviation from 90={result.deviation_from_90_deg:.2f}, passed={result.passed}")
    assert result.passed
    assert abs(result.angle_deg - 90.0) < 15.0


def test_parallel_axes_fail():
    """Both ROIs pointing along the same axis (e.g. ROI placement bug that
    lands both spheres on the same fiber population): should fail."""
    rng = np.random.default_rng(1)
    mask_a, mask_b = make_two_rois()
    dirs_a = sample_directions_near_axis(AXIS_Z, int(mask_a.sum()), dispersion_deg=15.0, rng=rng)
    dirs_b = sample_directions_near_axis(AXIS_Z, int(mask_b.sum()), dispersion_deg=15.0, rng=rng)
    evecs = build_evecs_volume(mask_a, dirs_a, mask_b, dirs_b)

    result = check_roi_orthogonality(evecs, mask_a, mask_b, tol_deg=30.0)
    print(f"[parallel] angle={result.angle_deg:.2f} deg, "
          f"deviation from 90={result.deviation_from_90_deg:.2f}, passed={result.passed}")
    assert not result.passed
    assert result.angle_deg < 20.0


def test_near_90_within_tolerance_passes():
    """Axes 70 degrees apart (not exactly 90, but close): should still pass
    with a reasonably generous tolerance, since the user's requirement is
    'close to 90', not 'exactly 90'."""
    rng = np.random.default_rng(2)
    tilted_axis = (
        AXIS_Y * np.cos(np.radians(20.0)) + AXIS_Z * np.sin(np.radians(20.0))
    )
    tilted_axis = tilted_axis / np.linalg.norm(tilted_axis)

    mask_a, mask_b = make_two_rois()
    dirs_a = sample_directions_near_axis(AXIS_Z, int(mask_a.sum()), dispersion_deg=10.0, rng=rng)
    dirs_b = sample_directions_near_axis(tilted_axis, int(mask_b.sum()), dispersion_deg=10.0, rng=rng)
    evecs = build_evecs_volume(mask_a, dirs_a, mask_b, dirs_b)

    result = check_roi_orthogonality(evecs, mask_a, mask_b, tol_deg=30.0)
    print(f"[~70deg apart] angle={result.angle_deg:.2f} deg, "
          f"deviation from 90={result.deviation_from_90_deg:.2f}, passed={result.passed}")
    assert result.passed


def test_sign_ambiguity_does_not_break_direction_estimate():
    """DTI eigenvectors are sign-ambiguous per voxel. Randomly flipping signs
    (already done inside sample_directions_near_axis) must not change the
    recovered cluster axis or the resulting angle -- confirms the dyadic
    averaging in principal_direction() is doing its job."""
    rng_a = np.random.default_rng(3)
    rng_b = np.random.default_rng(3)  # same seed -> same tilts, only sign flips will land differently across calls
    mask_a, mask_b = make_two_rois()

    dirs_1 = sample_directions_near_axis(AXIS_Z, int(mask_a.sum()), dispersion_deg=5.0, rng=rng_a)
    dirs_2 = dirs_1.copy()
    dirs_2[::2] *= -1  # force-flip every other voxel's sign

    dir_from_1 = principal_direction(*_wrap(mask_a, dirs_1))
    dir_from_2 = principal_direction(*_wrap(mask_a, dirs_2))
    angle = angle_between_axes(dir_from_1, dir_from_2)
    print(f"[sign ambiguity] angle between estimates with/without extra flips={angle:.4f} deg")
    assert angle < 1e-6


def _wrap(mask, dirs):
    evecs = np.zeros(VOLUME_SHAPE + (3, 3), dtype=np.float64)
    evecs[mask, :, 0] = dirs
    return evecs, mask


def test_fully_random_tensors_are_inconclusive():
    """Sanity/negative control: with no true alignment at all (isotropic
    random tensors in both ROIs), the mechanism should NOT reliably report
    'passed' -- it should behave like a coin flip, not silently agree every
    time. This guards against a bug where the check trivially always passes.
    """
    rng = np.random.default_rng(4)
    mask_a, mask_b = make_two_rois()

    n_trials = 200
    passes = 0
    for _ in range(n_trials):
        dirs_a = sample_random_directions(int(mask_a.sum()), rng)
        dirs_b = sample_random_directions(int(mask_b.sum()), rng)
        evecs = build_evecs_volume(mask_a, dirs_a, mask_b, dirs_b)
        result = check_roi_orthogonality(evecs, mask_a, mask_b, tol_deg=30.0)
        passes += int(result.passed)

    pass_rate = passes / n_trials
    print(f"[fully random] pass rate over {n_trials} trials = {pass_rate:.2f}")
    # with dispersion this high, the averaged direction is unstable/near-isotropic,
    # so the pass rate should be neither ~0 nor ~1 -- it must not be a rubber stamp
    assert 0.15 < pass_rate < 0.85


def run_all():
    tests = [
        test_perpendicular_axes_pass,
        test_parallel_axes_fail,
        test_near_90_within_tolerance_passes,
        test_sign_ambiguity_does_not_break_direction_estimate,
        test_fully_random_tensors_are_inconclusive,
    ]
    for t in tests:
        t()
        print(f"[OK] {t.__name__}")
    print(f"[OK] All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
