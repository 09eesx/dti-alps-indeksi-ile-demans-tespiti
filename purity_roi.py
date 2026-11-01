from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np

AXIS_Z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
AXIS_Y = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def alignment_score(evecs: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """|dot(principal eigenvector, axis)| at every voxel.

    DIPY convention: evecs[..., :, 0] is the principal eigenvector.
    """
    v1 = evecs[..., :, 0]
    return np.abs(np.einsum("...i,i->...", v1, axis))


def purity_score_map(cl: np.ndarray, evecs: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """High where diffusion is both strongly linear and aligned with `axis`."""
    return cl * alignment_score(evecs, axis)


def greedy_local_maxima(
    score: np.ndarray,
    search_mask: np.ndarray,
    n_centers: int,
    min_sep_vox: float,
) -> List[Tuple[int, int, int]]:
    """Pick up to n_centers maxima at least min_sep_vox apart."""
    candidates = np.argwhere(search_mask)
    if candidates.size == 0:
        return []

    values = score[search_mask]
    order = np.argsort(values)[::-1]

    chosen: List[np.ndarray] = []
    for idx in order:
        pos = candidates[idx]
        if all(np.linalg.norm(pos - c) >= min_sep_vox for c in chosen):
            chosen.append(pos)
        if len(chosen) == n_centers:
            break

    return [tuple(int(v) for v in c) for c in chosen]


def sphere_mask(shape, center, radius_vox: np.ndarray) -> np.ndarray:
    grid = np.indices(shape)
    dist2 = sum(((grid[i] - center[i]) / radius_vox[i]) ** 2 for i in range(3))
    return dist2 <= 1.0


@dataclass
class MultiCenterROI:
    centers: List[Tuple[int, int, int]]
    masks: List[np.ndarray]
    union_mask: np.ndarray


def find_multi_center_roi(
    cl: np.ndarray,
    evecs: np.ndarray,
    axis: np.ndarray,
    search_mask: np.ndarray,
    voxel_size_mm: Tuple[float, float, float],
    n_centers: int = 5,
    radius_mm: float = 5.0,
    min_sep_mm: float = 6.0,
) -> MultiCenterROI:
    score = purity_score_map(cl, evecs, axis)

    radius_vox = np.array(radius_mm, dtype=np.float32) / np.array(voxel_size_mm, dtype=np.float32)
    min_sep_vox = min_sep_mm / float(np.mean(voxel_size_mm))

    centers = greedy_local_maxima(score, search_mask, n_centers, min_sep_vox)
    masks = [sphere_mask(cl.shape, c, radius_vox) & search_mask for c in centers]

    union = np.zeros_like(search_mask, dtype=bool)
    for mask in masks:
        union |= mask

    return MultiCenterROI(
        centers=centers,
        masks=masks,
        union_mask=union,
    )


def save_multicenter_roi(
    roi: MultiCenterROI,
    reference_nifti: Path,
    out_dir: Path,
    prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_img = nib.load(str(reference_nifti))

    union_img = nib.Nifti1Image(
        roi.union_mask.astype(np.uint8),
        ref_img.affine,
        ref_img.header,
    )
    nib.save(union_img, str(out_dir / f"{prefix}_union.nii.gz"))

    for idx, mask in enumerate(roi.masks, start=1):
        center_img = nib.Nifti1Image(
            mask.astype(np.uint8),
            ref_img.affine,
            ref_img.header,
        )
        nib.save(center_img, str(out_dir / f"{prefix}_center{idx}.nii.gz"))

    centers_txt = out_dir / f"{prefix}_centers.txt"
    with centers_txt.open("w", encoding="utf-8") as f:
        for idx, center in enumerate(roi.centers, start=1):
            f.write(f"center{idx}: {center}\n")
